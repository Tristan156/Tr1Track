"""Corrected E06B candidate cache. Scoring truth is read by inspect only.

Reuses E05 square tiling, crop display and E04 full-image decoding/NMS.
No generation, downloads, GT-based reset or closed-loop feedback occurs here.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

from tracking_lab.e05 import (ROOT, EXPECTED_WEIGHT, marked_crop, module,
                             overlap, read, sha, slug, support, tile_geometry, write)

SCHEMA = 'e06b_v2_20261029'
QUERY_FRAMES = {
    'validation/01_4485_2985-4484': [802, 850, 888, 903, 918, 948],
    'validation/02_1610_0001_1500': [219, 246, 262, 277, 292, 322],
    'test/16-USV': [350, 385, 406, 421, 436, 449],
}
CODE = ['tracking_lab/e06b_v2.py', 'tracking_lab/e06b_v2_candidates.py',
        'tracking_lab/e05.py', 'tracking_lab/e05_qwen.py',
        'tracking_lab/e05_metrics.py',
        'experiments/04_boat_identity/global_candidates.py',
        'experiments/04_boat_identity/support.py']


def safe_path(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError('Asset path escapes root: ' + str(relative))
    return path


def _bbox(box):
    values = [float(v) for v in box]
    if len(values) != 4 or not all(math.isfinite(v) for v in values) or min(values[2:]) <= 0:
        raise ValueError('Invalid bbox')
    return values


def _digest_tensor(tensor):
    if hasattr(tensor, 'detach'):
        tensor = tensor.detach().cpu().numpy()
    return hashlib.sha256(tensor.tobytes()).hexdigest()


def full_image_scan(runner, rgb, init_bbox, *, sample_target, decoder=None,
                    torch_module=None, device='cpu', cap=400, ledger_path=None):
    """Run the actual network once per covering square tile, without local state updates.

    Dependencies are injectable so CPU tests can count real calls and inspect
    sample geometry rather than testing a manually assigned tile count.
    """
    import numpy as np
    if decoder is None:
        decoder = module('e06b_v2_fullimage_decode', ROOT / CODE[-2])
    height, width = rgb.shape[:2]
    scale, centers = tile_geometry(width, height, _bbox(init_bbox), cap=cap)
    template_before = _digest_tensor(runner.template)
    state_before = list(runner.state)
    boxes, scores, tile_ids, grids, tiles = [], [], [], [], []
    if device.startswith('cuda') and torch_module is not None:
        torch_module.cuda.synchronize()
    started = time.perf_counter()
    def ledger(record):
        if ledger_path is not None:
            path = Path(ledger_path); path.parent.mkdir(parents=True, exist_ok=True)
            with path.open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(record, allow_nan=False) + '\n'); stream.flush()
    for tile_id, (cx, cy) in enumerate(centers):
        pseudo = [cx - scale / 2, cy - scale / 2, scale, scale]
        patch, resize, _ = sample_target(rgb, pseudo, 5., output_sz=384)
        context = torch_module.inference_mode() if torch_module is not None else contextlib.nullcontext()
        ledger({'tile_id': tile_id, 'status': 'STARTED', 'pseudo_box_xywh': pseudo})
        try:
            with context:
                out = runner.network(template=runner.template, search=runner._tensor(patch),
                                     ce_template_mask=runner.template_mask)
        except Exception as exc:
            ledger({'tile_id': tile_id, 'status': 'ERROR', 'error': str(exc)})
            raise
        ledger({'tile_id': tile_id, 'status': 'COMPLETE'})
        b, s, t, g = decoder.decode(out, resize, pseudo, height, width, tile_id)
        boxes.append(b); scores.append(s); tile_ids.append(t); grids.append(g)
        side = math.ceil(math.sqrt(pseudo[2] * pseudo[3]) * 5.)
        x1, y1 = round(cx - side / 2), round(cy - side / 2)
        tiles.append({'tile_id': tile_id, 'center_xy': [cx, cy],
                      'pseudo_box_xywh': pseudo, 'sample_factor': 5., 'sample_side': side,
                      'actual_crop_bounds_xyxy': [x1, y1, x1 + side, y1 + side],
                      'resize_factor': float(resize), 'search_size': 384,
                      'forward_completed': True, 'decoded_grid_cells': len(s)})
    pool = decoder.nms_top(np.concatenate(boxes), np.concatenate(scores),
                           np.concatenate(tile_ids), np.concatenate(grids), 16)
    if device.startswith('cuda') and torch_module is not None:
        torch_module.cuda.synchronize()
    if template_before != _digest_tensor(runner.template) or state_before != list(runner.state):
        raise ValueError('Global scan changed template/local tracker state')
    if len(tiles) != len(centers) or not tiles:
        raise ValueError('Incomplete full-image forwards')
    return pool, {'tiles_executed': len(tiles), 'forward_count': len(tiles), 'tiles': tiles,
                  'width': width, 'height': height, 'fixed_init_scale': scale,
                  'seconds': time.perf_counter() - started, 'hann': False,
                  'template_sha256_before': template_before,
                  'template_sha256_after': _digest_tensor(runner.template),
                  'local_state_preserved': True, 'top_k_diagnostic': 16,
                  'nms_iou': .5}


def canonical_candidates(pool, sequence, source_frame):
    """Identity comes from the physical decoded tile/grid, never presentation letter."""
    out = []
    for c in pool:
        identity = f'{SCHEMA}:{sequence}:{source_frame}:{int(c["tile_id"])}:{int(c["grid_index"])}'
        candidate_id = 'p_' + hashlib.sha256(identity.encode('utf-8')).hexdigest()[:24]
        out.append({'candidate_id': candidate_id, 'bbox_xywh': _bbox(c['bbox']),
                    'native_rank': int(c['rank']), 'original_rank': int(c['rank']),
                    'tracker_score': float(c['score']), 'raw_score': float(c['score']),
                    'tile_id': int(c['tile_id']), 'grid_index': int(c['grid_index'])})
    if len({c['candidate_id'] for c in out}) != len(out):
        raise ValueError('Duplicate physical candidate')
    if [c['native_rank'] for c in out] != list(range(1, len(out) + 1)):
        raise ValueError('Noncanonical native ranking')
    return out


def label_candidates(candidates, sequence, source_frame):
    return support().shuffled_labels(candidates, sequence, source_frame, seed=20261029)


def _source_metadata(sequence, frame, root, modality):
    return {'sequence': sequence, 'modality': modality,
            'source_frame': int(frame['source_frame']),
            'source_path': str(safe_path(root, frame['path'])),
            'source_sha256': frame['sha256']}


def _crop(rgb, bbox, output, source, role):
    return {**marked_crop(rgb, bbox, output), **source, 'role': role,
            'bbox_xywh': _bbox(bbox)}


def _load_sequences(assets, boat_assets=None):
    roots = [(Path(assets).resolve(), 'inference_manifest.json', set(QUERY_FRAMES) - {'test/16-USV'})]
    if boat_assets is not None:
        roots.append((Path(boat_assets).resolve(), 'mvtd_exploration/inference_manifest.json', {'test/16-USV'}))
    sequences, manifests = [], []
    for root, relative, wanted in roots:
        manifest_path = safe_path(root, relative)
        manifest = read(manifest_path)
        manifests.append({'path': str(manifest_path), 'sha256': sha(manifest_path)})
        found = set()
        for s in manifest['sequences']:
            if s['sequence'] not in wanted:
                continue
            found.add(s['sequence'])
            modality = s.get('modality', 'RGB' if s['sequence'] == 'test/16-USV' else None)
            expected = 'RGB' if s['sequence'] == 'test/16-USV' else 'IR'
            if modality != expected:
                raise ValueError('Unexpected modality: ' + s['sequence'])
            frames = s['frames']; ids = [int(f['source_frame']) for f in frames]
            if not ids or ids != list(range(ids[0], ids[-1] + 1)):
                raise ValueError('Noncontiguous source frames')
            if not set(QUERY_FRAMES[s['sequence']]).issubset(ids) or ids[0] in QUERY_FRAMES[s['sequence']]:
                raise ValueError('Frozen queries unavailable or overlap initialization')
            _bbox(s['init_bbox_xywh'])
            for f in frames:
                if set(f) & {'exist', 'absence', 'bbox_valid', 'bbox_xywh', 'gt_rect'}:
                    raise ValueError('Scoring truth leaked into frame manifest')
                if sha(safe_path(root, f['path'])) != f['sha256']:
                    raise ValueError('Frame hash mismatch')
            from PIL import Image
            with Image.open(safe_path(root, frames[0]['path'])) as im:
                scale, centers = tile_geometry(im.width, im.height, s['init_bbox_xywh'])
                size = [im.width, im.height]
            sequences.append({**s, 'asset_root': str(root), 'modality': modality,
                              'role': 'exploratory', 'independent_evaluation': False,
                              'init_source_frame': ids[0], 'query_source_frames': QUERY_FRAMES[s['sequence']],
                              'image_size_wh': size, 'fixed_init_scale': scale,
                              'actual_global_tiles_per_query': len(centers)})
        if found != wanted:
            raise ValueError('Missing required sequences: ' + repr(wanted - found))
    return sequences, manifests


def prepare(args):
    """Verify assets, lock old query points and zero generation budget, without a model load."""
    assets = args.assets.resolve(); ostrack = args.ostrack.resolve()
    boat = getattr(args, 'boat_assets', None)
    if not getattr(args, 'include_boat', False):
        boat = None
    elif boat is None:
        raise ValueError('--include-boat requires --boat-assets pointing at E04 assets')
    sequences, manifests = _load_sequences(assets, boat)
    wrapper = ostrack / 'ostrack_runner.py'; checkpoint = ostrack / 'pytorch_model.bin'
    if sha(checkpoint) != EXPECTED_WEIGHT:
        raise ValueError('OSTrack checkpoint differs from verified E04 weight')
    code_hashes = {p: sha(ROOT / p) for p in CODE}
    try:
        commit = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
        diff = subprocess.check_output(['git', '-C', str(ROOT), 'diff', '--binary', 'HEAD'], text=True)
    except (OSError, subprocess.CalledProcessError):
        commit, diff = 'unavailable; code hashes authoritative', 'unavailable'
    external = {str(p.relative_to(ostrack)): sha(p)
                for p in sorted((ostrack / 'OSTrack-main').rglob('*'))
                if p.is_file() and p.suffix in {'.py', '.yaml', '.yml'}}
    if not external:
        raise ValueError('No OSTrack runtime source/config files found')
    plan = {'schema': SCHEMA, 'experiment': 'E06B_corrected_candidate_cache',
            'run_id': args.run.name, 'status': 'frozen_before_candidate_inference',
            'assets': str(assets), 'ostrack': str(ostrack), 'sequences': sequences,
            'inference_manifests': manifests, 'git_commit': commit,
            'code_sha256': code_hashes, 'host_wrapper_sha256': sha(wrapper),
            'checkpoint_sha256': EXPECTED_WEIGHT, 'checkpoint_bytes': checkpoint.stat().st_size,
            'external_source_sha256': external,
            'candidate_rule': 'E05 fixed-init square covering tiles; actual network on each; raw score; NMS .5; top4 of top16',
            'history_rule': 'last two strictly past local predictions; append only after current query snapshot; no GT filtering',
            'local_rule': 'initialize at first frame; track each subsequent source frame once; record current output before scoring',
            'generation_budget': 0, 'max_generations': 0,
            'global_forward_budget': sum(len(s['query_source_frames']) * s['actual_global_tiles_per_query'] for s in sequences),
            'new_training': False, 'new_downloads': False, 'no_closedloop': True,
            'exploratory': True, 'independent_evaluation': False,
            'modality_gt_policy': 'IR frames score only Anti-UAV IR annotations; RGB boat scores only matching MVTD evaluation',
            'preprocessing': 'original RGB decode of IR/RGB; E05 1.5x marked crops; no processor monkey patch',
            'query_rule': 'original E06B 12 source frames, no replacement based on outcomes; optional original boat6 separate',
            'command': list(sys.argv)}
    args.run.mkdir(parents=True, exist_ok=False)
    for relative in CODE:
        target = args.run / 'actual_code' / relative
        target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(ROOT / relative, target)
    shutil.copy2(wrapper, args.run / 'actual_code' / 'ostrack_runner.py')
    (args.run / 'actual_code' / 'working_tree.diff').write_text(diff, encoding='utf-8')
    write(args.run / 'plan.json', plan)
    (args.run / 'plan.sha256').write_text(sha(args.run / 'plan.json') + '\n', encoding='utf-8')
    print(json.dumps({'prepared': str(args.run), 'queries': sum(len(s['query_source_frames']) for s in sequences),
                      'generation_budget': 0, 'model_loaded': False}, ensure_ascii=False))


def plan_for(run):
    run = Path(run)
    if sha(run / 'plan.json') != (run / 'plan.sha256').read_text(encoding='utf-8').strip():
        raise ValueError('Frozen plan changed')
    plan = read(run / 'plan.json')
    if plan['schema'] != SCHEMA:
        raise ValueError('Wrong plan schema')
    for relative, digest in plan['code_sha256'].items():
        if sha(ROOT / relative) != digest:
            raise ValueError('Code changed since prepare: ' + relative)
    if sha(Path(plan['ostrack']) / 'ostrack_runner.py') != plan['host_wrapper_sha256']:
        raise ValueError('OSTrack host wrapper changed')
    for manifest in plan['inference_manifests']:
        if sha(manifest['path']) != manifest['sha256']:
            raise ValueError('Inference manifest changed')
    if sha(Path(plan['ostrack']) / 'pytorch_model.bin') != plan['checkpoint_sha256']:
        raise ValueError('OSTrack checkpoint changed')
    for relative, digest in plan['external_source_sha256'].items():
        if sha(safe_path(plan['ostrack'], relative)) != digest:
            raise ValueError('OSTrack runtime source/config changed: ' + relative)
    return plan


def validate_cached_row(row, plan=None, verify_files=True):
    """Reject falsified frame roles/mappings before generation; optionally hash all files."""
    source_frame = int(row['source_frame']); sequence = row['sequence']; modality = row['modality']
    if modality not in {'IR', 'RGB'}:
        raise ValueError('Unsupported modality')
    if row['local_source_frame'] != source_frame or row['local_operation'] != 'track_current_frame':
        raise ValueError('Local prediction is not the current-frame track output')
    reference = row['reference']; expected_init = row['init_source_frame']
    if reference['source_frame'] != expected_init or reference['source_frame'] >= source_frame:
        raise ValueError('Reference is not the actual initialization frame')
    evidence = [reference] + list(row['history']) + list(row['candidates'])
    for item in evidence:
        if item['sequence'] != sequence or item['modality'] != modality:
            raise ValueError('Evidence sequence/modality mismatch')
        if verify_files and (sha(item['source_path']) != item['source_sha256'] or sha(item['image']) != item['sha256']):
            raise ValueError('Evidence source/crop hash mismatch')
        if verify_files:
            # Hashing self-reported paths alone would accept a current crop
            # relabeled as an old source. Rebuild from the actual source pixels.
            import cv2
            import numpy as np
            bgr = cv2.imdecode(np.fromfile(item['source_path'], dtype=np.uint8), cv2.IMREAD_COLOR)
            if bgr is None:
                raise ValueError('Evidence source failed to decode')
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            with tempfile.TemporaryDirectory(prefix='e06b_crop_verify_') as folder:
                rebuilt = marked_crop(rgb, item['bbox_xywh'], Path(folder) / 'crop.png')
                if rebuilt['sha256'] != item['sha256']:
                    raise ValueError('Crop pixels do not originate from claimed source frame/box')
                if any(rebuilt[key] != item[key] for key in ('crop_bounds_xyxy', 'inner_prediction_box_xyxy')):
                    raise ValueError('Recorded crop geometry differs from actual source/box')
    if any(h['source_frame'] >= source_frame for h in row['history']):
        raise ValueError('History includes current/future frame')
    past_frames = [h['source_frame'] for h in row['history']]
    if past_frames != sorted(set(past_frames)) or len(past_frames) > 2:
        raise ValueError('History must be distinct ordered past predictions, maximum two')
    if any(c['source_frame'] != source_frame for c in row['candidates']):
        raise ValueError('Candidate image is not current frame')
    candidates = row['candidates']
    if len(candidates) != 4 or {c['label'] for c in candidates} != set('ABCD'):
        raise ValueError('Expected four uniquely lettered candidates')
    if {c['native_rank'] for c in candidates} != set(range(1, 5)):
        raise ValueError('Lost native rank mapping')
    if len({c['candidate_id'] for c in candidates}) != 4:
        raise ValueError('Candidate identity collision')
    expected = {c['label']: c['candidate_id'] for c in candidates}
    if row['letter_to_candidate_id'] != expected:
        raise ValueError('Letter/physical candidate mapping differs')
    all_candidates = row['global_top16']
    expected_canonical = canonical_candidates([
        {'bbox': c['bbox_xywh'], 'rank': c['native_rank'], 'score': c['tracker_score'],
         'tile_id': c['tile_id'], 'grid_index': c['grid_index']} for c in all_candidates],
        sequence, source_frame)
    if any(c['candidate_id'] != expected_canonical[i]['candidate_id'] for i, c in enumerate(all_candidates)):
        raise ValueError('Physical candidate identity cannot be reconstructed')
    by_rank = {c['native_rank']: c for c in all_candidates}
    for candidate in candidates:
        native = by_rank.get(candidate['native_rank'])
        if native is None or any(candidate[k] != native[k] for k in (
                'candidate_id', 'bbox_xywh', 'original_rank', 'tracker_score', 'tile_id', 'grid_index')):
            raise ValueError('Top4 no longer corresponds to frozen native top16')
    ordered = [by_rank[i] for i in range(1, 5)]
    expected_labels = {c['candidate_id']: c['label'] for c in label_candidates(ordered, sequence, source_frame)}
    if any(c['label'] != expected_labels[c['candidate_id']] for c in candidates):
        raise ValueError('Letters disagree with deterministic frozen permutation')
    st = row['global_scan']
    if st['forward_count'] != st['tiles_executed'] or st['forward_count'] <= 0:
        raise ValueError('No genuine full-image forwards recorded')
    if len(st['tiles']) != st['forward_count'] or not all(t['forward_completed'] for t in st['tiles']):
        raise ValueError('Incomplete tile forward ledger')
    if plan is not None:
        sequence_plan = next(s for s in plan['sequences'] if s['sequence'] == sequence)
        if modality != sequence_plan['modality'] or expected_init != sequence_plan['init_source_frame']:
            raise ValueError('Manifest identity/modality mismatch')
        manifest_frames = {f['source_frame']: f for f in sequence_plan['frames']}
        if sequence_plan['frames'][row['frame_index']]['source_frame'] != source_frame:
            raise ValueError('Local frame index/source-frame mismatch')
        for item in evidence:
            expected_source = manifest_frames[item['source_frame']]
            if item['source_sha256'] != expected_source['sha256'] or Path(item['source_path']).resolve() != safe_path(sequence_plan['asset_root'], expected_source['path']):
                raise ValueError('Evidence disagrees with frozen inference manifest')
        if reference['bbox_xywh'] != _bbox(sequence_plan['init_bbox_xywh']):
            raise ValueError('Reference used an unauthorized init box')
        if st['forward_count'] != sequence_plan['actual_global_tiles_per_query']:
            raise ValueError('Global forward count differs from frozen covering grid')
    return True


def validate_row(row, sequence=None, verify_files=True):
    """Public adapter for the shared generation validator."""
    plan = {'sequences': [sequence]} if sequence is not None else None
    return validate_cached_row(row, plan=plan, verify_files=verify_files)


def cache(args, *, runner_factory=None, sample_target_fn=None, torch_module=None):
    import cv2
    import numpy as np
    import torch as installed_torch
    torch = torch_module or installed_torch
    plan = plan_for(args.run)
    output = args.run / 'cache'; output.mkdir(exist_ok=False)
    if runner_factory is None or sample_target_fn is None:
        sys.path.insert(0, plan['ostrack'])
        from ostrack_runner import OSTrackRunner, sample_target
        runner_factory = runner_factory or OSTrackRunner
        sample_target_fn = sample_target_fn or sample_target
    torch.manual_seed(20261029); np.random.seed(20261029)
    runner = runner_factory(device=args.device, checkpoint=Path(plan['ostrack']) / 'pytorch_model.bin')
    rows, initial, totals = [], {}, []
    started = time.perf_counter()
    def sync():
        if str(args.device).startswith('cuda'):
            torch.cuda.synchronize()
    try:
        for sequence in plan['sequences']:
            seq = sequence['sequence']; sid = slug(seq); history = []
            queries = set(sequence['query_source_frames']); root = Path(sequence['asset_root'])
            baseline = args.run / 'baseline' / (sid + '.jsonl'); baseline.parent.mkdir(exist_ok=True)
            reference = None
            for fi, frame in enumerate(sequence['frames']):
                source = _source_metadata(seq, frame, root, sequence['modality'])
                if sha(source['source_path']) != source['source_sha256']:
                    raise ValueError('Source image changed')
                bgr = cv2.imdecode(np.fromfile(source['source_path'], dtype=np.uint8), cv2.IMREAD_COLOR)
                if bgr is None:
                    raise ValueError('Image failed to decode')
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                if [rgb.shape[1], rgb.shape[0]] != sequence['image_size_wh']:
                    raise ValueError('Image dimensions changed within sequence')
                sync(); local_start = time.perf_counter()
                local = runner.initialize(rgb, sequence['init_bbox_xywh']) if fi == 0 else runner.track(rgb)
                sync(); local_seconds = time.perf_counter() - local_start
                box = _bbox(local['bbox'])
                record = {**source, 'frame_index': fi, 'bbox_xywh': box,
                          'score': float(local['score']), 'seconds': local_seconds,
                          'operation': 'initialize' if fi == 0 else 'track_current_frame',
                          'scoring_performed': False}
                with baseline.open('a', encoding='utf-8') as stream:
                    stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
                sf = frame['source_frame']
                if fi == 0:
                    reference = _crop(rgb, sequence['init_bbox_xywh'], args.run / 'images' / sid / 'reference.png', source, 'initial_reference')
                    initial[seq] = {'reference': reference, 'distractors': [], 'init_source_frame': sf}
                    continue
                if sf in queries:
                    pool, stats = full_image_scan(runner, rgb, sequence['init_bbox_xywh'],
                                                  sample_target=sample_target_fn, torch_module=torch,
                                                  device=str(args.device),
                                                  ledger_path=output / 'forward_ledger' / f'{sid}_{sf:06d}.jsonl')
                    top16 = canonical_candidates(pool, seq, sf)
                    if len(top16) < 4:
                        raise ValueError('Fewer than four global proposals; preserve failure, do not replace query')
                    candidates = []
                    for c in label_candidates(top16[:4], seq, sf):
                        crop = _crop(rgb, c['bbox_xywh'], args.run / 'images' / sid / f'{sf:06d}_{c["candidate_id"]}.png', source, 'current_candidate')
                        candidates.append({**c, **crop})
                    row = {'schema': SCHEMA, 'case_id': f'{sid}_{sf:06d}',
                           'sequence': seq, 'source_frame': sf, 'frame_index': fi,
                           'modality': sequence['modality'], 'role': 'exploratory',
                           'init_source_frame': sequence['init_source_frame'],
                           'local_bbox_xywh': box, 'local_score': float(local['score']),
                           'local_source_frame': sf, 'local_operation': 'track_current_frame',
                           'reference': reference, 'history': list(history), 'distractors': [],
                           'candidates': candidates, 'global_top16': top16, 'global_scan': stats,
                           'letter_to_candidate_id': {c['label']: c['candidate_id'] for c in candidates},
                           'generation_status': 'NOT_RUN', 'closedloop': False}
                    validate_cached_row(row, plan)
                    rows.append(row); totals.append({'case_id': row['case_id'], **stats})
                    with (output / 'queries.jsonl').open('a', encoding='utf-8') as stream:
                        stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
                # Appending after query snapshot is what makes the history strictly past.
                past = _crop(rgb, box, args.run / 'images' / sid / f'{sf:06d}_past_local.png', source, 'past_local_prediction')
                history = (history + [past])[-2:]
            if len([r for r in rows if r['sequence'] == seq]) != len(queries):
                raise ValueError('Missing frozen queries')
            print(f'Cached {seq}: {len(sequence["frames"])} local frames, {len(queries)} global queries', flush=True)
        write(output / 'candidates.json', {'schema': SCHEMA, 'rows': rows, 'initial': initial,
              'global_timing': totals, 'plan_sha256': sha(args.run / 'plan.json'),
              'gpu': torch.cuda.get_device_name(0) if str(args.device).startswith('cuda') else 'CPU',
              'python': sys.version, 'torch': torch.__version__, 'opencv': cv2.__version__,
              'cuda': torch.version.cuda, 'device': str(args.device),
              'elapsed_seconds': time.perf_counter() - started,
              'inference_reads_evaluation_file': False, 'complete_baseline_frames': True,
              'generation_attempts': 0, 'command': list(sys.argv)})
        write(output / 'completed.json', {'status': 'COMPLETE', 'queries': len(rows),
              'global_forward_count': sum(t['forward_count'] for t in totals),
              'local_frames': sum(len(s['frames']) for s in plan['sequences']),
              'generation_attempts': 0, 'candidate_cache_sha256': sha(output / 'candidates.json')})
    except Exception as exc:
        write(output / 'failed.json', {'status': 'FAILED', 'exception_type': type(exc).__name__,
              'message': str(exc), 'completed_queries': len(rows), 'generation_attempts': 0})
        raise


def truth_with_sources(plan):
    """Independent scoring index, with explicit source-frame and modality pairing."""
    truth, sources = {}, []
    for sequence in plan['sequences']:
        root = Path(sequence['asset_root']); modality = sequence['modality']
        if modality == 'IR' and sequence['sequence'].startswith('validation/'):
            annotation = root / 'scoring/annotation.json'
        elif modality == 'RGB' and sequence['sequence'] == 'test/16-USV':
            annotation = root / 'mvtd_exploration/evaluation.json'
        else:
            raise ValueError('No corresponding-modality GT provider')
        document = read(annotation)
        found = next(s for s in document['sequences'] if s['sequence'] == sequence['sequence'])
        frames = {int(f['source_frame']): f for f in found['frames']}
        for f in sequence['frames']:
            sf = f['source_frame']
            if sf not in frames or frames[sf]['local_index'] != f['local_index']:
                raise ValueError('Scoring frame alignment differs')
        if modality == 'IR':
            for g in frames.values():
                if g.get('exist') not in (0, 1) or g.get('absence') != 1 - g['exist']:
                    raise ValueError('IR exist/absence contradiction')
        truth[sequence['sequence']] = {sf: {**g, 'modality': modality} for sf, g in frames.items()}
        sources.append({'sequence': sequence['sequence'], 'modality': modality,
                        'path': str(annotation), 'sha256': sha(annotation)})
    return truth, sources


def truth_state(gt):
    if gt.get('exist') == 0 and gt.get('absence') == 1:
        return 'absent'
    if gt.get('bbox_valid') and gt.get('bbox_xywh') is not None:
        _bbox(gt['bbox_xywh'])
        return 'visible'
    return 'unknown'


def truth_for(plan):
    nested, _ = truth_with_sources(plan)
    return {(seq, sf): {**gt, 'state': truth_state(gt)}
            for seq, frames in nested.items() for sf, gt in frames.items()}


def inspect(args):
    plan = plan_for(args.run); cache_file = args.run / 'cache/candidates.json'
    data = read(cache_file)
    if data['plan_sha256'] != sha(args.run / 'plan.json'):
        raise ValueError('Candidate cache belongs to another plan')
    truth, sources = truth_with_sources(plan)
    rows, by_sequence = [], {}
    keys = ['queries', 'visible', 'absent', 'unknown', 'local_success', 'native_top1_success',
            'recall4', 'recall16', 'candidate_ranking_headroom', 'actual_local_recovery_headroom',
            'native_damages_local']
    counts = {k: 0 for k in keys}; ir_counts = {k: 0 for k in keys}
    for row in data['rows']:
        validate_cached_row(row, plan)
        seq = row['sequence']; gt = truth[seq][row['source_frame']]
        if row['modality'] != gt['modality'] or row['local_source_frame'] != row['source_frame']:
            raise ValueError('Prediction/GT modality or current-frame mismatch')
        state = truth_state(gt)
        result = {'case_id': row['case_id'], 'sequence': seq, 'source_frame': row['source_frame'],
                  'modality': row['modality'], 'state': state,
                  'exist': gt.get('exist'), 'absence': gt.get('absence'),
                  'local_iou': None, 'native_top1_iou': None, 'max_iou4': None, 'max_iou16': None}
        group = by_sequence.setdefault(seq, {k: 0 for k in keys})
        group['queries'] += 1; group[state] += 1
        if state == 'visible':
            box = gt['bbox_xywh']; native = min(row['candidates'], key=lambda c: c['native_rank'])
            result.update(local_iou=overlap(row['local_bbox_xywh'], box),
                          native_top1_iou=overlap(native['bbox_xywh'], box),
                          max_iou4=max(overlap(c['bbox_xywh'], box) for c in row['candidates']),
                          max_iou16=max(overlap(c['bbox_xywh'], box) for c in row['global_top16']))
            local = result['local_iou'] >= .5; rank1 = result['native_top1_iou'] >= .5
            recall4 = result['max_iou4'] >= .5; recall16 = result['max_iou16'] >= .5
            for key, value in {'local_success': local, 'native_top1_success': rank1,
                               'recall4': recall4, 'recall16': recall16,
                               'candidate_ranking_headroom': recall4 and not rank1,
                               'actual_local_recovery_headroom': recall4 and not local,
                               'native_damages_local': local and not rank1}.items():
                group[key] += int(value)
        rows.append(result)
    expected_ids = {f'{slug(s["sequence"])}_{sf:06d}' for s in plan['sequences'] for sf in s['query_source_frames']}
    if {r['case_id'] for r in rows} != expected_ids or len(rows) != len(expected_ids):
        raise ValueError('Candidate query set is incomplete or duplicated')
    for group in by_sequence.values():
        for key in keys:
            counts[key] += group[key]
    for seq, group in by_sequence.items():
        if next(s for s in plan['sequences'] if s['sequence'] == seq)['modality'] == 'IR':
            for key in keys:
                ir_counts[key] += group[key]
    opportunity = {'schema': SCHEMA, 'scope': 'exploratory candidate opportunity, no VLM result',
                   'counts': counts, 'totals': counts, 'ir_totals': ir_counts,
                   'by_sequence': by_sequence, 'by_role': {'exploratory': counts},
                   **counts, 'rows': rows, 'truth_sources': sources,
                   'candidate_cache_sha256': sha(cache_file), 'plan_sha256': sha(args.run / 'plan.json'),
                   'has_ranking_space': counts['candidate_ranking_headroom'] > 0,
                   'has_local_recovery_space': counts['actual_local_recovery_headroom'] > 0,
                   'generation_attempts': 0, 'no_query_selection_based_on_truth': True,
                   'threshold_iou': .5, 'closedloop': False}
    write(args.run / 'opportunity.json', opportunity)
    print(json.dumps({'opportunity': str(args.run / 'opportunity.json'), 'counts': counts}, ensure_ascii=False, indent=2))
