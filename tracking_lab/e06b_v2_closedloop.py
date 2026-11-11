"""Stage-B no-GT closed-loop recovery entry.

This entry deliberately separates inference from scoring.  It runs five arms
with the same frozen candidate cache: A, G_always, V0_always, G_score_gate and
V0_score_gate.  The first pass uses a fixed initialization template, writes no
template/memory updates, and performs zero language generation.  Ground truth
is only read by ``score`` after inference has finished.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

from tracking_lab.e05 import read, sha, write
from tracking_lab import e06b_v2_candidates as front

ROOT = Path(__file__).resolve().parents[1]
ARMS = ('A', 'G_always', 'V0_always', 'G_score_gate', 'V0_score_gate')
SELECTOR_ARMS = ('V0_always', 'V0_score_gate')


def utc():
    return datetime.now(timezone.utc).isoformat()


def box(value):
    return [float(v) for v in value]


def digest(value):
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy().tobytes()
    elif isinstance(value, bytes):
        pass
    else:
        value = json.dumps(value, sort_keys=True, allow_nan=False).encode()
    return hashlib.sha256(value).hexdigest()


def template_hash(runner):
    return digest(runner.template) if hasattr(runner, 'template') else None


def frame_source(sequence, frame):
    root = Path(sequence['asset_root'])
    path = (root / frame['path']).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError('Frame path escapes asset root')
    if sha(path) != frame['sha256']:
        raise ValueError('Frame hash differs from frozen manifest: ' + str(path))
    return path


def _candidate_by_rank(row, rank):
    return next(c for c in row['candidates'] if int(c['native_rank']) == int(rank))


def _selector_result(selector, row):
    if selector is None:
        return {'status': 'NOT_RUN', 'answer': None, 'scores': {}, 'image_encoder_calls': 0}
    candidates = [{'label': c['label'], 'image': c['image']} for c in row['candidates']]
    result = selector.select(row['reference']['image'], candidates, row['case_id'])
    if result.get('status') != 'VALID' or result.get('answer') not in {c['label'] for c in row['candidates']}:
        return {**result, 'status': result.get('status', 'INVALID'), 'answer': None}
    return result


def _make_qwen_selector(model_path, output_dir, device):
    from tracking_lab.e05_qwen import E05Qwen
    runner = E05Qwen(model_path, output_dir, max_calls=1, device=device)

    class Selector:
        def select(self, reference, candidates, case_id):
            result = runner.visual_scores(reference, [], candidates)
            return {'status': 'VALID', 'answer': result['answer'],
                    'scores': result['scores'],
                    'image_pixel_hashes': result['image_pixel_hashes'],
                    'image_encoder_calls': result['image_encoder_calls'],
                    'seconds': result['seconds'], 'feature_api': result['feature_api'],
                    'case_id': case_id}

        @property
        def metadata(self):
            return runner.metadata

    return Selector()


def run_closed_loop(plan, rows, output, *, device='cuda', runner_factory=None,
                    selector=None, sample_loader=None):
    """Run inference only. ``plan`` must contain no scoring labels.

    The injectable runner and selector make the causal ordering testable without
    importing a model.  The production path uses the verified OSTrack wrapper
    and the Qwen visual tower, never Qwen generation.
    """
    import cv2
    import numpy as np
    import torch
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    if sample_loader is None:
        def sample_loader(path):
            image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError('Cannot decode image: ' + str(path))
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if runner_factory is None:
        ostrack = Path(plan['ostrack']).resolve()
        sys.path.insert(0, str(ostrack))
        from ostrack_runner import OSTrackRunner
        runner_factory = OSTrackRunner
    row_by_key = {(r['sequence'], int(r['source_frame'])): r for r in rows}
    if len(row_by_key) != len(rows):
        raise ValueError('Duplicate candidate query')
    query_selection = []
    query_selection_by_key = {}
    for row in rows:
        if row['modality'] != 'IR':
            raise ValueError('Stage-B first entry accepts IR only')
        result = _selector_result(selector, row)
        selection = {'case_id': row['case_id'], 'sequence': row['sequence'],
                     'source_frame': int(row['source_frame']), 'modality': row['modality'],
                     'candidates': [{k: c[k] for k in ('candidate_id', 'label', 'native_rank',
                                                       'tracker_score', 'bbox_xywh')}
                                    for c in row['candidates']],
                     'V0': result, 'native_rank1_id': _candidate_by_rank(row, 1)['candidate_id']}
        query_selection.append(selection)
        query_selection_by_key[(row['sequence'], int(row['source_frame']))] = (row, result)

    all_logs = []
    timing = []
    for sequence in plan['sequences']:
        seq = sequence['sequence']; frames = sequence['frames']
        image_paths = [frame_source(sequence, f) for f in frames]
        runners = {arm: runner_factory(device=device, checkpoint=Path(plan['ostrack'])/'pytorch_model.bin')
                   for arm in ARMS}
        for index, (frame, path) in enumerate(zip(frames, image_paths)):
            rgb = sample_loader(path)
            source_frame = int(frame['source_frame'])
            for arm in ARMS:
                runner = runners[arm]
                before_state = list(getattr(runner, 'state', []) or [])
                before_template = template_hash(runner)
                started = time.perf_counter()
                if index == 0:
                    result = runner.initialize(rgb, sequence['init_bbox_xywh'])
                    operation = 'initialize'
                else:
                    result = runner.track(rgb)
                    operation = 'track_current_frame'
                pre_bbox = box(result['bbox']); pre_score = float(result.get('score', 0.))
                query = query_selection_by_key.get((seq, source_frame))
                query_scheduled = query is not None
                selection_status = 'NOT_SCHEDULED'; answer = None; selected = None
                gate_feature = None; accept = False; selector_info = None
                if query is not None:
                    row, selector_info = query
                    if arm.startswith('G_'):
                        selection_status = 'VALID'
                    elif arm.startswith('V0_'):
                        selection_status = selector_info.get('status', 'INVALID')
                    else:
                        selection_status = 'NOT_APPLICABLE'
                    answer = selector_info.get('answer') if arm.startswith('V0_') else None
                    if arm.startswith('G_'):
                        selected = _candidate_by_rank(row, 1)
                    elif arm.startswith('V0_') and answer is not None:
                        selected = next(c for c in row['candidates'] if c['label'] == answer)
                    if arm.endswith('_score_gate'):
                        gate_feature = pre_score
                        accept = selection_status == 'VALID' and pre_score < float(plan['tau'])
                    elif arm != 'A':
                        accept = selection_status == 'VALID' and selected is not None
                if accept and selected is not None:
                    runner.state = box(selected['bbox_xywh'])
                post_bbox = box(getattr(runner, 'state', pre_bbox))
                after_template = template_hash(runner)
                if before_template != after_template:
                    raise ValueError('Template changed; Stage-B first entry forbids template updates')
                all_logs.append({'schema': 'e06b_v2_closedloop_20261102', 'sequence': seq,
                                 'modality': sequence['modality'], 'source_frame': source_frame,
                                 'frame_index': int(frame['local_index']), 'arm': arm,
                                 'source_path': str(path), 'source_sha256': sha(path),
                                 'operation': operation, 'query_scheduled': query_scheduled,
                                 'pre_action_bbox_xywh': pre_bbox, 'pre_action_score': pre_score,
                                 'selector_status': selection_status, 'selector_answer': answer,
                                 'selected_candidate_id': selected['candidate_id'] if selected else None,
                                 'selected_native_rank': selected['native_rank'] if selected else None,
                                 'selected_bbox_xywh': selected['bbox_xywh'] if selected else None,
                                 'gate_tau': float(plan['tau']) if arm.endswith('_score_gate') else None,
                                 'gate_feature_raw_score': gate_feature, 'accepted': bool(accept),
                                 'action': 'WRITE_STATE' if accept else 'KEEP_PRE',
                                 'post_action_bbox_xywh': post_bbox,
                                 'state_before_xywh': before_state,
                                 'state_after_xywh': post_bbox,
                                 'template_hash_before': before_template,
                                 'template_hash_after': after_template,
                                 'template_update': False,
                                 'gt_read_during_inference': False,
                                 'seconds': time.perf_counter() - started,
                                 'tracker_forward_count': 1,
                                 'selector_cache_hit': bool(selector_info and arm == 'V0_score_gate'),
                                 # The selector is computed once per query and
                                 # reused by V0_always/V0_score_gate. Count it
                                 # on the first arm only; do not double-count.
                                 'selector_encoder_calls': selector_info.get('image_encoder_calls', 0)
                                 if selector_info and arm == 'V0_always' else 0})
                timing.append({'sequence': seq, 'source_frame': source_frame,
                               'arm': arm, 'seconds': all_logs[-1]['seconds']})
        del runners
        if str(device).startswith('cuda') and hasattr(torch, 'cuda'):
            torch.cuda.empty_cache()
    with (output/'query_selection.jsonl').open('w', encoding='utf-8') as stream:
        for row in query_selection:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
    write(output/'completed.json', {'status': 'COMPLETE', 'closed_loop': True,
          'gt_read_during_inference': False, 'language_generation_attempts': 0,
          'arms': list(ARMS), 'frames': len(all_logs),
          'queries': len(query_selection), 'template_updates': 0,
          'tau': float(plan['tau']), 'utc': utc()})
    with (output/'per_frame.jsonl').open('w', encoding='utf-8') as stream:
        for row in all_logs:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
    return all_logs


def score_logs(log_dir, annotation_root, output):
    """Independent GT scoring; this is the only function that reads annotation."""
    logs = [json.loads(x) for x in (Path(log_dir)/'per_frame.jsonl').read_text(encoding='utf-8').splitlines() if x]
    ann = read(Path(annotation_root)/'scoring/annotation.json')
    truth = {s['sequence']: {int(f['source_frame']): f for f in s['frames']} for s in ann['sequences']}
    result = []
    for row in logs:
        gt = truth[row['sequence']][row['source_frame']]
        if row['modality'] != 'IR':
            raise ValueError('Modality mismatch')
        state = 'visible' if gt.get('bbox_valid') else ('absent' if gt.get('exist') == 0 else 'unknown')
        bbox_gt = gt.get('bbox_xywh')
        pre = front.overlap(row['pre_action_bbox_xywh'], bbox_gt) if state == 'visible' else None
        post = front.overlap(row['post_action_bbox_xywh'], bbox_gt) if state == 'visible' else None
        result.append({**row, 'state': state, 'pre_iou': pre, 'post_iou': post,
                       'pre_success': pre is not None and pre >= .5,
                       'post_success': post is not None and post >= .5,
                       'false_accept': state == 'absent' and row['accepted']})
    by_arm = {}
    for arm in ARMS:
        group = [x for x in result if x['arm'] == arm]
        visible = [x for x in group if x['state'] == 'visible']
        by_arm[arm] = {'frames': len(group), 'visible': len(visible),
                       'pre_success': sum(x['pre_success'] for x in visible),
                       'post_success': sum(x['post_success'] for x in visible),
                       'absent': sum(x['state'] == 'absent' for x in group),
                       'false_accept': sum(x['false_accept'] for x in group),
                       'unknown': sum(x['state'] == 'unknown' for x in group),
                       'accepted': sum(x['accepted'] for x in group)}
    report = {'schema': 'e06b_v2_closedloop_score_20261102', 'gt_read_stage': 'score_after_inference',
              'annotation_sha256': sha(Path(annotation_root)/'scoring/annotation.json'),
              'by_arm': by_arm, 'rows': result, 'closed_loop': True,
              'gt_read_during_inference': False}
    write(Path(output)/'score.json', report)
    return report


def prepare(args):
    plan = front.plan_for(args.cache_run)
    if args.tau <= 0:
        raise ValueError('tau must be positive and frozen before inference')
    cache = read(args.cache_run/'cache/candidates.json')
    rows = cache['rows']
    if len(rows) != 12 or any(r['modality'] != 'IR' for r in rows):
        raise ValueError('Stage-B first entry expects the twelve frozen IR queries')
    for row in rows:
        front.validate_cached_row(row, plan=plan, verify_files=True)
    qwen = Path(args.qwen).resolve()
    if not (qwen/'config.json').is_file():
        raise FileNotFoundError('Stage-B prepare requires the existing offline Qwen model directory')
    args.output.mkdir(parents=True, exist_ok=False)
    frozen = {'schema': 'e06b_v2_closedloop_20261102', 'run_id': args.output.name,
              'cache_run': str(args.cache_run.resolve()),
              'candidate_cache_sha256': sha(args.cache_run/'cache/candidates.json'),
              'candidate_plan_sha256': sha(args.cache_run/'plan.json'),
              'ostrack': str(args.ostrack.resolve()), 'tau': float(args.tau),
              'qwen': str(qwen), 'qwen_config_sha256': sha(qwen/'config.json'),
              'arms': list(ARMS), 'template_updates': False,
              'language_generation_attempts': 0, 'gt_read_during_inference': False,
              'selector': 'Qwen visual tower only for V0; no Qwen generation',
              'selection': 'G native rank1 or V0 visual tower over the shared fixed top4 cache',
              'gate': 'query_scheduled AND raw current arm score < frozen tau AND selection valid',
              'query_ids': [r['case_id'] for r in rows], 'closed_loop': True,
              'created_utc': utc()}
    write(args.output/'closedloop_plan.json', frozen)
    (args.output/'closedloop_plan.sha256').write_text(sha(args.output/'closedloop_plan.json')+'\n', encoding='utf-8')
    print(json.dumps({'prepared': str(args.output), 'queries': 12, 'arms': list(ARMS),
                      'language_generation_attempts': 0}, ensure_ascii=False))


def run(args):
    plan_path = args.run/'closedloop_plan.json'
    if sha(plan_path) != (args.run/'closedloop_plan.sha256').read_text().strip():
        raise ValueError('Closed-loop plan changed')
    plan = read(plan_path)
    if plan['language_generation_attempts'] != 0 or plan['template_updates']:
        raise ValueError('Unsupported Stage-B plan')
    if Path(args.qwen).resolve() != Path(plan['qwen']).resolve():
        raise ValueError('Qwen path differs from frozen Stage-B plan')
    if sha(Path(args.qwen)/'config.json') != plan['qwen_config_sha256']:
        raise ValueError('Qwen config changed since Stage-B prepare')
    cache = read(Path(plan['cache_run'])/'cache/candidates.json')
    if sha(Path(plan['cache_run'])/'cache/candidates.json') != plan['candidate_cache_sha256']:
        raise ValueError('Candidate cache changed')
    rows = cache['rows']
    selector = _make_qwen_selector(args.qwen, args.run/'visual_selector', args.device)
    logs = run_closed_loop(plan, rows, args.run/'inference', device=args.device, selector=selector)
    metadata = {'model': selector.metadata, 'language_generation_attempts': 0,
                'visual_encoder_calls': sum(x['selector_encoder_calls'] for x in logs),
                'runtime_argv': sys.argv}
    write(args.run/'inference/model_metadata.json', metadata)
    print(json.dumps({'status': 'COMPLETE', 'rows': len(logs), 'arms': list(ARMS),
                      'language_generation_attempts': 0}, ensure_ascii=False))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['prepare', 'run', 'score'])
    p.add_argument('--cache-run', type=Path)
    p.add_argument('--run', type=Path)
    p.add_argument('--output', type=Path)
    p.add_argument('--ostrack', type=Path)
    p.add_argument('--qwen', type=Path)
    p.add_argument('--ir-root', type=Path)
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    p.add_argument('--tau', type=float, default=0.30)
    args = p.parse_args()
    if args.command == 'prepare':
        for x in (args.cache_run, args.output, args.ostrack, args.qwen):
            if x is None: p.error('prepare requires --cache-run --output --ostrack --qwen')
        prepare(args)
    elif args.command == 'run':
        for x in (args.run, args.qwen):
            if x is None: p.error('run requires --run --qwen')
        run(args)
    else:
        if args.run is None or args.ir_root is None: p.error('score requires --run --ir-root')
        score_logs(args.run/'inference', args.ir_root, args.run/'score')


if __name__ == '__main__':
    main()
