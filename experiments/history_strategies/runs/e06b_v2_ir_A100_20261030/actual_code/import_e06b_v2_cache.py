"""Import a completed zero-generation E06B-v2 cache onto an offline host.

Only paths and lossless PNG encodings may change. Candidate boxes, rankings,
physical IDs and source pixels must remain identical. No model is loaded.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tracking_lab.e05 import marked_crop, read, sha, write
from tracking_lab import e06b_v2_candidates as front


def normalized(value):
    return str(value).replace('\\', '/').rstrip('/')


def remap(value, mappings):
    if isinstance(value, list):
        return [remap(v, mappings) for v in value]
    if isinstance(value, dict):
        return {k: remap(v, mappings) for k, v in value.items()}
    if isinstance(value, str):
        candidate = normalized(value)
        for old, new in mappings:
            if candidate == old or candidate.startswith(old + '/'):
                return new + candidate[len(old):]
    return value


def prediction_signature(data):
    """Path-independent digest of all predictions and physical proposals."""
    values = []
    for row in data['rows']:
        values.append({k: row[k] for k in ('case_id', 'sequence', 'source_frame', 'frame_index',
                                          'local_bbox_xywh', 'local_score', 'global_top16',
                                          'letter_to_candidate_id')})
    return hashlib.sha256(json.dumps(values, sort_keys=True, allow_nan=False).encode()).hexdigest()


def verify_executed_code(snapshot, current, expected_sha256):
    """Permit CRLF/LF transport differences only, retaining the original raw hash."""
    snapshot = Path(snapshot); current = Path(current)
    if sha(snapshot) != expected_sha256:
        raise ValueError('Source executed-code snapshot checksum differs')
    current_sha = sha(current)
    if current_sha == expected_sha256:
        return None
    if snapshot.read_bytes().replace(b'\r\n', b'\n') != current.read_bytes().replace(b'\r\n', b'\n'):
        raise ValueError('Current code differs beyond CRLF/LF encoding: ' + str(current))
    return {'source_sha256': expected_sha256, 'target_sha256': current_sha,
            'newline_only': True, 'normalization_checked': 'CRLF to LF only; every other byte equal'}


def import_cache(args):
    source = args.source.resolve(); target = args.run.resolve()
    assets = args.assets.resolve(); ostrack = args.ostrack.resolve()
    if not source.is_dir() or target.exists():
        raise ValueError('Source directory must exist and destination must be new')
    if target.is_relative_to(source) or source.is_relative_to(target):
        raise ValueError('Source and target directories must be separate')
    if (source / 'inference').exists() or (source / 'validation_plan.json').exists():
        raise ValueError('Import only a candidate-only run, before validation freeze')
    plan_path = source / 'plan.json'; cache_path = source / 'cache/candidates.json'
    plan = read(plan_path); data = read(cache_path); completed = read(source / 'cache/completed.json')
    source_plan_sha = sha(plan_path); source_cache_sha = sha(cache_path)
    if source_plan_sha != (source / 'plan.sha256').read_text(encoding='utf-8').strip():
        raise ValueError('Source plan checksum mismatch')
    if data['plan_sha256'] != source_plan_sha or completed['candidate_cache_sha256'] != source_cache_sha:
        raise ValueError('Source completion/cache/plan binding differs')
    if completed['status'] != 'COMPLETE' or completed['generation_attempts'] != 0 or data['generation_attempts'] != 0:
        raise ValueError('Source is incomplete or has generation attempts')
    if plan['generation_budget'] != 0 or plan['schema'] != front.SCHEMA:
        raise ValueError('Source is not a zero-generation corrected candidate plan')
    if any(s['modality'] != 'IR' for s in plan['sequences']):
        raise ValueError('This import accepts the IR12 cache only; boat paths need a separate import protocol')
    if len(data['rows']) != 12 or completed['queries'] != 12:
        raise ValueError('Expected all twelve unchanged IR queries')
    if set(plan['code_sha256']) != set(front.CODE):
        raise ValueError('Incomplete executed-source checksum list')
    if completed['global_forward_count'] != plan['global_forward_budget']:
        raise ValueError('Completed full-image forward budget differs')
    for row in data['rows']:
        front.validate_row(row, verify_files=False)
    newline_adaptations = []
    source_code_sha = dict(plan['code_sha256'])
    for relative, digest in source_code_sha.items():
        adaptation = verify_executed_code(source / 'actual_code' / relative, ROOT / relative, digest)
        if adaptation is not None:
            newline_adaptations.append({'file': relative, **adaptation})
    query_file = source / 'cache/queries.jsonl'
    if [json.loads(line) for line in query_file.read_text(encoding='utf-8').splitlines() if line] != data['rows']:
        raise ValueError('Source query ledger differs from final cache')
    original_run = normalized(data['rows'][0]['reference']['image']).split('/images/')[0]
    if original_run == normalized(data['rows'][0]['reference']['image']):
        raise ValueError('Cannot resolve original run path')
    mappings = sorted([(original_run, normalized(target)),
                       (normalized(plan['assets']), normalized(assets)),
                       (normalized(plan['ostrack']), normalized(ostrack))], key=lambda item: -len(item[0]))
    source_signature = prediction_signature(data)
    source_artifacts = {}
    for path in source.rglob('*'):
        if path.is_symlink():
            raise ValueError('Source contains a symlink')
        if path.is_file():
            source_artifacts[path.relative_to(source).as_posix()] = {'bytes': path.stat().st_size, 'sha256': sha(path)}
    shutil.copytree(source, target)
    archive = target / 'import_source'; archive.mkdir()
    shutil.copytree(source / 'actual_code', archive / 'actual_code')
    for relative in ('plan.json', 'plan.sha256', 'cache/candidates.json', 'cache/completed.json', 'opportunity.json'):
        path = source / relative
        if path.exists():
            dest = archive / relative; dest.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(path, dest)
    # Preserve unmodified source metadata before translating JSON/JSONL paths.
    for path in list(target.rglob('*')):
        if not path.is_file() or path.is_relative_to(archive) or path.is_relative_to(target / 'actual_code'):
            continue
        if path.suffix == '.json':
            write(path, remap(read(path), mappings))
        elif path.suffix == '.jsonl':
            rows = [remap(json.loads(line), mappings) for line in path.read_text(encoding='utf-8').splitlines() if line]
            path.write_text(''.join(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n' for row in rows), encoding='utf-8')
    plan = read(target / 'plan.json')
    plan['run_id'] = target.name
    plan['executed_code_sha256'] = source_code_sha
    plan['code_sha256'] = {relative: sha(ROOT / relative) for relative in source_code_sha}
    plan['actual_code_scope'] = 'Original GPU execution snapshot, byte-for-byte; current runtime may differ only by verified CRLF/LF encoding'
    plan['imported_from_run_id'] = read(source / 'plan.json')['run_id']
    plan['command'] = list(sys.argv)
    plan['external_source_sha256'] = {normalized(k): v for k, v in plan['external_source_sha256'].items()}
    write(target / 'plan.json', plan)
    (target / 'plan.sha256').write_text(sha(target / 'plan.json') + '\n', encoding='utf-8')
    # Verify the target assets/runtime against the exact source code and weights.
    front.plan_for(target)
    data = read(target / 'cache/candidates.json')
    data['plan_sha256'] = sha(target / 'plan.json')
    encodings, seen = [], {}
    import cv2
    import numpy as np
    from PIL import Image
    evidence = []
    for row in data['rows']:
        evidence.extend([row['reference'], *row['history'], *row['candidates']])
    for value in data['initial'].values():
        evidence.append(value['reference'])
    for item in evidence:
        image = Path(item['image'])
        if image in seen:
            item['sha256'] = seen[image]
            continue
        if not image.resolve().is_relative_to(target) or sha(image) != item['sha256']:
            raise ValueError('Imported image path/checksum differs')
        if sha(item['source_path']) != item['source_sha256']:
            raise ValueError('Target source image hash differs')
        original_sha = sha(image)
        bgr = cv2.imdecode(np.fromfile(item['source_path'], dtype=np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError('Source failed to decode on this host')
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        with tempfile.TemporaryDirectory(prefix='e06b_import_crop_') as folder:
            rebuilt = marked_crop(rgb, item['bbox_xywh'], Path(folder) / 'crop.png')
            with Image.open(image) as old, Image.open(rebuilt['image']) as new:
                if old.size != new.size or old.convert('RGB').tobytes() != new.convert('RGB').tobytes():
                    raise ValueError('Host crop RGB pixels differ; do not silently change inputs')
            if rebuilt['sha256'] != original_sha:
                shutil.copy2(rebuilt['image'], image)
                encodings.append({'image': str(image), 'original_sha256': original_sha,
                                  'new_sha256': rebuilt['sha256'], 'RGB_pixels_equal': True})
        item['sha256'] = sha(image); seen[image] = item['sha256']
    if prediction_signature(data) != source_signature:
        raise ValueError('Import changed prediction geometry/ranks/physical candidate IDs')
    write(target / 'cache/candidates.json', data)
    (target / 'cache/queries.jsonl').write_text(''.join(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n' for row in data['rows']), encoding='utf-8')
    completed = read(target / 'cache/completed.json')
    completed['candidate_cache_sha256'] = sha(target / 'cache/candidates.json')
    completed['imported_without_model_execution'] = True
    write(target / 'cache/completed.json', completed)
    # Original opportunity remains archived; target scoring reads matching local annotations.
    if (target / 'opportunity.json').exists():
        (target / 'opportunity.json').unlink()
    front.inspect(SimpleNamespace(run=target))
    receipt = {'status': 'COMPLETE', 'source': str(source), 'run': str(target),
               'source_plan_sha256': source_plan_sha, 'source_cache_sha256': source_cache_sha,
               'source_artifacts': source_artifacts, 'path_mappings': [{'from': old, 'to': new} for old, new in mappings],
               'source_executed_code_sha256': source_code_sha,
               'target_runtime_code_sha256': plan['code_sha256'],
               'newline_adaptations': newline_adaptations,
               'actual_code_scope': 'Original source GPU run; preserved in actual_code and import_source/actual_code. No tracker execution occurred on target.',
               'target_plan_sha256': sha(target / 'plan.json'),
               'target_cache_sha256': sha(target / 'cache/candidates.json'),
               'source_prediction_signature': source_signature,
               'target_prediction_signature': prediction_signature(data),
               'lossless_png_reencodings': encodings, 'checked_unique_evidence_crops': len(seen),
               'generation_attempts': 0, 'new_tracker_forwards': 0,
               'script_sha256': sha(__file__), 'command': list(sys.argv),
               'notes': 'Imported candidate measurements retain original GPU/environment metadata; target inspect is CPU-only. No candidate rerun or VLM.'}
    write(target / 'import_receipt.json', receipt)
    print(json.dumps({'imported': str(target), 'checked_crops': len(seen), 'PNG_reencoded': len(encodings),
                      'new_model_calls': 0}, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--assets', type=Path, required=True)
    parser.add_argument('--ostrack', type=Path, required=True)
    args = parser.parse_args()
    target_existed = args.run.exists()
    try:
        import_cache(args)
    except Exception as error:
        if not target_existed and args.run.is_dir() and not (args.run / 'import_receipt.json').exists():
            write(args.run / 'import_failed.json', {'status': 'FAILED', 'error_class': type(error).__name__,
                  'error': str(error), 'generation_attempts': 0, 'new_tracker_forwards': 0})
        raise


if __name__ == '__main__':
    main()
