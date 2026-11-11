"""Corrected E06B: cache first, inspect, then freeze a bounded identity verifier.

No old prediction is overwritten. This is offline candidate verification, not
closed-loop recovery or a deployed failure gate. The default generation budget
is zero until a separate freeze command passes the candidate-opportunity check.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import sys
import time

from tracking_lab.e05 import read, write, sha
from tracking_lab.e05_metrics import iou_xywh
from tracking_lab.e05_qwen import (E05Qwen, DIRECT_PROMPT, PROMPT_VERSION,
                                   BudgetExceeded, GenerationFailure)

ROOT = Path(__file__).resolve().parents[1]


def utc():
    return datetime.now(timezone.utc).isoformat()


def append(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def validate_mapping(row):
    candidates = row['candidates']
    ids = [c['candidate_id'] for c in candidates]
    labels = [c['label'] for c in candidates]
    ranks = [c['native_rank'] for c in candidates]
    if not 1 <= len(candidates) <= 4 or len(set(ids)) != len(ids):
        raise ValueError('Need unique physical top4 candidates')
    if len(set(labels)) != len(labels) or not set(labels) <= set('ABCD'):
        raise ValueError('Invalid candidate letters')
    if sorted(ranks) != list(range(1, len(candidates) + 1)):
        raise ValueError('Native ranks must reconstruct top4')
    if any(c.get('original_rank', c['native_rank']) != c['native_rank'] for c in candidates):
        raise ValueError('Conflicting native ranks')
    return {c['label']: c['candidate_id'] for c in candidates}


def validate_evidence(row, sequence=None):
    from tracking_lab.e06b_v2_candidates import validate_row
    validate_mapping(row)
    return validate_row(row, sequence) if sequence is not None else validate_row(row)


def decision(row, result):
    """Retain local on every rejection/error; never call NOT_RUN a rejection."""
    mapping = validate_mapping(row)
    status = result.get('status', 'NOT_RUN')
    answer = result.get('answer')
    attempted = status != 'NOT_RUN'
    selected = next((c for c in row['candidates']
                     if status == 'VALID' and c['label'] == answer), None)
    return {'status': status, 'answer': answer, 'attempted': attempted,
            'accepted': selected is not None,
            'rejected': status == 'VALID' and answer in {'NONE', 'ABSTAIN'},
            'candidate_id': mapping[answer] if selected else None,
            'selected_bbox_xywh': selected['bbox_xywh'] if selected else None,
            'applied_bbox_xywh': selected['bbox_xywh'] if selected else row['local_bbox_xywh'],
            'action': 'OFFLINE_PROPOSAL' if selected else 'KEEP_LOCAL',
            'closed_loop_writeback': False}


def preprocess_image(source, destination, minimum=28):
    """Explicit shared resize; no processor monkey patch or hidden compatibility layer."""
    from PIL import Image
    source = Path(source)
    destination = Path(destination)
    with Image.open(source) as image:
        image = image.convert('RGB')
        before = list(image.size)
        scale = max(1., minimum / min(image.size))
        after = [math.ceil(v * scale) for v in image.size]
        if before != after:
            image = image.resize(tuple(after), resample=Image.Resampling.BICUBIC)
        destination.parent.mkdir(parents=True, exist_ok=True)
        image.save(destination)
    return {'image': str(destination.resolve()), 'sha256': sha(destination),
            'input_image': str(source.resolve()), 'input_sha256': sha(source),
            'input_size_wh': before, 'output_size_wh': after,
            'policy': 'RGB; uniform upscale only if minimum side <28; ceil; PIL BICUBIC',
            'changes_sampling': before != after, 'shared_by': ['V0', 'Vh', 'C']}


def model_files(path):
    path = Path(path)
    names = sorted(p for p in path.iterdir() if p.is_file() and
                   (p.suffix in {'.json', '.safetensors', '.bin', '.model', '.txt', '.jinja'}))
    if not (path / 'config.json').is_file() or not any(p.suffix in {'.safetensors', '.bin'} for p in names):
        raise FileNotFoundError('Need existing offline Qwen config and weights')
    config = read(path / 'config.json')
    if config.get('model_type') != 'qwen2_5_vl':
        raise ValueError('This adapter requires the existing Qwen2.5-VL model')
    return {p.name: {'bytes': p.stat().st_size, 'sha256': sha(p)} for p in names}


def freeze(args):
    from tracking_lab.e06b_v2_candidates import plan_for
    plan = plan_for(args.run)
    cache_path = args.run / 'cache/candidates.json'
    opportunity = read(args.run / 'opportunity.json')
    rows = read(cache_path)['rows']
    # No GT-based per-case selection: every fixed IR query is retained.
    rows = [r for r in rows if r['modality'] == 'IR']
    if len(rows) != 12 or len({r['case_id'] for r in rows}) != 12:
        raise ValueError('Main protocol requires all twelve original IR queries')
    if opportunity.get('candidate_cache_sha256') != sha(cache_path):
        raise ValueError('Opportunity check does not bind this candidate cache')
    counts = opportunity.get('ir_totals', opportunity.get('totals', opportunity.get('total', {})))
    if counts.get('candidate_ranking_headroom', 0) < 1:
        raise ValueError('No native-rank1 correction opportunity: stop before language calls')
    for row in rows:
        validate_evidence(row, next(s for s in plan['sequences'] if s['sequence'] == row['sequence']))
    target = args.run / 'validation_plan.json'
    if target.exists():
        raise FileExistsError('Validation already frozen; do not replace or retry prompts')
    qwen = args.qwen.resolve()
    frozen = {'experiment': 'E06B_v2', 'frozen_utc': utc(),
              'candidate_plan_sha256': sha(args.run/'plan.json'),
              'candidate_cache_sha256': sha(cache_path),
              'opportunity_sha256': sha(args.run/'opportunity.json'),
              'case_ids': [r['case_id'] for r in rows],
              'max_generation_attempts': 12, 'calls_per_case': 1,
              'automatic_retries': 0, 'extra_diagnostic_calls': 0,
              'prompt_version': PROMPT_VERSION, 'prompt': DIRECT_PROMPT,
              'mode': 'direct', 'features': [], 'qwen': str(qwen),
              'model_files': model_files(qwen), 'max_new_tokens': 192,
              'max_pixels': 262144, 'do_sample': False,
              'preprocess': 'explicit shared minimum-side-28 RGB BICUBIC upscale; no monkey patch',
              'baselines': ['A_current_local', 'G_native_rank1', 'V0_real_init', 'Vh_same_past'],
              'thresholds': {'iou': .5}, 'rejection_policy': 'retain current local; record raw choice separately',
              'subset_policy': 'all twelve original IR points; no per-case GT selection',
              'independent_evaluation': False, 'closed_loop': False,
              'gate': 'candidate opportunity is an offline study stop rule, not a deployed GT gate'}
    write(target, frozen)
    (args.run/'validation_plan.sha256').write_text(sha(target)+'\n', encoding='utf-8')
    print(json.dumps({'frozen': str(target), 'max_attempts': 12}, ensure_ascii=False))


def validation_for(run, verify_model=True):
    from tracking_lab.e06b_v2_candidates import plan_for
    plan_for(run)
    path = run/'validation_plan.json'
    if sha(path) != (run/'validation_plan.sha256').read_text().strip():
        raise ValueError('Validation plan changed')
    frozen = read(path)
    for filename, key in [('plan.json', 'candidate_plan_sha256'),
                          ('cache/candidates.json', 'candidate_cache_sha256'),
                          ('opportunity.json', 'opportunity_sha256')]:
        if sha(run/filename) != frozen[key]:
            raise ValueError('Frozen input changed: '+filename)
    if frozen['prompt'] != DIRECT_PROMPT or frozen['max_generation_attempts'] != 12:
        raise ValueError('Unsupported prompt/budget mutation')
    if verify_model and model_files(frozen['qwen']) != frozen['model_files']:
        raise ValueError('Offline model files changed')
    return frozen


def environment():
    versions = {}
    for name in ('torch', 'transformers', 'numpy', 'Pillow', 'opencv-python'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {'utc': utc(), 'python': sys.version, 'platform': platform.platform(),
            'packages': versions, 'actual_argv': sys.argv}


def vlm(args):
    from tracking_lab.e06b_v2_candidates import plan_for
    frozen = validation_for(args.run)
    plan = plan_for(args.run)
    source_rows = read(args.run/'cache/candidates.json')['rows']
    lookup = {r['case_id']: r for r in source_rows}
    output = args.run/'inference'
    output.mkdir(exist_ok=False)  # No implicit rerun, even after failure.
    write(output/'environment.json', environment())
    runner = E05Qwen(frozen['qwen'], output/'model', max_calls=12,
                     device=args.device, max_pixels=frozen['max_pixels'],
                     max_new_tokens=frozen['max_new_tokens'])
    rows = [lookup[case_id] for case_id in frozen['case_ids']]
    start = time.perf_counter()
    stop_reason = None
    try:
        runner.load()
        runner.metadata.update({'experiment': 'E06B_v2', 'compatibility_monkey_patch': False,
                                'shared_preprocess': frozen['preprocess']})
    except Exception as error:
        stop_reason = {'stage': 'model_load', 'error_class': type(error).__name__, 'error': str(error)}
    for row in rows:
        result = {'status': 'NOT_RUN', 'reason': stop_reason}
        visual = {'V0': {'status': 'NOT_RUN'}, 'Vh': {'status': 'NOT_RUN'}}
        prepared = None
        if stop_reason is None:
            try:
                seq = next(s for s in plan['sequences'] if s['sequence'] == row['sequence'])
                validate_evidence(row, seq)
                folder = output/'shared_images'/row['case_id']
                ref = preprocess_image(row['reference']['image'], folder/'reference.png')
                past = [preprocess_image(h['image'], folder/f'history_{i}.png')
                        for i, h in enumerate(row['history'])]
                candidates = [{**c, **preprocess_image(c['image'], folder/f"candidate_{c['label']}.png")}
                              for c in row['candidates']]
                prepared = {'reference': ref, 'history': past, 'candidates': candidates}
                write(folder/'manifest.json', prepared)
                visual['V0'] = runner.visual_scores(ref['image'], [], candidates)
                visual['Vh'] = runner.visual_scores(ref['image'], [h['image'] for h in past], candidates)
                result = runner.choose(ref['image'], [h['image'] for h in past], candidates,
                                       mode='direct', case_id=row['case_id'])
            except GenerationFailure as error:
                # Reserved failed attempt is still in the persistent ledger.
                result = read(output/'model/events'/error.event_id/'result.json')
                stop_reason = {'stage': 'generation', 'event_id': error.event_id,
                               'reason': 'stop after first failed attempt; no automatic retry'}
            except BudgetExceeded as error:
                stop_reason = {'stage': 'budget', 'error': str(error)}
                result = {'status': 'NOT_RUN', 'reason': stop_reason}
            except Exception as error:
                stop_reason = {'stage': 'preflight_or_visual', 'error_class': type(error).__name__, 'error': str(error)}
                result = {'status': 'NOT_RUN', 'reason': stop_reason}
        prediction = {**row, 'shared_input': prepared, 'visual': visual,
                      'C': result, 'C_action': decision(row, result)}
        append(output/'predictions.jsonl', prediction)
    write(output/'completed.json', {'utc': utc(), 'rows': len(rows),
          'status': 'STOPPED' if stop_reason else 'COMPLETE', 'stop_reason': stop_reason,
          'generation_attempts': runner.calls_used, 'maximum': 12,
          'elapsed_seconds_including_load_visual_io': time.perf_counter()-start,
          'closed_loop': False, 'validation_plan_sha256': sha(args.run/'validation_plan.json')})
    print(json.dumps(read(output/'completed.json'), ensure_ascii=False, indent=2))


def score_predictions(rows, truth):
    """Pure scoring after all predictions; explicit absent/unknown and coverage."""
    if len({r['case_id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate prediction case_id')
    scored = []
    for row in rows:
        g = truth[(row['sequence'], row['source_frame'])]
        if g.get('modality', row['modality']) != row['modality']:
            raise ValueError('Scoring modality differs from inference')
        state = g['state']
        if state not in {'visible', 'absent', 'unknown'}:
            raise ValueError('Unknown annotation state')
        gt = g.get('bbox_xywh')
        native = min(row['candidates'], key=lambda c: c['native_rank'])
        arms = {'A': {'status': 'VALID', 'bbox': row['local_bbox_xywh'], 'accepted': False},
                'G': {'status': 'VALID', 'bbox': native['bbox_xywh'], 'accepted': True}}
        for arm in ('V0', 'Vh'):
            v = row['visual'][arm]
            chosen = next((c for c in row['candidates'] if c['label'] == v.get('answer')), None)
            arms[arm] = {'status': v['status'], 'bbox': chosen['bbox_xywh'] if chosen else None,
                         'accepted': chosen is not None and v['status'] == 'VALID'}
        action = decision(row, row['C'])
        arms['C_raw'] = {'status': action['status'], 'bbox': action['selected_bbox_xywh'],
                         'accepted': action['accepted'], 'rejected': action['rejected']}
        arms['C_applied'] = {'status': action['status'], 'bbox': action['applied_bbox_xywh'],
                             'accepted': action['accepted'], 'rejected': action['rejected']}
        for a in arms.values():
            a['attempted'] = a['status'] != 'NOT_RUN'
            a['iou'] = iou_xywh(a['bbox'], gt) if state == 'visible' and a['bbox'] is not None else None
            a['success'] = state == 'visible' and a['attempted'] and a['iou'] is not None and a['iou'] >= .5
            a['false_accept'] = state == 'absent' and a['accepted'] and a['attempted']
        scored.append({'case_id': row['case_id'], 'sequence': row['sequence'],
                       'source_frame': row['source_frame'], 'state': state, 'arms': arms,
                       'oracle4': state == 'visible' and any(iou_xywh(c['bbox_xywh'], gt) >= .5 for c in row['candidates'])})
    groups = {}
    for seq in sorted({r['sequence'] for r in scored}):
        subset = [r for r in scored if r['sequence'] == seq]
        metrics = {'queries': len(subset), **{state: sum(r['state'] == state for r in subset)
                                            for state in ('visible', 'absent', 'unknown')}}
        metrics['oracle4'] = sum(r['oracle4'] for r in subset)
        metrics['arms'] = {}
        for arm in ('A', 'G', 'V0', 'Vh', 'C_raw', 'C_applied'):
            covered = [r for r in subset if r['arms'][arm]['attempted']]
            metrics['arms'][arm] = {'attempted': len(covered),
                 'visible_n': sum(r['state'] == 'visible' for r in covered),
                 'success': sum(r['arms'][arm]['success'] for r in covered),
                 'absent_n': sum(r['state'] == 'absent' for r in covered),
                 'false_accept': sum(r['arms'][arm]['false_accept'] for r in covered),
                 'absent_correct_rejection': sum(r['state'] == 'absent' and r['arms'][arm].get('rejected', False) for r in covered),
                 'visible_reachable_rejection': sum(r['oracle4'] and r['arms'][arm].get('rejected', False) for r in covered),
                 'unknown_n': sum(r['state'] == 'unknown' for r in covered),
                 'rejected': sum(r['arms'][arm].get('rejected', False) for r in covered),
                 'errors': sum(r['arms'][arm]['status'] == 'ERROR' for r in covered),
                 'invalid': sum(r['arms'][arm]['status'] == 'INVALID' for r in covered),
                 'not_run': len(subset)-len(covered)}
        for c_arm in ('C_raw', 'C_applied'):
            metrics['paired_'+c_arm] = {}
            for base in ('A', 'G', 'V0', 'Vh'):
                paired = [r for r in subset if r['state'] == 'visible' and
                          r['arms'][c_arm]['attempted'] and r['arms'][base]['attempted']]
                fixes = sum(r['arms'][c_arm]['success'] and not r['arms'][base]['success'] for r in paired)
                damages = sum(not r['arms'][c_arm]['success'] and r['arms'][base]['success'] for r in paired)
                metrics['paired_'+c_arm][base] = {'n': len(paired), 'fixes': fixes, 'damages': damages, 'net': fixes-damages}
        groups[seq] = metrics
    return {'rows': scored, 'by_sequence': groups, 'closed_loop': False,
            'interpretation': 'raw choice and offline retain-local policy are separate; NOT_RUN excluded from model denominator'}


def score(args):
    from tracking_lab.e06b_v2_candidates import plan_for, truth_for
    frozen = validation_for(args.run, verify_model=False)
    rows = [json.loads(line) for line in (args.run/'inference/predictions.jsonl').read_text(encoding='utf-8').splitlines() if line]
    if [r['case_id'] for r in rows] != frozen['case_ids']:
        raise ValueError('Predictions must cover every frozen query exactly once in frozen order')
    cached = {r['case_id']: r for r in read(args.run/'cache/candidates.json')['rows']}
    for row in rows:
        source = cached[row['case_id']]
        for key in ('sequence', 'source_frame', 'modality', 'local_bbox_xywh', 'candidates', 'reference', 'history'):
            if row[key] != source[key]:
                raise ValueError('Prediction input differs from frozen cache: '+key)
    report = score_predictions(rows, truth_for(plan_for(args.run)))
    report['execution'] = read(args.run/'inference/completed.json')
    report['prediction_sha256'] = sha(args.run/'inference/predictions.jsonl')
    write(args.run/'report.json', report)
    print(json.dumps(report['by_sequence'], ensure_ascii=False, indent=2))


def main():
    from tracking_lab import e06b_v2_candidates as front
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['prepare', 'cache', 'inspect', 'freeze', 'vlm', 'score'])
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--assets', type=Path, default=ROOT/'assets/e06_target/ir_event_bundle')
    p.add_argument('--ostrack', type=Path, default=ROOT/'assets/e04/ostrack')
    p.add_argument('--include-boat', action='store_true')
    p.add_argument('--boat-assets', type=Path, default=ROOT/'assets/e04')
    p.add_argument('--qwen', type=Path, default=ROOT/'models/qwen25-7b')
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    args = p.parse_args()
    args.run = args.run.resolve()
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    {'prepare': front.prepare, 'cache': front.cache, 'inspect': front.inspect,
     'freeze': freeze, 'vlm': vlm, 'score': score}[args.command](args)


if __name__ == '__main__':
    main()
