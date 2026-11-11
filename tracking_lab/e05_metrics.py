"""CPU-only calibration and scoring for fixed-proposal E05 diagnostics.

This module never loads a model, edits predictions, or claims closed-loop gains.
Truth is loaded by the caller only after inference. Missing model outputs are
NOT_RUN, not invented refusals. All synthetic tests are software fixtures only.
"""
from __future__ import annotations

from collections import Counter
import copy
import math
from typing import Any


DEV_ROLES = frozenset({'dev', 'development'})
EVAL_ROLES = frozenset({'eval', 'evaluation', 'exploratory_evaluation', 'test', 'validation'})
REJECTIONS = frozenset({'NONE', 'ABSTAIN'})
RAW_GRID = (0.0, 0.2, 0.4, 0.6, 0.8, 1.01)
SIM_GRID = (-1.0, 0.0, 0.2, 0.4, 0.6, 0.8, 0.9)
B2_SIM_GRID = (-2.0, -1.0, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 2.01)
MARGIN_GRID = (0.0, 0.02, 0.05, 0.1)
ARMS = ('local', 'B0', 'B1', 'B2', 'rawmax', 'fixed_A', 'fixed_D', 'direct', 'qa')


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _valid_box(box: Any) -> bool:
    return (isinstance(box, (list, tuple)) and len(box) == 4
            and all(_finite(v) for v in box) and box[2] > 0 and box[3] > 0)


def iou_xywh(a: list[float], b: list[float]) -> float:
    if not _valid_box(a) or not _valid_box(b):
        raise ValueError('IoU requires finite positive-extent xywh boxes')
    width = max(0.0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    height = max(0.0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    intersection = width * height
    return intersection / (a[2] * a[3] + b[2] * b[3] - intersection)


def _ratio(numerator: int | float, denominator: int | float) -> float | None:
    return numerator / denominator if denominator else None


def _index_truth(truth: dict) -> dict:
    """Accept the delivered evaluation.json or a sequence/local_index index."""
    if 'sequences' not in truth:
        return truth
    result = {}
    for sequence in truth['sequences']:
        name = sequence['sequence']
        if name in result:
            raise ValueError('Duplicate truth sequence: ' + name)
        frames = {}
        for frame in sequence['frames']:
            index = frame['local_index']
            if not isinstance(index, int) or isinstance(index, bool) or index < 1 or index in frames:
                raise ValueError('Truth local_index must be unique and one based')
            frames[index] = frame
        result[name] = frames
    return result


def _score_dict(row: dict, key: str, labels: set[str]) -> dict | None:
    value = row.get(key)
    if value is None:
        return None
    # The adapter may return its full visual_scores metadata object.
    if isinstance(value, dict) and isinstance(value.get('scores'), dict):
        value = value['scores']
    if not isinstance(value, dict) or set(value) != labels or not all(_finite(x) for x in value.values()):
        raise ValueError(key + ' must give one finite score for every supplied candidate')
    return dict(value)


def _prepare(rows: list[dict], truth: dict) -> list[dict]:
    index = _index_truth(truth)
    cases, seen = [], set()
    for row in rows:
        seq, frame = row['sequence'], row['frame_index']
        if not isinstance(seq, str) or not isinstance(frame, int) or isinstance(frame, bool) or frame < 0:
            raise ValueError('Need sequence and zero-based nonnegative frame_index')
        identity = (seq, frame)
        if identity in seen:
            raise ValueError('Duplicate sequence/frame prediction')
        seen.add(identity)
        role = row['role']
        if role not in DEV_ROLES | EVAL_ROLES:
            raise ValueError('Unknown role: ' + str(role))
        frames = index.get(seq, {})
        gt = frames.get(frame + 1, frames.get(str(frame + 1)))
        if gt is None:
            raise ValueError('Missing truth frame for ' + str(identity))
        if 'source_frame' in gt and row.get('source_frame') != gt['source_frame']:
            raise ValueError('Source-frame/zero-based-index mapping differs')
        if not _valid_box(row['local_bbox_xywh']):
            raise ValueError('Invalid post-local-track bbox')
        candidates = row['candidates']
        if not isinstance(candidates, list) or not 1 <= len(candidates) <= 4:
            raise ValueError('Expected one to four primary proposals')
        labels, ranks = set(), set()
        for candidate in candidates:
            label, rank = candidate['label'], candidate['original_rank']
            if label not in {'A', 'B', 'C', 'D'} or label in labels:
                raise ValueError('Candidate labels must be unique A-D')
            if not isinstance(rank, int) or isinstance(rank, bool) or rank < 1 or rank in ranks:
                raise ValueError('original_rank must be unique positive integers')
            if not _valid_box(candidate['bbox_xywh']) or not _finite(candidate['tracker_score']):
                raise ValueError('Malformed proposal bbox/response')
            labels.add(label); ranks.add(rank)
        if 1 not in ranks:
            raise ValueError('Primary proposal pool must contain native original_rank=1')
        if frame == 0:
            state = 'initialization'
        elif gt.get('absence') == 1:
            state = 'verified_absent'
        elif gt.get('bbox_valid'):
            if not _valid_box(gt.get('bbox_xywh')):
                raise ValueError('bbox_valid contradicts ground-truth extent')
            state = 'annotated'
        else:
            state = 'unknown'
        ious = {c['label']: iou_xywh(c['bbox_xywh'], gt['bbox_xywh']) if state == 'annotated' else None
                for c in candidates}
        native = next(c for c in candidates if c['original_rank'] == 1)
        cases.append({'row': row, 'sequence': seq, 'role': role, 'frame_index': frame,
                      'source_frame': row.get('source_frame'), 'state': state, 'gt': gt,
                      'candidate_ious': ious, 'reachable': any(v is not None and v >= .5 for v in ious.values()) if state == 'annotated' else None,
                      'native_label': native['label'], 'visual': _score_dict(row, 'visual_scores', labels),
                      'contrastive_visual': _score_dict(row, 'contrastive_visual_scores', labels)})
    return cases


def _visual_choice(case: dict, scores: dict | None, similarity: float, margin: float) -> str | None:
    if scores is None:
        return None
    ranked = sorted(case['row']['candidates'], key=lambda c: (-scores[c['label']], c['original_rank']))
    first = ranked[0]['label']
    gap = scores[first] - scores[ranked[1]['label']] if len(ranked) > 1 else math.inf
    return first if scores[first] >= similarity and gap >= margin else 'NONE'


def _choice(case: dict, arm: str, settings: dict) -> tuple[str | None, dict | None]:
    row, candidates = case['row'], case['row']['candidates']
    if arm == 'local':
        return 'LOCAL', None
    if arm == 'B0':
        native = next(c for c in candidates if c['original_rank'] == 1)
        return (native['label'] if native['tracker_score'] >= settings['b0_threshold'] else 'NONE'), None
    if arm in {'B1', 'B2'}:
        prefix = arm.lower()
        scores = case['visual'] if arm == 'B1' else case['contrastive_visual']
        if scores is None:
            return None, None
        return _visual_choice(case, scores, settings[prefix + '_sim'], settings[prefix + '_margin']), None
    if arm == 'rawmax':
        return min(candidates, key=lambda c: (-c['tracker_score'], c['original_rank']))['label'], None
    if arm.startswith('fixed_'):
        label = arm[-1]
        return (label if label in case['candidate_ious'] else 'INVALID'), None
    output = row.get(arm)
    if output is None:
        return None, None
    if not isinstance(output, dict):
        raise ValueError('Model arm output must be an object or None')
    if output.get('status') == 'NOT_RUN':
        return None, None
    answer = output.get('answer')
    if output.get('status') not in (None, 'VALID') or answer not in set(case['candidate_ious']) | REJECTIONS:
        return 'INVALID', output
    return answer, output


def _outcome(case: dict, arm: str, settings: dict) -> dict:
    answer, supplied = _choice(case, arm, settings)
    attempted = answer is not None
    accepted = attempted and answer in case['candidate_ious']
    annotated = case['state'] == 'annotated'
    by_label = {c['label']: c for c in case['row']['candidates']}
    applied = by_label[answer]['bbox_xywh'] if accepted else case['row']['local_bbox_xywh']
    applied_iou = iou_xywh(applied, case['gt']['bbox_xywh']) if annotated and attempted else None
    selected_iou = case['candidate_ious'].get(answer) if accepted else None
    seconds = None
    if supplied:
        for key in ('seconds', 'elapsed_seconds'):
            if _finite(supplied.get(key)) and supplied[key] >= 0:
                seconds = supplied[key]; break
    return {'arm': arm, 'attempted': attempted, 'answer': answer if attempted else 'NOT_RUN',
            'accepted': bool(accepted), 'rejected': answer in REJECTIONS,
            'invalid': answer == 'INVALID', 'candidate_choice_iou': selected_iou,
            'candidate_choice_success': bool(accepted and selected_iou >= .5) if annotated and attempted else None,
            'applied_bbox_xywh': list(applied) if attempted else None,
            'applied_action': ('local_baseline' if arm == 'local' else 'overwrite_search_state' if accepted else 'keep_local_state') if attempted else 'NOT_RUN',
            'applied_iou': applied_iou, 'applied_success': applied_iou >= .5 if applied_iou is not None else None,
            'wrong_accept': bool(accepted and (case['state'] == 'verified_absent' or annotated and selected_iou < .5)),
            'false_reject_reachable': bool(answer in REJECTIONS and annotated and case['reachable']),
            'invalid_reachable': bool(answer == 'INVALID' and annotated and case['reachable']),
            'selected_original_rank': by_label[answer]['original_rank'] if accepted else None,
            'seconds': seconds}


def _development_loss(cases: list[dict], arm: str, settings: dict) -> tuple[int, int]:
    loss, accepted = 0, 0
    for case in cases:
        if case['state'] not in {'annotated', 'verified_absent'}:
            continue
        result = _outcome(case, arm, settings)
        if not result['attempted']:
            raise ValueError('Missing ' + arm + ' feature record in development')
        loss += int(result['wrong_accept']) + int(result['false_reject_reachable']) + int(result['invalid_reachable'])
        accepted += int(result['accepted'] and case['state'] == 'annotated')
    return loss, accepted


def calibrate(rows: list[dict], truth: dict, *, manifest_sha256: str | None = None,
              raw_grid=RAW_GRID, sim_grid=SIM_GRID, margin_grid=MARGIN_GRID,
              b2_sim_grid=B2_SIM_GRID) -> dict:
    """Choose thresholds and main B using development truth only, never eval results."""
    cases = _prepare([r for r in rows if r.get('role') in DEV_ROLES], truth)
    if not cases or not any(c['state'] in {'annotated', 'verified_absent'} for c in cases):
        raise ValueError('Need known-truth development queries for calibration')
    for grid in (raw_grid, sim_grid, margin_grid, b2_sim_grid):
        if not grid or not all(_finite(value) for value in grid):
            raise ValueError('Calibration grid must be finite and nonempty')
    settings = {'schema_version': 1, 'b0_threshold': float(raw_grid[0]), 'main_b': 'B0',
                'manifest_sha256': manifest_sha256,
                'calibration_scope': 'development roles only; no test-selected winner',
                'loss_definition': 'wrong accepted proposal + rejected reachable query; UNKNOWN/init excluded',
                'calibration_records': [], 'available_visual_arms': ['B0']}
    best_key, best_value = None, None
    for i, threshold in enumerate(raw_grid):
        settings['b0_threshold'] = float(threshold)
        loss, accepted = _development_loss(cases, 'B0', settings)
        key = (loss, -accepted, i)
        if best_key is None or key < best_key:
            best_key, best_value = key, float(threshold)
    settings['b0_threshold'] = best_value
    metrics = {'B0': {'loss': best_key[0], 'valid_acceptance_count': -best_key[1]}}
    for arm, field, similarities in (('B1', 'visual', sim_grid), ('B2', 'contrastive_visual', b2_sim_grid)):
        present = [c[field] is not None for c in cases]
        if not any(present):
            continue
        if not all(present):
            raise ValueError('Do not partially omit ' + arm + ' development records')
        best_key, best_value = None, None
        prefix = arm.lower()
        for i, similarity in enumerate(similarities):
            for j, margin in enumerate(margin_grid):
                settings[prefix + '_sim'], settings[prefix + '_margin'] = float(similarity), float(margin)
                loss, accepted = _development_loss(cases, arm, settings)
                key = (loss, -accepted, i, j)
                if best_key is None or key < best_key:
                    best_key, best_value = key, (float(similarity), float(margin))
        settings[prefix + '_sim'], settings[prefix + '_margin'] = best_value
        settings['available_visual_arms'].append(arm)
        metrics[arm] = {'loss': best_key[0], 'valid_acceptance_count': -best_key[1]}
    # Predetermined priority is only a final tie breaker, not an eval-time choice.
    priority = {'B2': 0, 'B1': 1, 'B0': 2}
    settings['main_b'] = min(metrics, key=lambda a: (metrics[a]['loss'], -metrics[a]['valid_acceptance_count'], priority[a]))
    settings['development_by_arm'] = metrics
    settings['calibration_records'] = [{'sequence': c['sequence'], 'frame_index': c['frame_index'], 'state': c['state']} for c in cases]
    return settings


def _summarize(outcomes: list[tuple[dict, dict]]) -> dict:
    present = [(case, result) for case, result in outcomes if result['attempted']]
    valid = [(c, r) for c, r in present if c['state'] == 'annotated']
    accepted_known = [(c, r) for c, r in present if c['state'] in {'annotated', 'verified_absent'} and r['accepted']]
    successes = sum(bool(r['applied_success']) for _, r in valid)
    candidate_successes = sum(bool(r['candidate_choice_success']) for _, r in valid)
    times = [r['seconds'] for _, r in present if r['seconds'] is not None]
    return {'attempted': len(present), 'not_run': len(outcomes) - len(present),
            'valid_localization_n': len(valid), 'unknown_n': sum(c['state'] == 'unknown' for c, _ in present),
            'verified_absent_n': sum(c['state'] == 'verified_absent' for c, _ in present),
            'initialization_excluded_n': sum(c['state'] == 'initialization' for c, _ in present),
            'applied_success_count': successes, 'applied_success_rate': _ratio(successes, len(valid)),
            'mean_applied_iou': _ratio(sum(r['applied_iou'] for _, r in valid), len(valid)),
            'candidate_choice_success_count': candidate_successes, 'candidate_choice_success_rate': _ratio(candidate_successes, len(valid)),
            'accepted': sum(r['accepted'] for _, r in present), 'coverage': _ratio(sum(r['accepted'] for _, r in present), len(present)),
            'rejected': sum(r['rejected'] for _, r in present), 'invalid': sum(r['invalid'] for _, r in present),
            'wrong_accept': sum(r['wrong_accept'] for _, r in present),
            'accept_precision_known': _ratio(sum(bool(r['candidate_choice_success']) for _, r in accepted_known), len(accepted_known)),
            'false_reject_reachable': sum(r['false_reject_reachable'] for _, r in present),
            'invalid_reachable': sum(r['invalid_reachable'] for _, r in present),
            'answer_counts': dict(Counter(r['answer'] for _, r in present)),
            'supplied_timed_records': len(times), 'supplied_seconds_sum': sum(times) if times else None}


def _pair(scored: list[dict], arm: str, baseline: str) -> dict:
    comparable = [c for c in scored if c['state'] == 'annotated'
                  and c['arms'][arm]['attempted'] and c['arms'][baseline]['attempted']]
    fixes = [c for c in comparable if c['arms'][arm]['applied_success'] and not c['arms'][baseline]['applied_success']]
    damages = [c for c in comparable if not c['arms'][arm]['applied_success'] and c['arms'][baseline]['applied_success']]
    def ids(items):
        return [{'sequence': c['sequence'], 'frame_index': c['frame_index'], 'source_frame': c['source_frame']} for c in items]
    return {'paired_valid_n': len(comparable), 'fixes': len(fixes), 'damages': len(damages), 'net': len(fixes) - len(damages),
            'fix_queries': ids(fixes), 'damage_queries': ids(damages)}


def _group(scored: list[dict], main_b: str) -> dict:
    valid = [c for c in scored if c['state'] == 'annotated']
    reachable = sum(c['reachable'] for c in valid)
    native = sum(c['candidate_ious'][c['native_label']] >= .5 for c in valid)
    arms = {arm: _summarize([(c, c['arms'][arm]) for c in scored]) for arm in ARMS}
    pairs = {arm: {'vs_main_b': _pair(scored, arm, main_b), 'vs_local': _pair(scored, arm, 'local')}
             for arm in ('B0', 'B1', 'B2', 'direct', 'qa')}
    return {'queries': len(scored), 'valid_localization_n': len(valid),
            'unknown_n': sum(c['state'] == 'unknown' for c in scored),
            'verified_absent_n': sum(c['state'] == 'verified_absent' for c in scored),
            'recall_at_k_count': reachable, 'recall_at_k_rate': _ratio(reachable, len(valid)),
            'native_top1_success_count': native, 'native_top1_success_rate': _ratio(native, len(valid)),
            'candidate_headroom_count': reachable - native, 'candidate_headroom_rate': _ratio(reachable - native, len(valid)),
            'has_fixed_pool_accuracy_gain_space': bool(reachable > native),
            'headroom_interpretation': 'Recall minus native rank1 is only a fixed-pool oracle opportunity, never an achieved gain.',
            'arms': arms, 'paired': pairs}


def measure(rows: list[dict], truth: dict, settings: dict | None = None) -> dict:
    """Score frozen decisions; never tune settings on evaluation rows."""
    if settings is None:
        settings = calibrate(rows, truth)
    settings = copy.deepcopy(settings)
    required = {'b0_threshold', 'main_b'}
    if not required <= settings.keys() or settings['main_b'] not in {'B0', 'B1', 'B2'}:
        raise ValueError('Missing or malformed frozen settings')
    cases = _prepare(rows, truth)
    for arm, field in (('B1', 'visual'), ('B2', 'contrastive_visual')):
        if any(c[field] is not None for c in cases):
            if not all(_finite(settings.get(arm.lower() + suffix)) for suffix in ('_sim', '_margin')):
                raise ValueError('Missing finite thresholds for present ' + arm)
    scored = []
    for c in cases:
        record = {key: value for key, value in c.items() if key not in {'row', 'gt', 'visual', 'contrastive_visual'}}
        record['arms'] = {arm: _outcome(c, arm, settings) for arm in ARMS}
        scored.append(record)
    main_b = settings['main_b']
    if any(not c['arms'][main_b]['attempted'] for c in scored):
        raise ValueError('Frozen main B is missing on some scored queries')
    per_sequence = {seq: _group([c for c in scored if c['sequence'] == seq], main_b) for seq in sorted({c['sequence'] for c in scored})}
    summary = _group(scored, main_b)
    macro = {}
    for arm in ARMS:
        macro[arm] = {}
        for field in ('applied_success_rate', 'mean_applied_iou', 'candidate_choice_success_rate', 'coverage'):
            values = [g['arms'][arm][field] for g in per_sequence.values() if g['arms'][arm][field] is not None]
            macro[arm][field] = _ratio(sum(values), len(values))
        macro[arm]['sequences_with_valid_localization'] = sum(g['arms'][arm]['valid_localization_n'] > 0 for g in per_sequence.values())
    dev = _group([c for c in scored if c['role'] in DEV_ROLES], main_b)
    eva = _group([c for c in scored if c['role'] in EVAL_ROLES], main_b)
    qa_pair = dev['paired']['qa']['vs_main_b']
    qa_local = dev['paired']['qa']['vs_local']
    signal = (dev['candidate_headroom_count'] > 0 and qa_pair['paired_valid_n'] > 0
              and qa_pair['net'] >= 1 and qa_local['net'] >= 0)
    return {'schema_version': 1, 'scope': 'Fixed-proposal offline scoring; no closed-loop recovery claim',
            'frozen_settings': settings,
            'metric_definitions': {'localization': 'IoU>=0.5, excluding first frame, UNKNOWN and verified absence',
                'candidate_choice': 'The selected supplied box is correct; NONE/ABSTAIN do not count as a correct identity selection.',
                'applied': 'Accept a supplied box, otherwise keep this same frame post-local-track bbox. NOT_RUN is never scored.',
                'recall': 'Any supplied primary candidate has IoU>=0.5; never reuse a selector success count.',
                'native': 'original_rank=1, independent of label order; rawmax is separately diagnostic.',
                'risk': 'Wrong accepted proposals / known accepted proposals; operational coverage includes UNKNOWN.',
                'aggregation': 'summary is actual valid-frame weighting; macro weights sequences equally. Development/evaluation stay separately available.'},
            'comparison_warning': 'Per-arm output coverage may differ. Compare direct/qa with main B only on their matched query IDs; do not rank unpaired rates with different denominators.',
            'summary': summary, 'weighted': summary['arms'], 'macro': macro,
            'development': dev, 'evaluation': eva,
            'by_role': {role: _group([c for c in scored if c['role'] == role], main_b) for role in sorted({c['role'] for c in scored})},
            'per_sequence': per_sequence, 'rows': scored,
            'development_signal': {'candidate_headroom_count': dev['candidate_headroom_count'],
                'qa_vs_main_b_net': qa_pair['net'], 'qa_paired_valid_n': qa_pair['paired_valid_n'],
                'qa_vs_local_net': qa_local['net'], 'local_recovery_pilot_signal': signal and qa_local['net'] > 0,
                'further_investment_signal': signal, 'auto_evaluation_authorized': False,
                'meaning': 'Development headroom>0, QA vs main B net>=1 and QA vs local net>=0 is only an investment signal. Actual local recovery additionally needs QA vs local net>0; neither automatically unlocks test clips or establishes novelty.'}}
