"""Small E04 protocol helpers; no model loading, GT reading, or inference."""
from __future__ import annotations
import hashlib
import math
import random


def normalize_candidates(raw: list[dict]) -> list[dict]:
    """Bridge the old decoder to the strict E04/VLM schema; keep diagnostics separately."""
    return [{'bbox_xywh': list(candidate['bbox']), 'tracker_score': float(candidate['score']),
             'original_rank': int(candidate['rank'])} for candidate in raw]


def query_indices(frame_count: int, cap: int = 10, spacing: int = 10) -> list[int]:
    """Leave >=5 subsequent frames; never schedule by ground truth or scores."""
    if frame_count < 0 or cap < 0 or spacing < 1:
        raise ValueError('Invalid schedule argument')
    first, last = 10, frame_count - 6
    if last < first or cap == 0:
        return []
    count = min(cap, 1 + (last - first) // spacing)
    if count == 1:
        return [first]
    return [first + i * (last - first) // (count - 1) for i in range(count)]


def shuffled_labels(candidates: list[dict], sequence: str, frame: int, seed: int = 20261028) -> list[dict]:
    """Same opaque ordering for the same fixed query; never use scores or GT."""
    if not 1 <= len(candidates) <= 4:
        raise ValueError('Need one to four candidates')
    key = hashlib.sha256(f'{seed}:{sequence}:{frame}'.encode()).hexdigest()
    order = list(range(len(candidates)))
    random.Random(int(key, 16)).shuffle(order)
    return [{**candidates[index], 'label': chr(65 + rank)} for rank, index in enumerate(order)]


def apply_choice(local_xywh: list[float], candidates: list[dict], answer: str) -> dict:
    """Rejecting a proposal keeps the *post-local-track* state, not previous state."""
    by_label = {candidate['label']: candidate for candidate in candidates}
    if len(by_label) != len(candidates):
        raise ValueError('Duplicate candidate label')
    if answer in by_label:
        box = list(by_label[answer]['bbox_xywh'])
        action = 'overwrite_search_state'
    else:
        box = list(local_xywh)
        action = 'keep_local_state'
        if answer not in ('NONE', 'ABSTAIN', 'INVALID'):
            answer = 'INVALID'
    if len(box) != 4 or not all(math.isfinite(v) for v in box) or min(box[2:]) <= 0:
        raise ValueError('Invalid applied bbox')
    return {'answer': answer, 'bbox_xywh': box, 'action': action, 'template_update': False}


def may_continue(per_sequence: list[dict]) -> bool:
    """Exploratory investment criterion, not statistical significance."""
    if len(per_sequence) < 2:
        return False
    net = [row['fixes'] - row['damages'] for row in per_sequence]
    return all(value >= 0 for value in net) and sum(net) >= 2
