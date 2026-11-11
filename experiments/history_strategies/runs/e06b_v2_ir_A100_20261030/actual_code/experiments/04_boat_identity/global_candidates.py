"""Pure full-image box decoding/NMS, reused from the Oct-24 local pilot.
Not an official MSRTrack reproduction; no model, absolute path or GT reads.
"""
import numpy as np

def decode(out, resize, pseudo, height, width, tile_id, search_size=384, grid=24):
    size = out["size_map"][0].permute(1, 2, 0).reshape(-1, 2).cpu().numpy().astype(float)
    offset = out["offset_map"][0].permute(1, 2, 0).reshape(-1, 2).cpu().numpy().astype(float)
    scores = out["score_map"].flatten().cpu().numpy().astype(float)
    scale = search_size / resize
    center = np.array([pseudo[0] + pseudo[2] / 2, pseudo[1] + pseudo[3] / 2])
    indices = np.arange(grid * grid)
    cell = np.column_stack((indices % grid, indices // grid))
    xy = (cell + offset) / grid * scale + center - scale / 2
    wh = size * scale
    xy1 = xy - wh / 2
    xy2 = xy + wh / 2
    xy1[:, 0] = np.clip(xy1[:, 0], 0, width - 10)
    xy1[:, 1] = np.clip(xy1[:, 1], 0, height - 10)
    xy2[:, 0] = np.clip(xy2[:, 0], 10, width)
    xy2[:, 1] = np.clip(xy2[:, 1], 10, height)
    boxes = np.column_stack((xy1, np.maximum(10, xy2 - xy1)))
    return boxes, scores, np.full(grid * grid, tile_id), indices

def nms_top(boxes, scores, tile_ids, grids, top_k=16):
    order = np.argsort(-scores, kind="stable")
    accepted = []
    while len(order) and len(accepted) < top_k:
        i = int(order[0])
        accepted.append(i)
        rest = order[1:]
        if not len(rest):
            break
        b = boxes[i]
        other = boxes[rest]
        wh = np.maximum(0, np.minimum(b[:2] + b[2:], other[:, :2] + other[:, 2:]) - np.maximum(b[:2], other[:, :2]))
        intersection = wh[:, 0] * wh[:, 1]
        overlap = intersection / np.maximum(1e-8, b[2] * b[3] + other[:, 2] * other[:, 3] - intersection)
        order = rest[overlap <= 0.5]
    return [{"rank": rank + 1, "bbox": boxes[i].tolist(), "score": float(scores[i]),
             "tile_id": int(tile_ids[i]), "grid_index": int(grids[i])} for rank, i in enumerate(accepted)]
