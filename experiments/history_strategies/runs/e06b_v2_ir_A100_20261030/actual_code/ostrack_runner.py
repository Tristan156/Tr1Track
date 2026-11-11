"""OSTrack-384 official network + official sampling/postprocessing, thin RGB API.

The checkpoint is the ModelScope distribution linked from botaoye/OSTrack.
score is the raw center heatmap value at the Hann-selected location; it is
not a calibrated presence probability. No template updates or training.
"""
from pathlib import Path
import argparse
import copy
import importlib.util
import json
import sys
import time

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT / "deps"), str(ROOT / "OSTrack-main")]
import cv2
import numpy as np
import torch
from lib.models.ostrack import build_ostrack
from lib.config.ostrack.config import cfg, update_config_from_file
from lib.utils.box_ops import clip_box
from lib.utils.ce_utils import generate_mask_cond


def _source_module(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / "OSTrack-main" / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Load these official functions without importing optional dataset/JPEG loaders.
sample_target = _source_module("ostrack_processing", "lib/train/data/processing_utils.py").sample_target
hann2d = _source_module("ostrack_hann", "lib/test/utils/hann.py").hann2d


class OSTrackRunner:
    def __init__(self, device="cuda", checkpoint=None):
        self.device = torch.device(device)
        update_config_from_file(str(ROOT / "OSTrack-main/experiments/ostrack/vitb_384_mae_ce_32x4_ep300.yaml"))
        self.cfg = copy.deepcopy(cfg)
        self.network = build_ostrack(self.cfg, training=False)
        weights = torch.load(checkpoint or ROOT / "pytorch_model.bin", map_location="cpu", weights_only=False)
        self.network.load_state_dict(weights["net"], strict=True)
        self.network = self.network.to(self.device).eval()
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
        sz = self.cfg.TEST.SEARCH_SIZE // self.cfg.MODEL.BACKBONE.STRIDE
        self.window = hann2d(torch.tensor([sz, sz]), centered=True).to(self.device)
        self.state = None
        self.frame_id = 0

    def _tensor(self, image):
        t = torch.as_tensor(np.ascontiguousarray(image), device=self.device).float().permute(2, 0, 1)[None]
        return (t / 255.0 - self.mean) / self.std

    @torch.inference_mode()
    def initialize(self, image_rgb, bbox_xywh):
        """Initialize from uint8 HWC RGB and an [x,y,width,height] pixel box."""
        bbox = [float(v) for v in bbox_xywh]
        if len(bbox) != 4 or bbox[2] <= 0 or bbox[3] <= 0:
            raise ValueError("bbox must contain x,y,w,h with positive w,h")
        patch, _, _ = sample_target(image_rgb, bbox, self.cfg.TEST.TEMPLATE_FACTOR,
                                    output_sz=self.cfg.TEST.TEMPLATE_SIZE)
        self.template = self._tensor(patch)
        # The official 384 configuration uses the fixed CTR_POINT mask.
        self.template_mask = generate_mask_cond(self.cfg, 1, self.device, None)
        self.state = bbox
        self.frame_id = 0
        return {"bbox": self.state.copy(), "score": 1.0, "frame_id": 0}

    @torch.inference_mode()
    def track(self, image_rgb, top_k=0, candidate_nms_iou=0.5):
        if self.state is None:
            raise RuntimeError("initialize must be called first")
        height, width = image_rgb.shape[:2]
        patch, resize_factor, _ = sample_target(image_rgb, self.state, self.cfg.TEST.SEARCH_FACTOR,
                                                output_sz=self.cfg.TEST.SEARCH_SIZE)
        out = self.network(template=self.template, search=self._tensor(patch),
                           ce_template_mask=self.template_mask)
        response = self.window * out["score_map"]
        boxes = self.network.box_head.cal_bbox(response, out["size_map"], out["offset_map"])
        box = (boxes.view(-1, 4).mean(0) * self.cfg.TEST.SEARCH_SIZE / resize_factor).tolist()
        center_x = self.state[0] + 0.5 * self.state[2]
        center_y = self.state[1] + 0.5 * self.state[3]
        half_side = 0.5 * self.cfg.TEST.SEARCH_SIZE / resize_factor
        cx, cy, w, h = box
        mapped = [cx + center_x - half_side - 0.5*w, cy + center_y - half_side - 0.5*h, w, h]
        candidates = []
        if top_k:
            # Decode the same official head at every grid point, then greedily
            # suppress duplicate boxes. All inputs are causal tracker outputs.
            sz = self.cfg.TEST.SEARCH_SIZE // self.cfg.MODEL.BACKBONE.STRIDE
            sizes = out["size_map"][0].permute(1, 2, 0).reshape(-1, 2).cpu().numpy()
            offsets = out["offset_map"][0].permute(1, 2, 0).reshape(-1, 2).cpu().numpy()
            raw_scores = out["score_map"].flatten().cpu().numpy()
            responses = response.flatten().cpu().numpy()
            scale = self.cfg.TEST.SEARCH_SIZE / resize_factor
            for index in np.argsort(-responses, kind="stable"):
                cy_grid, cx_grid = divmod(int(index), sz)
                cx_candidate = (cx_grid + float(offsets[index, 0])) / sz * scale + center_x - half_side
                cy_candidate = (cy_grid + float(offsets[index, 1])) / sz * scale + center_y - half_side
                cw, ch = sizes[index].astype(float) * scale
                cb = clip_box([cx_candidate-cw/2, cy_candidate-ch/2, float(cw), float(ch)], height, width, margin=10)
                duplicate = False
                for previous in candidates:
                    pb = previous["bbox"]
                    ix = max(0., min(cb[0]+cb[2], pb[0]+pb[2])-max(cb[0], pb[0]))
                    iy = max(0., min(cb[1]+cb[3], pb[1]+pb[3])-max(cb[1], pb[1]))
                    intersection = ix*iy
                    iou = intersection / max(1e-8, cb[2]*cb[3]+pb[2]*pb[3]-intersection)
                    if iou > candidate_nms_iou:
                        duplicate = True
                        break
                if not duplicate:
                    candidates.append({"bbox": cb, "score": float(raw_scores[index]),
                                       "windowed_score": float(responses[index]), "grid_index": int(index)})
                if len(candidates) >= top_k:
                    break
        self.state = clip_box(mapped, height, width, margin=10)
        selected = response.flatten().argmax()
        raw_score = out["score_map"].flatten()[selected].item()
        self.frame_id += 1
        result = {"bbox": self.state.copy(), "score": raw_score,
                  "windowed_score": response.flatten()[selected].item(), "frame_id": self.frame_id}
        if top_k:
            result["candidates"] = candidates
            result["candidate_nms_iou"] = candidate_nms_iou
        return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--video")
    p.add_argument("--bbox", type=float, nargs=4, metavar=("X", "Y", "W", "H"))
    p.add_argument("--out", type=Path, default=ROOT / "track.jsonl")
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--start-frame", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--top-k", type=int, default=0)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    tracker = OSTrackRunner(device=args.device)
    if args.smoke:
        image = np.random.default_rng(20261023).integers(0, 256, (480, 640, 3), dtype=np.uint8)
        tracker.initialize(image, [220, 160, 100, 70])
        result = tracker.track(image, top_k=args.top_k)
        result.update({"smoke": True, "device": str(tracker.device), "torch": torch.__version__,
                       "checkpoint_load": "strict=True, no missing/unexpected keys",
                       "finite": bool(np.isfinite(result["bbox"] + [result["score"]]).all())})
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(result, ensure_ascii=False))
        return
    if args.video is None or args.bbox is None:
        p.error("--video and --bbox are required unless --smoke")
    cap = cv2.VideoCapture(args.video)
    if args.start_frame:
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)
    ok, bgr = cap.read()
    if not ok:
        raise RuntimeError("Cannot read first frame: " + args.video)
    result = tracker.initialize(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), args.bbox)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        result["source_frame"] = args.start_frame
        f.write(json.dumps(result) + "\n")
        count = 1
        while not args.max_frames or count < args.max_frames:
            ok, bgr = cap.read()
            if not ok:
                break
            start = time.perf_counter()
            result = tracker.track(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), top_k=args.top_k)
            result["seconds"] = time.perf_counter() - start
            result["source_frame"] = args.start_frame + count
            f.write(json.dumps(result) + "\n")
            count += 1
    cap.release()
    print(json.dumps({"frames": count, "out": str(args.out)}))


if __name__ == "__main__":
    main()
