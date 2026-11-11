"""shipreid DINOv2 独立视觉基线：与 Qwen 视觉塔 94% 对比（同 100 对）。"""
import os, sys, json
import numpy as np, torch
from PIL import Image
from pathlib import Path
ROOT = Path("/home/user/code/tracking-vlm-a100-lab")
RUN = ROOT / "runs/shipreid_verify_20261030"
OUT = RUN

def prep(p, min_h=64):
    im = Image.open(p).convert("RGB")
    w, h = im.size
    if h < min_h:
        s = min_h/h
        im = im.resize((max(28, int(w*s)), min_h), Image.BICUBIC)
    return im

def main():
    from transformers import AutoModel, AutoImageProcessor
    mname = "facebook/dinov2-base"
    model = AutoModel.from_pretrained(mname, local_files_only=True, torch_dtype=torch.float16).to("cuda").eval()
    proc = AutoImageProcessor.from_pretrained(mname, local_files_only=True)
    pairs = json.load(open(RUN/"pairs.json"))["pairs"]
    import statistics as st
    rows = []
    for pr in pairs:
        q = prep("/home/user/datasets/reid/shipreid/" + ("query/" if "query" else "") + pr["q"])
        # pairs.json 里 q/cand 是文件名（不含目录），需要还原目录
        def findp(fn):
            for d in ("query","bounding_box_test","bounding_box_train"):
                p = Path("/home/user/datasets/reid/shipreid")/d/fn
                if p.exists(): return p
            raise FileNotFoundError(fn)
        q = prep(str(findp(pr["q"]))); c = prep(str(findp(pr["cand"])))
        with torch.inference_mode():
            inp = proc(images=[q, c], return_tensors="pt").to("cuda")
            out = model(**inp)
            v = out.last_hidden_state[:, 0]   # CLS
            v = torch.nn.functional.normalize(v.float(), dim=-1).cpu().numpy()
        cos = float(np.dot(v[0], v[1]))
        rows.append({"same": pr["same"], "cos": round(cos, 4)})
    same = [r["cos"] for r in rows if r["same"]]; diff = [r["cos"] for r in rows if not r["same"]]
    best = (0, -1)
    for th in [x/100 for x in range(20, 99, 2)]:
        acc = sum(1 for r in rows if (r["cos"] >= th) == r["same"]) / len(rows)
        if acc > best[1]: best = (th, acc)
    print("=== DINOv2-base（独立视觉基线，同 100 对）===")
    print("  同ID 中位 %.3f | 异ID 中位 %.3f" % (st.median(same), st.median(diff)))
    print("  最佳阈值 %.2f | 准确率 %.1f%%" % (best[0], best[1]*100))
    json.dump({"model": mname, "median_same": st.median(same), "median_diff": st.median(diff),
               "best_th": best[0], "acc": best[1], "n": len(rows)},
              open(OUT/"dino_summary.json", "w"), indent=1)
    print("DONE")

if __name__ == "__main__":
    main()
