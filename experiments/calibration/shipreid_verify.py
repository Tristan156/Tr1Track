"""shipreid 身份核验标定：同ID(同船) vs 异ID(不同船) 二分类。
三通道：Qwen视觉塔特征余弦 / DINOv2(可选) / VLM生成式判断(YES/NO)。
采样：50 query × (1同ID + 1异ID) = 100 对。【标注：跨相机同ID比跟踪候选更难，如实报告】"""
import os, sys, json, random, glob, time
import numpy as np
import cv2
import torch
from PIL import Image
from pathlib import Path

ROOT = Path("/home/user/code/tracking-vlm-a100-lab")
SRC = Path("/home/user/datasets/reid/shipreid")
OUT = ROOT / "runs/shipreid_verify_20261030"
OUT.mkdir(parents=True, exist_ok=True)
QWEN = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "models/qwen25-7b")
N_QUERY = int(sys.argv[2]) if len(sys.argv) > 2 else 50
MAX_GEN = N_QUERY * 2
random.seed(20261030)

def load_ids(d):
    files = sorted(os.listdir(SRC/d))
    ids = {}
    for f in files:
        if not f.endswith(".jpg"): continue
        cid = f.split("_")[0]
        ids.setdefault(cid, []).append(str(SRC/d/f))
    return ids

def prep(p, min_h=64):
    im = Image.open(p).convert("RGB")
    w, h = im.size
    if h < min_h:
        s = min_h / h
        im = im.resize((max(28,int(w*s)), min_h), Image.BICUBIC)
    return im

def main():
    import re
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    from qwen_vl_utils import process_vision_info
    q_ids = load_ids("query")
    t_ids = load_ids("bounding_box_test")
    tr_ids = load_ids("bounding_box_train")
    pool = {**t_ids}
    for k, v in tr_ids.items(): pool.setdefault(k, []).extend(v)
    common = [i for i in q_ids if i in pool and len(q_ids[i]) >= 1 and len(pool[i]) >= 1]
    print("可配对的 query ID 数:", len(common))
    random.shuffle(common)
    pairs = []
    for cid in common[:N_QUERY]:
        qp = random.choice(q_ids[cid])
        sp = random.choice(pool[cid])
        other = random.choice([i for i in pool if i != cid])
        op = random.choice(pool[other])
        pairs.append({"q": qp, "cand": sp, "same": True, "qid": cid, "cid": cid})
        pairs.append({"q": qp, "cand": op, "same": False, "qid": cid, "cid": other})
    print("配对:", len(pairs), "（同", sum(1 for p in pairs if p["same"]), "/异", sum(1 for p in pairs if not p["same"]), "）")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        QWEN, torch_dtype=torch.bfloat16, local_files_only=True).to("cuda").eval()
    processor = AutoProcessor.from_pretrained(QWEN, max_pixels=262144, local_files_only=True)

    def qvec(pil):
        msgs = [{"role": "user", "content": [{"type": "image", "image": pil}]}]
        txt = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        ii, _ = process_vision_info(msgs)
        inp = processor(text=[txt], images=ii, return_tensors="pt").to("cuda")
        with torch.inference_mode():
            emb = model.visual(inp["pixel_values"], grid_thw=inp["image_grid_thw"])
            v = emb.mean(dim=0)
        return torch.nn.functional.normalize(v, dim=-1).float().cpu().numpy()

    PROMPT = ("The first image is a reference vessel. The second image is a candidate crop. "
              "Decide whether both images show the SAME individual vessel, judging only visible evidence "
              "(hull shape, superstructure, color pattern, markings). Different vessels of the same type "
              "must be answered NO. Do not guess unseen details. "
              'Return exactly one JSON object: {"same": "YES" or "NO", "evidence": "short reason"}')
    rows = []
    budget = 0
    for i, pr in enumerate(pairs):
        ref = prep(pr["q"]); cand = prep(pr["cand"])
        v_q = qvec(ref); v_c = qvec(cand)
        cos = float(np.dot(v_q, v_c))
        gen = None
        if budget < MAX_GEN:
            content = [{"type": "text", "text": "参考（目标）："}, {"type": "image", "image": ref},
                       {"type": "text", "text": "候选：："}, {"type": "image", "image": cand},
                       {"type": "text", "text": PROMPT}]
            msgs = [{"role": "user", "content": content}]
            txt = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            ii, _ = process_vision_info(msgs)
            inp = processor(text=[txt], images=ii, return_tensors="pt").to("cuda")
            t0 = time.time()
            with torch.inference_mode():
                ids = model.generate(**inp, do_sample=False, max_new_tokens=96)
            raw = processor.batch_decode(ids[:, inp.input_ids.shape[1]:], skip_special_tokens=True)[0]
            budget += 1
            m = re.search(r'"same"\s*:\s*"(YES|NO)"', raw)
            ans = m.group(1) if m else "INVALID"
            gen = {"raw": raw.strip(), "answer": ans, "seconds": round(time.time()-t0, 2)}
        rows.append({"q": pr["q"].split("/")[-1], "cand": pr["cand"].split("/")[-1],
                     "same": pr["same"], "qid": pr["qid"], "cid": pr["cid"],
                     "cos": round(cos, 4), "vlm": gen})
        if (i+1) % 10 == 0: print(f"  {i+1}/{len(pairs)} 完成", flush=True)
    json.dump({"pairs": rows, "budget": budget}, open(OUT/"pairs.json", "w"), ensure_ascii=False, indent=1)
    # 汇总
    same = [r for r in rows if r["same"]]; diff = [r for r in rows if not r["same"]]
    import statistics as st
    print("\n=== 视觉特征余弦 ===")
    print("  同ID: 中位 %.3f | 异ID: 中位 %.3f" % (st.median([r["cos"] for r in same]), st.median([r["cos"] for r in diff])))
    # 最佳阈值（在 0-1 网格）
    best = (0, -1)
    for th in [x/100 for x in range(30, 96, 2)]:
        acc = sum(1 for r in rows if (r["cos"] >= th) == r["same"]) / len(rows)
        if acc > best[1]: best = (th, acc)
    print("  最佳阈值 %.2f | 准确率 %.1f%%" % (best[0], best[1]*100))
    print("\n=== VLM 生成式判断 ===")
    ok = sum(1 for r in rows if r["vlm"] and ((r["vlm"]["answer"] == "YES") == r["same"]) and r["vlm"]["answer"] in ("YES","NO"))
    valid = sum(1 for r in rows if r["vlm"] and r["vlm"]["answer"] in ("YES","NO"))
    same_ok = sum(1 for r in same if r["vlm"] and r["vlm"]["answer"] == "YES")
    diff_ok = sum(1 for r in diff if r["vlm"] and r["vlm"]["answer"] == "NO")
    print("  有效 %d/%d | 准确率 %.1f%% | 同ID判对 %d/%d | 异ID判对 %d/%d" % (
        valid, len(rows), ok/len(rows)*100 if rows else 0, same_ok, len(same), diff_ok, len(diff)))
    json.dump({"cos_median_same": st.median([r["cos"] for r in same]),
               "cos_median_diff": st.median([r["cos"] for r in diff]),
               "cos_best_th": best[0], "cos_acc": best[1],
               "vlm_valid": valid, "vlm_acc": ok/len(rows) if rows else 0,
               "vlm_same_ok": same_ok, "vlm_diff_ok": diff_ok, "budget": budget},
              open(OUT/"summary.json", "w"), indent=1)
    print("\nSHIPREID_DONE ->", OUT)

if __name__ == "__main__":
    main()
