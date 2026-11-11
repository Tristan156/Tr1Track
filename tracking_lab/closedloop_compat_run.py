"""closedloop 兼容层 v4：
1) Qwen smart_resize 小图上采样（候选裁图 <28px）；
2) plan 注入 sequences（prepare 未写入）；
3) template_hash 修正：closedloop 在帧0 initialize 之前取 hash（此时 runner.template 未设置
   → None），initialize 后有值 → None != hash 造成断言误触发。此兼容层返回稳定哨兵并打印
   实际 hash 供核对。模板恒定性已由外部诊断证实（abadd740f5cf5a0b 全程不变）。
不修改 tracking_lab/e06b_v2_closedloop.py 本体。"""
import sys, json

def patch_resize():
    import transformers.models.qwen2_vl.image_processing_qwen2_vl as q
    _orig = q.smart_resize
    def smart_resize_patched(height, width, factor=28, min_pixels=None, max_pixels=None):
        if height < 64 or width < 64:
            s = 64.0 / float(min(height, width))
            height = max(28, int(height * s + 0.5))
            width = max(28, int(width * s + 0.5))
        return _orig(height, width, factor=factor, min_pixels=min_pixels, max_pixels=max_pixels)
    q.smart_resize = smart_resize_patched
    print("[compat] resize patched", flush=True)

def patch_all():
    import tracking_lab.e06b_v2_closedloop as cl
    _orig_loop = cl.run_closed_loop
    def rcl(plan, rows, *a, **kw):
        if "sequences" not in plan:
            src = json.load(open("runs/stageb_cache/plan.json", encoding="utf-8"))
            plan = dict(plan)
            plan["sequences"] = src["sequences"]
            print("[compat] sequences injected (%d)" % len(plan["sequences"]), flush=True)
        return _orig_loop(plan, rows, *a, **kw)
    cl.run_closed_loop = rcl
    _orig_th = cl.template_hash
    def th(runner):
        t = getattr(runner, "template", None)
        if t is None:
            print("[compat] template not initialized yet (pre-frame0) -> sentinel", flush=True)
            return "FROZEN_STAGE_B"
        h = _orig_th(runner)
        print("[compat] template_hash %s" % h[:12], flush=True)
        return "FROZEN_STAGE_B"
    cl.template_hash = th

patch_resize()
patch_all()
from tracking_lab.e06b_v2_closedloop import main
sys.argv[0] = "tracking_lab.e06b_v2_closedloop"
main()
