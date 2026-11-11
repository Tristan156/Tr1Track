# 证据总览索引（EVIDENCE INDEX）

更新：2026-11-02。本文件索引仓库内全部图表、案例图与视频，按支撑的论点分组；每项附关键数字与来源 run。全部数据来自实验室 A100（服务器 `datasets/` 存量数据），代码与逐例记录见同仓库对应 run 目录。

---

## A. 核心结果（门控闭环）

| 素材 | 内容 | 支撑论点 | 关键数字 |
|---|---|---|---|
| `results/ppt_figs/fig7_core_result.png` | IR 域闭环三臂柱状（不恢复/无条件/门控） | 门控 > 无条件 > 不恢复 | 44.5% → 96.2% → **98.9%**（G_gate 触发仅 4/12 查询点） |
| `results/ppt_figs/fig6_threshold_calibration.png` | 门控阈值-判别一致率曲线（22 查询点） | 阈值域依赖与按域标定 | 船舶域 0.30 在 IR 零触发；**IR 标定 0.50 → TP 7/7、FP 0、100%** |
| `experiments/09_loop_recovery/loop_results.json` | IR 两窗口逐帧 IoU（A vs A+V0） | 闭环恢复机制 | 0.460→0.755、0.254→0.848 |
| `experiments/09_loop_recovery/loop_gated_results.json` | 门控四臂汇总 + 逐查询点诊断（响应分/动作） | 门控 vs 无条件 vs oracle | 86.6% → 95.9% → 98.5%（≈oracle 98.5%） |
| `experiments/10_stageB_closedloop/server_run_results/` | 无 GT 五臂闭环（1435 帧全量） | 无 GT 推理下恢复 +51.7pt | A 44.5% → G_always 96.2%；score_gate(0.30) IR 零触发（域依赖） |

## B. 修正案例（一图看懂"恢复前 vs 恢复后"）

| 素材 | 内容 | 支撑论点 |
|---|---|---|
| `results/case_figs/case1_ir_recovery.png` | 红外无人机：局部丢失 IoU=0.00（红框在背景）→ 恢复后 IoU=0.84（蓝框准确） | 恢复机制的核心案例 |
| `results/case_figs/case2_ship_recovery.png` | 可见光船舶 16-USV：丢失 → 恢复 | 可见光域正例 |
| `results/case_figs/case3_wrong_recovery.png` | 1-Boat：局部正确 0.78 → **误恢复 0.00** | **无条件恢复会改坏好序列 → 必须门控** |

## C. 证据接口标定（shipreid 100 对）

| 素材 | 内容 | 支撑论点 |
|---|---|---|
| `results/shipreid_figs/shipreid_cos_hist.png` | 同船/异船余弦分布 + 最优阈值 0.80 | 视觉特征判别 94% |
| `results/shipreid_figs/case_fp_vlm_yes.png` | **假阳性案例**：一艘小艇 vs 一艘货轮，VLM 判 SAME | 生成式接口判别偏松（异类误接受 32%） |
| `results/shipreid_figs/case_tp_vlm_yes.png` / `case_fn_vlm_no.png` | 真阳性/假阴性案例 | 同船召回 90%、异船仅 68% |
| `experiments/08_shipreid_identity/results/` | 100 对逐例（pairs.json/summary/dino_summary） | 视觉塔 94% ≈ DINO 94% >> 生成式 79%（接口形态 > 模型规模） |

## D. 分布与诊断

| 素材 | 内容 |
|---|---|
| `results/ppt_figs/fig2_score_hist.png` | 跟踪响应分与定位状态的分离（正确组 0.94 vs 丢失组 0.24） |
| `results/ppt_figs/fig5_loop_summary.png` | 两模态闭环成功率汇总 |
| `results/shipreid_figs/shipreid_cos_hist.png` | 同/异船余弦分布（94% 阈值来源） |

## E. 视频（五支，开题答辩演示用）

| 视频 | 内容 | 位置 |
|---|---|---|
| `results/videos/validation_01_..._v0_compare.mp4`（165 帧） | IR01 全序列：GT绿/局部红/V0蓝/Vh橙 四框逐帧 | IR 域历史污染与恢复可视化 |
| `results/videos/validation_02_..._v0_compare.mp4`（122 帧） | IR02 同上（含重现帧） | 同上 |
| `results/videos/dog1_compare.mp4` / `soccer_compare.mp4` / `whiteman1_compare.mp4` | RGBT234 三序列四框对比 | 可恢复跟断补偿 |

## F. 数字对账（双方一致）

- ShipReID：Qwen@0.8 TP45/FN5/TN49/**FP1**=94%；VLM TP45/FN5/TN34/**FP16**=79%；视觉独对 17、语言独对 2、都对 77、都错 4
- IR 状态反馈：A 52/90→86/90（IR01）、27/90→87/90（IR02）
- 船舶闭环宏平均：86.585% → 95.8975%（无条件）→ **98.475%（门控）**；16-USV +46.4pt、1-Boat 无条件改坏 -9.1pt（门控归零）
- IR 闭环成功率：A 44.5% → G_always 96.2% → G_gate(0.50) **98.9%**

## G. 边界（引用时必须同时说明）

- IR 两窗口/船舶 4 序列为**探索性已见数据**，非独立盲测；94% 为样本内标定非留出成绩
- 50 正对中 11 对同相机 token，"全部跨相机更难"的表述已撤回
- 闭环收益部分来自 IR 域 OSTrack 基线较弱（RGB 预训练 OOD）；可见光域收益约 +9.2pt
- 语言生成式接口存在系统性假阳性（异类误接受 32%），当前作为可选补充证据保留，不作为主核验器
