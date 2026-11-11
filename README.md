# Tr1Track · 大模型辅助目标跟踪可靠恢复

> 面向舰艇与反无人机场景：跟踪失效状态认知 → 大模型多层证据调度 → 恢复采纳门控。
> 硕士课题主仓库。开题报告、实验证据、核心代码与素材全部在此。

## 核心结果一览

| 闭环场景 | 不恢复 | 无条件恢复 | **门控恢复（按域标定）** |
|---|---|---|---|
| 可见光船舶（4 序列） | 86.6% | 95.9%（改坏 1 序列 -9.1pt） | **98.5%（零损害 ≈ oracle）** |
| 红外无人机（2 序列，1435 帧，无 GT 推理） | 44.5% | 96.2% | **98.9%（按域标定 tau=0.50，按需触发 4/12）** |

**接口形态标定**（shipreid 100 对硬标签）：大模型视觉表征 **94%** ≈ 独立小模型 **94%** >> 生成式接口 **79%**（异类误接受 32% vs 2%）——**证据接口的形态比模型规模更关键**。

## 三个创新点

1. **基于大模型多层表征的跟踪失效状态认知与可恢复性估计**——响应分两域分离（船舶空带 0.36–0.52、IR 空带 0.483–0.614）+ 大模型语义状态描述作为第四类信号。
2. **状态条件化的大模型多层证据调度与恢复采纳门控**——表征层（主核验）/语义层（条件性补充）/元认知层（拒答安全阀）按失效状态调度；语义证据仅在表征层不确定时引入。
3. **大模型语义反馈下的记忆一致性维护与历史证据选择性使用**——历史中毒与语义污染的双重门控：写入门槛、一致性校验、失效回退。

## 仓库结构

```
Tr1Track/
├── README.md                  # 本文件
├── DELIVERY_RULES.md          # 仓库管理规则（必读）
├── EVIDENCE_INDEX.md          # 证据总览索引（全部图表/视频/案例按论点分组）
├── proposal/                  # 开题报告
│   ├── 开题报告_v0.4.md / .docx
│   └── ...
├── figures/                   # 论文/PPT 图表（7 张核心图）
├── case_figs/                 # 修正案例图（恢复前/后对比 + 反例）
├── shipreid_figs/             # 接口标定案例图
├── videos/                    # 演示视频（5 支四框对比）
├── experiments/               # 实验脚本与结果
│   ├── closedloop/            # 无 GT 五臂闭环（兼容层 + 1435 帧结果）
│   ├── gating/                # 门控对照（船舶四序列，oracle 对比）
│   ├── calibration/           # shipreid 100 对接口标定
│   └── history_strategies/    # 历史三策略对比
├── tracking_lab/              # 核心代码（closedloop / e06b_v2 / e05 系）
└── docs/                      # 数据盘点、任务清单、决策记录
```

## 快速复现

```bash
# 无 GT 五臂闭环（需 OSTrack 权重 + Qwen2.5-VL-7B + IR 数据包，见 DELIVERY_RULES §环境）
python closedloop_compat_run.py run  --run runs/closedloop --qwen $QWEN --device cuda
python -m tracking_lab.e06b_v2_closedloop score --run runs/closedloop --ir-root $IR
```

## 相关仓库

- [`tracking-vlm-a100-lab`](https://github.com/Tristan156/tracking-vlm-a100-lab)：服务器实验交付仓库（过程性 run 与审计）
