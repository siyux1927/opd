# OPD / OPD+：单卡 A100 上的 On-Policy Distillation 复现与 advantage 修正验证

把 **Qwen3-4B-Instruct-2507** 的数学推理能力，用 on-policy distillation 蒸馏进 **Qwen3-0.6B-Base**，
并在同一套框架里验证 **OPD+**（[arXiv:2606.01039](https://arxiv.org/abs/2606.01039)）提出的
advantage 修正：现行 OPD 实现对 reward 做 stop-gradient 会带来有偏梯度，正确的逐 token advantage 应当是

$$w_f(u) = -f(u) + u f'(u)$$

其中 $u$ 是同一个 token 位置上教师与学生的概率比 $q_{\mathrm{teacher}} / p_{\theta}$。

全流程从零手写（不依赖 TRL / verl / tinker），单张 A100 40GB 约 2 小时跑完 9 个对照臂。

---

## 1. 为什么做这个题

On-policy distillation 是 2025 年底以来后训练方向最实用的一条线：学生自己 rollout、教师逐 token 打分，
既避免了 SFT 的 exposure bias（学生永远见不到自己的错误状态），又避免了 outcome-reward RL 的稀疏信用分配。
Thinking Machines Lab 在 [2025-10 的博客](https://thinkingmachines.ai/blog/on-policy-distillation)里
用 Qwen3-32B 教 Qwen3-8B，在 AIME'24 上以约 1/10 的算力达到 RL 的效果；据 OPD+ 引用，DeepSeek-V4
的后训练配方中已包含 OPD。**"用大模型把小模型拉起来，再把小模型部署出去"**——这正是大厂降本增效的主线之一。

但这条线上有一个几乎没人验证过的细节。OPD 的 reward 是 $-f(u)$，而 $u$ 依赖学生参数 $\theta$；
业界为了数值稳定统一做了 stop-gradient（动机类比 DQN 的 target network）。OPD+ 证明这在数学上是有偏的，
并给出了修正项 $u f'(u)$。

| 散度 | 现行 OPD (stop-grad) advantage | OPD+ 修正后 advantage | 修正项 |
| --- | --- | --- | --- |
| Forward KL | $-u\ln u$ | $u$ | 非常数 |
| Reverse KL | $\ln u$ | $\ln u - 1$ | **常数 $-1$** |
| JSD | $-\frac12\left[u\ln u-(1+u)\ln\frac{1+u}{2}\right]$ | $\frac12\ln\frac{1+u}{2}$ | 非常数 |

关键洞察在于 Reverse KL 的修正项恰好是常数，在 score function identity
$\mathbb{E}_{p_\theta}[\nabla_\theta \log p_\theta] = 0$ 下作为 baseline 消失。
**这就解释了为什么整个业界"只有 reverse KL 能用"**——不是 forward KL 和 JSD 本身不行，
而是它们在 stop-gradient 约定下拿到的是错误的梯度系数。

本项目要在小规模上回答三个问题：

1. Forward KL / JSD 在 stop-gradient OPD 下是否真的会崩？
2. 换成 $w_f(u)$ 之后是否能被"救活"？
3. 在相同 rollout 预算下，dense 的教师信号相比稀疏的 outcome reward（GRPO）便宜多少？

## 2. 一图说清崩溃机理

<p align="center">
  <img src="figures/fig1_weight_functions.png" width="66%" />
  <img src="figures/fig2_log_ratio_distribution.png" width="32%" />
</p>

左图把三种散度的 $-f(u)$ 与 $w_f(u)$ 画在同一张对数横轴上。Forward KL 的崩溃原因一眼可见：
$-u\ln u$ 在 $u \to 0$ 时趋于 0。也就是说，
**教师很想出、但学生几乎不会出的那些 token 恰恰拿不到梯度**，
而 $-u\ln u$ 在 $u > 1$ 区间还是负的——等于在惩罚学生去靠近教师。修正成 $u$ 之后，
advantage 变成单调正的无界推力，方向才是对的。

右图叠加了训练中实际采样 token 的 $\log u$ 分布，说明这些区间不是理论边角，而是真实高频落点。

## 3. 实验设计

统一从同一个 SFT-init checkpoint 出发，rollout 预算完全一致，只改 advantage 的一行。

| 臂 | 目标 | 说明 |
| --- | --- | --- |
| `sft_init` | 学生初始化 | Qwen3-0.6B-Base 在 2k 条 GSM8K 人工 CoT 上 SFT，所有臂的共同起点 |
| `sft_more_data` | off-policy 对照 | 再喂 4k 条数据，回答"多给点 SFT 数据行不行" |
| `grpo` | outcome-reward RL 对照 | 组内归一化的答案正确性奖励，稀疏信号 |
| `{reverse_kl,forward_kl,jsd}_opd` | OPD 基线 | advantage $=-f(u)$，复现现行做法 |
| `{reverse_kl,forward_kl,jsd}_opd_plus` | OPD+ | advantage $=w_f(u)$ |

- 教师 Qwen3-4B-Instruct-2507，学生 Qwen3-0.6B-Base，同族共享 tokenizer（启动时 `assert_same_vocab` 强校验）
- 每步 16 道题 × 4 条 rollout = 64 条序列，40 步，`max_new_tokens=288`
- 评测：GSM8K test 200 题，avg@4（temp 0.7）+ greedy pass@1
- 结果表见 `figures/results_table.md`，训练曲线 `fig3`，成本-效果曲线 `fig4`，训练诊断 `fig5`

## 4. 结果

九个臂共享同一 SFT-init checkpoint 与同一份 rollout 预算（每臂 2560 条），总耗时 2 小时 15 分。
按 avg@4 选取最优 checkpoint：

| 臂 | avg@4 | Δ | greedy | pass@4 |
| --- | --- | --- | --- | --- |
| SFT-init（共同起点） | 0.331 | — | 0.490 | 0.660 |
| +2× off-policy SFT 数据 | 0.366 | +0.035 | 0.550 | 0.690 |
| GRPO（结果奖励 RL） | 0.404 | +0.073 | 0.540 | 0.700 |
| Reverse KL / OPD | **0.494** | **+0.163** | **0.585** | 0.790 |
| Reverse KL / OPD+ | 0.472 | +0.141 | 0.555 | 0.735 |
| Forward KL / OPD | 0.331 | 0.000 | 0.490 | 0.660 |
| Forward KL / OPD+ | 0.450 | +0.119 | 0.470 | 0.785 |
| JSD / OPD | 0.331 | 0.000 | 0.490 | 0.660 |
| JSD / OPD+ | 0.460 | +0.129 | 0.405 | **0.795** |

forward KL 与 JSD 的 OPD 臂最优点都落在 step 0，即全程未超过起点；它们的实际终点是 0.039 与 0.134。
avg@4 在 200 题上的标准误约 ±0.03，差距小于约 0.06 的一律不作结论。

**崩溃与救活完整复现。** forward KL 在 20 步内损失 88% 的相对准确率且不再恢复，
JSD 跌到 0.134；换成 $w_f(u)$ 后分别回到 0.450 与 0.460。

**梯度尺度给出了机理层面的证据，比准确率更硬。**

| 臂 | 平均梯度范数 | 平均 `adv_absmax` |
| --- | --- | --- |
| Reverse KL / OPD → OPD+ | 15.4 → 18.0（几乎不变） | 6.00 → 7.00 |
| Forward KL / OPD → OPD+ | **1941.7 → 13.8（140×）** | 2026.6 → 171.7 |
| JSD / OPD → OPD+ | **92.7 → 1.25（74×）** | 65.0 → 2.01 |

修正项对 reverse KL 是常数、不改变梯度，对其余散度不是——理论的可证伪预言在数据里精确成立。
`adv_absmax` 那两个数还提供了肉眼可读的验证：Reverse KL / OPD 恒为 6.000（clip 边界），
OPD+ 恒为 7.000（clip + 1），论文中抽象的常数 $-1$ 直接可见。

**被优化的目标本身在反向移动。** 健康臂的 per-token teacher KL 从 0.90 单调降到约 0.50，
两条崩溃臂反而升到 1.35 与 1.55——它们不是学得慢，是在远离教师。
两者的退化形态还相反：forward KL 的响应长度从 169 涨到 287（上限 288，即永不终止），
JSD 从 148 塌缩到 75。

![training diagnostics](figures/fig5_training_diagnostics.png)

**同等预算下稠密信号显著优于稀疏奖励。** reverse KL / OPD 用 1280 条 rollout 达到 0.494，
高于 GRPO 用 2560 条的 0.404，而教师前向只占单步开销的 11%（18.1 s 对 16.1 s）。
省的不是采样成本，是达到同等效果所需的步数。「多喂 off-policy 数据」则收效甚微（+0.035，落在噪声内）。

![cost efficiency](figures/fig4_cost_efficiency.png)

完整分析、局限与后续方向见 [`report/REPORT.md`](report/REPORT.md)。

## 5. 快速开始

```bash
pip install -r requirements.txt
python -m pytest tests -q               # CPU，验证 advantage 公式与 autograd 一致
python -m scripts.run_all --tier smoke  # ~8 min，端到端冒烟
python -m scripts.run_all --tier all    # 完整九臂
python -m scripts.make_plots            # 出图与结果表
```

全部脚本以模块方式从仓库根目录运行，这样 `opd` 包才能被正确导入。
Colab 用户直接打开 `notebooks/colab_opd.ipynb`。`run_all` 是可续跑的——
已有 `results/{arm}_final.json` 的臂会被跳过，适配 Colab 掉线。

## 6. 工程实现要点

手写而非调库，是因为这几个点调库时容易被藏住，而它们恰恰是 OPD 能不能训对的关键。

**采样必须是严格的 temperature=1.0，且不能有 top-k / top-p 截断。** 任何截断都会让实际采样分布偏离
$p_\theta$，policy gradient 的 on-policy 假设就不成立了，而框架默认参数往往带 `top_p=0.9`。

**师生必须共享同一套 prompt 渲染。** 这一点我最初做错了：为了让 instruct 教师留在它自然的上下文里，
我让教师走 chat template、学生走纯文本 completion。看着合理，实际引入了纯格式噪声——
学生以 `<|endoftext|>` 结束，而教师在 ChatML 上下文里几乎把全部终止概率给了 `<|im_end|>`，
于是**每条 rollout 的终止 token 都拿到一个被钉死在裁剪边界的 log-ratio**。在 reverse KL 下
advantage 就是 $\log u$，等于每一步都在系统性地教学生"不要停"，和推理质量毫无关系。
Tinker 的参考实现是把学生 rollout 的整条 sequence 直接喂给 `teacher.compute_logprobs`，
师生 token 完全一致。现在默认的 `--prompt-style chat` 与之对齐：base 学生在 SFT 阶段就学会 ChatML 格式，
`plain` 和 `split` 保留下来作为消融。

即便 prompt 一致，教师视角与学生视角的位置仍需对齐（`align_to_student()`）——
换成其它 prompt-style 时两边长度不同，同一个 response token 的绝对下标就不一样。
这是 OPD 实现里最经典的静默 bug，所以单独抽成了一个函数。

**logprob 采集要分块。** 教师是 4B、词表 151936，一次前向的完整 logits 张量就有好几个 GB。
`token_logprobs()` 只取 backbone 的 hidden state，再按位置分块过 `lm_head` 并立即 gather 目标 token，
把峰值显存压到可控范围。

**绝对不能对 advantage 做 z-score 归一化。** RL 训练里 batch 内归一化几乎是肌肉记忆，
但在这里它会把 $-f(u)$ 和 $w_f(u)$ 的差异——尤其是 reverse KL 那个常数 $-1$——直接抹平，
整个实验就失去了结论。代码里对此有显式注释和单测 `test_no_implicit_normalisation`。

**Forward KL 的 OPD+ advantage $u$ 是无界的**，必须裁剪 log-ratio。默认 clip 到 $\pm 6$
（即 $u \le 403$），并把 `clip_frac` 作为一等指标逐步记录——这是判断 forward KL 是否要炸的先行指标。
对 reward 做裁剪并非权宜之计：REOPOLD 用的 "mixture-based reward clipping to prevent over-trust of
extreme teacher signals" 就是同一类手段。

**教师监督在师生分布差距大的地方本就不可靠**，这是 strong-to-weak OPD 的已知问题而不是实现瑕疵。
FiRe-OPD 的说法是：教师 log-probability 低意味着"教师在给一种它不熟悉的推理风格打分"，
强行蒸馏会造成负迁移；它按归一化 teacher log-prob 丢掉最差 20% 的轨迹，消融显示这是全部收益里最大的一块。
综述另外归纳了 flawed prefix trap 与 local teachability collapse 两个更细的失效模式，
后者明确标注为 strong-to-weak 特有。本项目记录 `clip_frac` 与 `teacher_kl` 正是为了量化这个效应，
轨迹级过滤留作后续方向。

**梯度检查点与 KV cache 冲突。** 训练时开 gradient checkpointing、`use_cache=False`；
rollout 和评测前显式关掉并切 `eval()`，否则 transformers 会静默强制 `use_cache=False`，
解码速度掉一个数量级。

## 7. 目录结构

```
opd/
  divergences.py   f-散度 advantage 表，OPD 与 OPD+ 的唯一区别就在这里
  core.py          模型加载、分块 logprob、on-policy rollout、师生视角对齐
  data.py          GSM8K 加载、师生两套 prompt 渲染、答案抽取与判分
  evaluate.py      avg@k / greedy 评测
scripts/
  train_sft.py     学生初始化 + off-policy 对照臂
  train_pg.py      OPD / OPD+ / GRPO 共用的 policy-gradient 主循环
  run_all.py       九臂顺序编排，可续跑
  make_plots.py    五张图 + markdown 结果表
tests/             用 autograd 校验 advantage 公式的单测
figures/           复现出的五张图与结果表
report/REPORT.md   完整技术报告
notebooks/         Colab 入口
```

## 8. 许可

Apache License 2.0，与 Qwen3 模型权重的许可一致。见 [`LICENSE`](LICENSE)。

## 9. 参考

- Kevin Lu and Thinking Machines Lab. *On-Policy Distillation.* Connectionism, Oct 2025.
- Zhao, Chen, Lin, Winata, Yao, Tang. *OPD+: Rethinking the Advantage Design for On-Policy Distillation.* arXiv:2606.01039.
- Agarwal et al. *On-Policy Distillation of Language Models (GKD).* ICLR 2024.
- Li et al. *Filter, Then Reweight: Rethinking Optimization Granularity in On-Policy Distillation.* arXiv:2606.02684.
- *A Survey of On-Policy Distillation for Large Language Models.* arXiv:2604.00626.
- Busbridge et al. *Distillation Scaling Laws.* 2025（capacity gap：教师过强时 forward KL 的 mode-covering 压力会放大差距）。
