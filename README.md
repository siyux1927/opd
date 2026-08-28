# OPD / OPD+：单卡 A100 上的 On-Policy Distillation 复现与 advantage 修正验证

把 **Qwen3-4B-Instruct-2507** 的数学推理能力，用 on-policy distillation 蒸馏进 **Qwen3-0.6B-Base**，
并在同一套框架里验证 **OPD+**（[arXiv:2606.01039](https://arxiv.org/abs/2606.01039)）提出的
advantage 修正：现行 OPD 实现对 reward 做 stop-gradient 会带来有偏梯度，正确的逐 token advantage 应当是

$$w_f(u) = -f(u) + u\,f'(u), \qquad u = \frac{q_{\text{teacher}}(y_n \mid y_{<n}, x)}{p_{\theta}(y_n \mid y_{<n}, x)}$$

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

`figures/fig1_weight_functions.png` 把三种散度的 $-f(u)$ 与 $w_f(u)$ 画在同一张对数横轴上。

Forward KL 的崩溃原因一眼可见：$-u\ln u$ 在 $u \to 0$ 时趋于 0。也就是说，
**教师很想出、但学生几乎不会出的那些 token（$u \gg 1$ 的反面，即学生在这些位置概率极低）恰恰拿不到梯度**，
而 $-u\ln u$ 在 $u > 1$ 区间还是负的——等于在惩罚学生去靠近教师。修正成 $u$ 之后，
advantage 变成单调正的无界推力，方向才是对的。

`figures/fig2_log_ratio_distribution.png` 叠加了训练中实际采样 token 的 $\log u$ 分布，
说明这些区间不是理论边角，而是真实高频落点。

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
- 结果表见 `figures/results_table.md`，训练曲线 `fig3`，成本-效果曲线 `fig4`

## 4. 快速开始

```bash
pip install -r requirements.txt
python -m pytest tests -q          # CPU，验证 advantage 公式与 autograd 一致
python run_all.py --tier smoke     # ~8 min，端到端冒烟
python run_all.py --tier all       # 完整九臂
python make_plots.py               # 出图与结果表
```

Colab 用户直接打开 `notebooks/colab_opd.ipynb`。`run_all.py` 是可续跑的——
已有 `results/{arm}_final.json` 的臂会被跳过，适配 Colab 掉线。

## 5. 工程实现要点

手写而非调库，是因为这几个点调库时容易被藏住，而它们恰恰是 OPD 能不能训对的关键。

**采样必须是严格的 temperature=1.0，且不能有 top-k / top-p 截断。** 任何截断都会让实际采样分布偏离
$p_\theta$，policy gradient 的 on-policy 假设就不成立了，而框架默认参数往往带 `top_p=0.9`。

**师生 prompt 不同、response token 必须相同。** 教师是 instruct 模型，走 chat template；
学生是 base 模型，走纯文本 completion 格式。两者对同一串 response token 分别打分——
这要求 `align_to_student()` 把教师视角下的 logprob 正确搬到学生视角的位置上，
因为两边 prompt 长度不同，同一个 response token 的绝对下标不一样。这是 OPD 实现里最经典的静默 bug。

**logprob 采集要分块。** 教师是 4B、词表 151936，一次前向的完整 logits 张量就有好几个 GB。
`token_logprobs()` 只取 backbone 的 hidden state，再按位置分块过 `lm_head` 并立即 gather 目标 token，
把峰值显存压到可控范围。

**绝对不能对 advantage 做 z-score 归一化。** RL 训练里 batch 内归一化几乎是肌肉记忆，
但在这里它会把 $-f(u)$ 和 $w_f(u)$ 的差异——尤其是 reverse KL 那个常数 $-1$——直接抹平，
整个实验就失去了结论。代码里对此有显式注释和单测 `test_no_implicit_normalisation`。

**Forward KL 的 OPD+ advantage $u$ 是无界的**，必须裁剪 log-ratio。默认 clip 到 $\pm 6$
（即 $u \le 403$），并把 `clip_frac` 作为一等指标逐步记录——这是判断 forward KL 是否要炸的先行指标。

**梯度检查点与 KV cache 冲突。** 训练时开 gradient checkpointing、`use_cache=False`；
rollout 和评测前显式关掉并切 `eval()`，否则 transformers 会静默强制 `use_cache=False`，
解码速度掉一个数量级。

## 6. 目录结构

```
opd/
  divergences.py   f-散度 advantage 表，OPD 与 OPD+ 的唯一区别就在这里
  core.py          模型加载、分块 logprob、on-policy rollout、师生视角对齐
  data.py          GSM8K 加载、师生两套 prompt 渲染、答案抽取与判分
  evaluate.py      avg@k / greedy 评测
train_sft.py       学生初始化 + off-policy 对照臂
train_pg.py        OPD / OPD+ / GRPO 共用的 policy-gradient 主循环
run_all.py         九臂顺序编排，可续跑
make_plots.py      四张图 + markdown 结果表
tests/             用 autograd 校验 advantage 公式的单测
```

## 7. 参考

- Kevin Lu and Thinking Machines Lab. *On-Policy Distillation.* Connectionism, Oct 2025.
- Zhao, Chen, Lin, Winata, Yao, Tang. *OPD+: Rethinking the Advantage Design for On-Policy Distillation.* arXiv:2606.01039.
- Agarwal et al. *On-Policy Distillation of Language Models (GKD).* ICLR 2024.
- *A Survey of On-Policy Distillation for Large Language Models.* arXiv:2604.00626.
