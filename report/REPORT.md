# 技术报告：单卡 A100 上的 OPD 复现与 advantage 修正验证

> 跑完 `run_all.py --tier all` 与 `make_plots.py` 后，把 `figures/results_table.md`
> 的内容和四张图贴进对应位置，并按实际观测改写"实验结论"一节。带 `TODO` 的地方是待填的。

## 摘要

在单张 A100 40GB 上，用 Qwen3-4B-Instruct-2507 作教师、Qwen3-0.6B-Base 作学生，
在 GSM8K 上从零实现 on-policy distillation，并在同一框架内对比了
stop-gradient advantage $-f(u)$ 与 OPD+ 的梯度保真 advantage $w_f(u) = -f(u) + u f'(u)$，
覆盖 forward KL / reverse KL / JSD 三种 f-散度，另设 off-policy SFT 与 GRPO 两条对照。

主要发现：TODO（跑完后填 2–3 条）

## 1. 背景与动机

见 README 第 1 节。核心问题：业界"只有 reverse KL 能用"的共识，
究竟是散度本身的性质，还是 stop-gradient 实现约定造成的假象。

## 2. 方法

### 2.1 目标函数

沿学生轨迹的 f-散度目标：

$$J^f(\theta) = \mathbb{E}_{y \sim p_\theta(\cdot \mid x)} \left[ \sum_{n=1}^{L_y} f(u_{\theta,n}) \right],
\qquad u_{\theta,n} = \frac{q(y_n \mid y_{<n}, x)}{p_\theta(y_n \mid y_{<n}, x)}$$

对 $\theta$ 求导有两项：轨迹分布的 score function 项，与 reward 自身对 $\theta$ 的依赖项。
业界实现只保留前者。完整梯度等价于把逐 token advantage 从 $-f(u)$ 换成 $w_f(u) = -f(u) + u f'(u)$。

### 2.2 三种散度的 advantage

| 散度 | $f(u)$ | OPD: $-f(u)$ | OPD+: $w_f(u)$ |
| --- | --- | --- | --- |
| Forward KL | $u\ln u$ | $-u\ln u$ | $u$ |
| Reverse KL | $-\ln u$ | $\ln u$ | $\ln u - 1$ |
| JSD | $\frac12[u\ln u-(1+u)\ln\frac{1+u}{2}]$ | $-\frac12[u\ln u-(1+u)\ln\frac{1+u}{2}]$ | $\frac12\ln\frac{1+u}{2}$ |

reverse KL 的修正项是常数 $-1$，在 score identity 下作为 baseline 消失，
这解释了它为何是唯一在 stop-gradient 约定下仍然正确的散度。
公式实现由 `tests/test_divergences.py` 用 autograd 逐点校验。

### 2.3 损失

$\gamma = 0$，即每个 token 只优化自身即时 reward。损失采用 importance sampling 形式，
与 Tinker 的 `importance_sampling` loss 一致：

$$\mathcal{L} = -\frac{1}{\sum_n m_n} \sum_n m_n \cdot \frac{p_\theta(y_n \mid \cdot)}{p_{\theta_0}(y_n \mid \cdot)} \cdot A_n$$

不做任何 advantage 归一化。

## 3. 实验设置

| 项 | 取值 |
| --- | --- |
| 教师 / 学生 | Qwen3-4B-Instruct-2507 / Qwen3-0.6B-Base |
| 任务 | GSM8K |
| 学生初始化 | 2000 条人工 CoT SFT，1 epoch，lr 1e-5 |
| on-policy 步数 | 40 |
| 每步 rollout | 16 prompt × 4 sample = 64 序列 |
| 采样 | temperature 1.0，无 top-k / top-p |
| max_new_tokens | 288 |
| 优化器 | AdamW，lr 1e-5，3 步 warmup，grad clip 1.0 |
| log-ratio clip | ±6 |
| 评测 | GSM8K test 200 题，avg@4 (temp 0.7) + greedy |

硬件：单卡 A100 40GB，峰值显存 TODO GB，全部九臂总耗时 TODO 分钟。

## 4. 结果

### 4.1 主表

TODO：粘贴 `figures/results_table.md`

### 4.2 崩溃机理

![weight functions](../figures/fig1_weight_functions.png)

![log ratio distribution](../figures/fig2_log_ratio_distribution.png)

TODO：结合实际 $\log u$ 分布说明落点区间。

### 4.3 训练动态

![training curves](../figures/fig3_training_curves.png)

TODO：forward KL 的 OPD 臂是否掉到 SFT-init 以下；OPD+ 臂的 `clip_frac` 走势。

### 4.4 成本-效果

![cost efficiency](../figures/fig4_cost_efficiency.png)

TODO：OPD 达到 GRPO 最终精度所需的 A100 分钟数之比。

## 5. 结论与局限

TODO。局限务必写清：规模比论文小两个量级，40 步、单一 seed、单一任务，
因此定量排序（尤其 JSD+ 是否优于 reverse KL）不构成结论，
本项目支撑的是 advantage 形状导致的定性失效/恢复机理。

## 6. 后续方向

- logit top-k 蒸馏 vs 单 token likelihood 的信息量与成本权衡
- FiRe-OPD 式的轨迹级过滤：教师整体 likelihood 过低的 rollout 直接丢弃
- 从单轮生成扩展到多轮 agent 轨迹蒸馏
