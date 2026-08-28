# 面试问答准备

按"必被问到"的顺序排。每条都给了能在白板上写出来的程度，不要只背结论。

---

## 一、方法本身

**Q: 一句话讲清 on-policy distillation 和 SFT、RL 的关系。**

三者的区别只在两个维度：轨迹从哪来、监督信号有多密。SFT 是 off-policy + dense（教师轨迹、逐 token 监督），
outcome RL 是 on-policy + sparse（自己 rollout、只有最终对错），OPD 是 on-policy + dense
（自己 rollout、教师逐 token 打分）。SFT 的问题是 exposure bias——学生训练时永远待在教师的状态分布上，
推理时一旦偏离就误差累积；RL 的问题是信用分配，几百个 token 只有一个标量奖励。OPD 同时避开这两个。

**Q: 为什么用 reverse KL 而不是 forward KL？**

先答教科书答案：reverse KL 是 mode-seeking 的，$\mathbb{E}_{p}[\log p/q]$ 在 $q$ 小而 $p$ 大的地方惩罚极重，
逼学生把概率质量收缩到教师的某个高概率模式上，输出更"确定"；forward KL 是 mass-covering，
会让学生去覆盖教师的所有模式，小模型容量不够时就摊成一个糊掉的平均分布。
再答工程答案：reverse KL 的期望是在学生分布下取的，正好能用学生自己的 rollout 无偏估计，不需要重要性采样。

然后给出这个项目的真正答案（这是加分项）：**以上都不是全部**。OPD+ 证明，
业界观察到的"forward KL 训不了"很大程度上是实现假象——stop-gradient 让 forward KL 拿到了
$-u\ln u$ 这个错误的梯度系数，修正成 $u$ 之后 forward KL 和 JSD 都能正常训练，JSD 甚至反超。

**Q: stop-gradient 那一项到底错在哪？为什么 reverse KL 侥幸没错？**

OPD 的目标是 $J^f(\theta) = \mathbb{E}_{y \sim p_\theta}\left[\sum_n f(u_{\theta,n})\right]$，
其中 $u = q/p_\theta$。求梯度时有两项：轨迹分布对 $\theta$ 的依赖（score function 项）、
以及 reward $f(u_\theta)$ 自身对 $\theta$ 的依赖。业界只保留前者，把后者 stop-grad 掉了。
把第二项算出来，等价于把 advantage 从 $-f(u)$ 改成 $-f(u) + u f'(u)$。

对 reverse KL，$f(u) = -\ln u$，$f'(u) = -1/u$，所以 $u f'(u) = -1$，是常数。
常数 advantage 在 score identity $\mathbb{E}_{p_\theta}[\nabla_\theta \log p_\theta] = 0$ 下期望为零，
只相当于加了个 baseline，不改变梯度期望。这就是 reverse KL 侥幸正确的原因。
对一般的 $f$，$u f'(u)$ 不是常数，stop-grad 就是实打实的有偏。

**Q: 既然 reverse KL 的修正项理论上只是 baseline，为什么实测还是有提升？**

baseline 不改变梯度的期望，但改变方差。$\ln u$ 在采样点上通常是负的（学生概率高于教师的 token 更容易被采到），
减去常数 1 相当于改变了正负 advantage 的分界点，会改变有效的梯度信噪比。
论文和我这边都观察到 reverse KL+ 略优于 reverse KL，这是方差层面的收益而不是偏差层面的。

---

## 二、工程实现

**Q: 讲讲你踩了哪些坑。**

挑三个讲，每个都要说清"错了会怎样"：

1. **采样温度**。on-policy 要求采样分布严格等于 $p_\theta$，所以必须 temperature=1.0 且关掉 top-k/top-p。
   框架默认带 top_p=0.9，一旦保留，policy gradient 的估计就有偏了，而且这个偏差是静默的——训练看着照常收敛。
2. **师生视角对齐**。教师走 chat template、学生走纯 completion 格式，两边 prompt 长度不同，
   同一个 response token 的绝对下标就不同。teacher logprob 搬到学生布局时错一位，KL 就变成了在比较相邻 token，
   数值不会崩，指标只是"效果一般"，极难查。我把这段单独抽成 `align_to_student()` 并配了掩码断言。
3. **advantage 不能归一化**。batch 内 z-score 是 RL 的肌肉记忆，但在这里会把 $-f(u)$ 和 $w_f(u)$ 的差异
   直接抹平，实验就白做了。我写了单测 `test_no_implicit_normalisation` 把这条锁死。

**Q: 显存怎么控的？**

学生 0.6B 全参训练，bf16 权重 1.2GB + 梯度 + AdamW 状态约 11GB；教师 4B 纯推理 8GB。
真正的大头是 logits：词表 151936，教师一次前向 batch 8 × 长度 600 的完整 logits 就是 1.4GB（bf16），
fp32 做 log_softmax 会翻倍。所以 `token_logprobs()` 不走 `model(...)` 的默认路径，
而是取 backbone 的 hidden state，按位置分块过 `lm_head`，gather 完目标 token 立刻释放。
再叠加 micro-batch 和 gradient checkpointing，峰值稳定在 A100 40GB 以内。

**Q: 为什么不用 TRL / verl / tinker？**

演示项目里，框架把关键决策藏起来是负资产。上面三个坑全都在框架的默认参数和内部实现里，
自己写 400 行反而可控。生产环境我会用 verl，因为需要的是多机 rollout 调度、vLLM 权重同步、
序列并行这些我不该重复造的东西——但那和"证明我懂这个算法"是两件事。

---

## 三、业务迁移（这部分决定评级）

**Q: 这套东西在大厂真实业务里怎么落地？**

三个直接对应的场景：

1. **能力下沉 / 成本压降**。线上用 4B 或 8B 抗流量，用旗舰模型做教师做 OPD，
   在特定业务分布（客服、检索问答、代码补全）上把小模型拉到接近大模型的水平。
   相比"大模型生成数据 + 小模型 SFT"，OPD 省掉了数据生成这一大块离线算力，
   而且监督信号是在学生自己的错误状态上给的，同样步数下收益更高。
2. **持续学习 / 抗遗忘**。TML 博客里的第二个实验：企业把内部文档 mid-train 进模型后，
   指令跟随能力会退化；用**注入前的自己**当教师做一轮 OPD，能在不丢新知识的前提下把通用能力找回来。
   这在"每个业务线都要一个定制模型"的场景里是刚需。
3. **投机解码草稿模型**。Draft-OPD 那条线：草稿模型用 SFT 训会很快饱和，
   因为它评测时面对的是自己 propose 出来的状态，而训练时看的是 target 模型的轨迹——
   又是一个 exposure bias。改成 OPD 之后接受长度显著提升，直接换算成推理成本。

**Q: 你这个项目的结论能外推到大模型吗？**

诚实回答边界。我这里是 4B 教 0.6B、GSM8K、40 步，规模比论文小两个量级，
所以我能给的证据是**定性的机理验证**——forward KL 在 stop-grad 下的失效模式、
以及修正后能否恢复，这类现象由 advantage 函数的形状决定，和规模关系不大；
而"JSD+ 能不能反超 reverse KL"这种定量排序，我的实验规模不足以支撑，
论文在 8B + AIME 上给出的结论我不会当成自己的结论去讲。

**Q: 如果给你更多资源，下一步做什么？**

三个方向，按性价比排：把 logit 层面的 top-k 蒸馏和当前的单 token likelihood 做对比
（信息量更大但要改推理侧基建）；引入 FiRe-OPD 的轨迹级过滤——教师整体 likelihood 低的 rollout
说明师生分布差距太大、监督不可靠，应该直接丢掉而不是降权；
以及把 OPD 从单轮生成扩到多轮 agent 轨迹，这是综述里点名的开放问题，也是业务上最缺的。

---

## 四、可能被追问的细节

- **γ（discount factor）为什么取 0？** TML 报告 γ>0 尽管数学上更严格但实测无收益，
  所以取 0，每个 token 只优化自身的即时 reward。这也让 advantage 变成纯逐 token 的，实现和调试都简单。
- **importance sampling ratio 在纯 on-policy 下不是恒等于 1 吗？** 数值上是，但保留
  $\rho_\theta = p_\theta / p_{\theta_0}$ 的形式让梯度正确流动（$\nabla \rho = \rho \nabla \log p_\theta$），
  而且一旦要做多个 inner epoch 或异步 rollout，这个形式无需改动就能直接支持。
- **为什么 OPD 说比 RL 省 10 倍算力？** 同样一条 rollout，RL 只榨出 1 bit（对/错），
  OPD 榨出几百个 token 的 dense 信号。省的不是采样算力，是达到同等效果所需的采样次数。
- **教师推理的开销怎么算？** 教师只做一次 forward 算 logprob，不做自回归解码，
  这比"教师生成 SFT 数据"便宜得多，也是 OPD 成本优势的来源之一。
