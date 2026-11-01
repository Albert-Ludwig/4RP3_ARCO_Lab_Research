# WELL-E 项目核心逻辑

> 目标：用 Gaussian Process（GP）描述未知的三维时空温度场，用采样事件的条件互信息评价候选路线，再由 Generalized Recursive Greedy（GRG）在任务时间预算内选择高层采样顺序；机器人取得新测量后更新 GP，并以滚动时域方式重新规划。

## 0. 一条公式概括整条研究链

$$
\boxed{
\mathcal D_k
\xrightarrow{\mathrm{GP\ conditioning}}
(\mu_k,\Sigma_k)
\xrightarrow{\mathrm{sampling\text{-}event\ MI}}
J(P\mid\mathcal D_k)
\xrightarrow[\,T(P)\le T_{\mathrm{rem}}\,]{\mathrm{GRG}}
P_k^*
\xrightarrow{\mathrm{Nav2/sensing}}
(\mathbf z_{k+1},y_{k+1})
\xrightarrow{\mathrm{update}}
\mathcal D_{k+1}
}
$$

其中，GP 回答“温度及其不确定性是什么”，MI 回答“一组未来采样事件能增加多少信息”，GRG 回答“在有限时间内按什么顺序采样”，Nav2 则负责“如何实际到达相邻 waypoint”。

---

## 1. 温度场与离散含噪观测

未知温度场写为

$$
f:\mathbb R^3\rightarrow\mathbb R,
\qquad
T=f(x,y,t),
$$

并定义时空输入

$$
\mathbf z=[x,y,t]^\top.
$$

机器人或历史传感器只能得到有限的离散观测：

$$
\mathcal D=\{(\mathbf z_i,y_i)\}_{i=1}^{N},
\qquad
y_i=f(\mathbf z_i)+\epsilon_i,
\qquad
\epsilon_i\sim\mathcal N(0,\sigma_n^2).
$$

- $f(\mathbf z_i)$ 是潜在的真实温度；
- $y_i$ 是含测量噪声的观测；
- $\sigma_n^2$ 是观测噪声方差。

研究问题不是只拟合已经测过的点，而是用 $\mathcal D$ 推断连续温度场，并主动选择下一批最有信息价值的采样事件。

## 2. GP 先验与 3D ARD RBF Kernel

对未知函数设置 GP 先验：

$$
f(\mathbf z)\sim
\mathcal{GP}\!\left(m(\mathbf z),k(\mathbf z,\mathbf z')\right).
$$

WELL-E 使用的三维 ARD RBF kernel 为

$$
\boxed{
k(\mathbf z,\mathbf z')=
\sigma_f^2
\exp\!\left[
-\frac12\left(
\frac{(x-x')^2}{\ell_x^2}
+\frac{(y-y')^2}{\ell_y^2}
+\frac{(t-t')^2}{\ell_t^2}
\right)
\right]
}
$$

其超参数可记为

$$
\boldsymbol\theta=
\{\sigma_f^2,\ell_x,\ell_y,\ell_t,\sigma_n^2\}.
$$

- $\sigma_f^2$ 控制潜在温度场的信号方差；
- $\ell_x,\ell_y,\ell_t$ 分别控制两个空间方向和时间方向的相关尺度；
- ARD 允许三个方向具有不同变化速率。

Kernel 的关键含义是

$$
k(\mathbf z,\mathbf z')
=\operatorname{Cov}\!\left(f(\mathbf z),f(\mathbf z')\right).
$$

因此，它不仅决定某个测量如何影响邻近位置的预测均值，也决定测量后其他位置的不确定性下降多少。它还是后续 MI 的基础：两个高度相关的采样点通常存在信息冗余。

> **10.18 replay 的边界：** 回放期间使用训练阶段得到的 kernel 超参数，并将其保持固定。新观测只加入 GP conditioning dataset；回放并没有在每次采样后重新优化 $\boldsymbol\theta$。

## 3. GP 后验均值与协方差

令训练输入为 $X=[\mathbf z_1,\ldots,\mathbf z_N]^\top$，观测向量为 $\mathbf y$，查询集合为 $Z_*$。定义

$$
C=K_{XX}+\sigma_n^2I.
$$

GP conditioning 给出潜在温度场的后验均值：

$$
\boxed{
\boldsymbol\mu_*
=\mathbf m_*+K_{*X}C^{-1}(\mathbf y-\mathbf m_X)
}
$$

以及后验协方差：

$$
\boxed{
\Sigma_*
=K_{**}-K_{*X}C^{-1}K_{X*}
}
$$

即

$$
f(Z_*)\mid\mathcal D
\sim\mathcal N(\boldsymbol\mu_*,\Sigma_*).
$$

这里 $\operatorname{diag}(\Sigma_*)$ 是**潜在温度场**的 posterior variance；未来含噪观测的 predictive covariance 则为 $\Sigma_*+\sigma_n^2I$。固定时间 $t$ 后，$\mu(x,y,t)$ 形成预测温度图，$\sigma^2(x,y,t)$ 形成不确定性图。

实际计算中应通过 Cholesky 分解或线性方程求解实现上述表达式，而不是显式构造 $C^{-1}$。

## 4. 10.18 回放中的采样事件 Mutual Information

候选高层路线写为

$$
P=[v_0,v_1,\ldots,v_M].
$$

路线不只决定“去哪些空间节点”，还通过移动、等待和驻留时间决定“何时测量”。因此对应的采样事件集合是

$$
S(P)=\{\mathbf z_i^P\}_{i=1}^{M},
\qquad
\mathbf z_i^P=(x_{v_i},y_{v_i},t_i^P).
$$

给定当前数据 $\mathcal D$，先在这些事件上计算潜在温度的 GP posterior covariance：

$$
\Sigma_{SS\mid\mathcal D}
=K_{SS}-K_{SX}
(K_{XX}+\sigma_n^2I)^{-1}K_{XS}.
$$

10.18 Intel Lab GP-GRG replay 使用的路线奖励是采样事件的联合条件互信息：

$$
\boxed{
J(P\mid\mathcal D)
=I\!\left(
\mathbf y_{S(P)};
\mathbf f_{S(P)}mid\mathcal D
\right)
=\frac12\log\det\!\left(
I+\frac{1}{\sigma_n^2}
\Sigma_{S(P)S(P)\mid\mathcal D}
\right)
}
$$

它同时考虑：

1. 每个候选事件自身的 posterior uncertainty；
2. 不同事件之间由 kernel 给出的相关性与信息冗余；
3. 路线引起的实际采样时间，因此同一空间节点在不同时间可以具有不同价值。

若 $\lambda_j$ 是 $\Sigma_{SS\mid\mathcal D}$ 的特征值，则

$$
J(P\mid\mathcal D)
=\frac12\sum_j
\log\!\left(1+\frac{\lambda_j}{\sigma_n^2}\right).
$$

这也解释了为什么不能简单地把各点的 marginal variance 相加：联合 $\log\det$ 会对相关、重复的信息进行折减。

> **重要限制：MI 不等价于 RMSE。** MI 是由模型协方差定义的信息论目标，不直接使用未来真实温度；RMSE 是预测均值对 held-out ground truth 的误差。因此，MI 增加或 posterior variance 降低，并不保证有限测试集上的 RMSE 同步降低。

## 5. GRG：时间预算内的高层路线优化

设候选节点集合为 $V$，旅行时间矩阵为

$$
TT=[tt_{ij}],
\qquad
tt_{ij}=\text{从节点 }v_i\text{ 到 }v_j\text{ 的估计旅行时间}.
$$

对路线 $P=[v_0,ldots,v_M]$，任务时间可写为

$$
T_{\mathrm{mission}}(P)
=T_{\mathrm{travel}}(P)
+T_{\mathrm{waiting}}(P)
+T_{\mathrm{dwell}}(P)
+T_{\mathrm{computation}}(P),
$$

其中

$$
T_{\mathrm{travel}}(P)
=\sum_{i=0}^{M-1}tt_{v_i v_{i+1}}.
$$

规划问题是

$$
\boxed{
P^*\approx
\arg\max_{P\in\mathcal P}
J(P\mid\mathcal D)
\quad
\text{s.t.}\quad
T_{\mathrm{mission}}(P)\le T_{\max}
}
$$

若任务要求返回终点或基地 $g$，可行路线还必须满足 $v_M=g$。GRG 用候选中间节点 $v$ 和时间预算拆分递归构造路线：

$$
P_{s\rightarrow g}
=P_{s\rightarrow v}\oplus P_{v\rightarrow g},
\qquad
B_1+B_2\le B-tt_{sv}-tt_{vg}
$$

（具体预算扣除方式还需包含该实现定义的等待、驻留等成本。）GRG 是近似组合优化方法；它输出的是**有序测量 waypoint/事件**，而不是连续、无碰撞的机器人轨迹。

在 10.18 离线 replay 中：

- $TT$ 来自仿真的栅格移动/时间估计，而不是运行中的 Nav2；
- 使用模拟任务时钟；
- online computation 在模拟任务时钟中被设为零，因此 replay 的 870.26 s 不是现实系统端到端墙钟耗时的证明。

## 6. 高层 GRG 与低层 Nav2 的职责边界

两层规划解决的问题不同：

| 层级      | 输入                                    | 输出                               | 主要约束                     |
| --------- | --------------------------------------- | ---------------------------------- | ---------------------------- |
| 高层 GRG  | GP covariance、候选事件、$TT$、剩余预算 | waypoint/采样事件顺序 $P^*$        | 信息收益、总任务时间、起终点 |
| 低层 Nav2 | 当前位姿、下一个 waypoint、地图与障碍物 | 可执行路径与控制行为 $\pi_{i,i+1}$ | 碰撞、运动学、局部避障、控制 |

相邻高层节点之间，Nav2 需要产生

$$
\pi_{i,i+1}(s)=
\begin{bmatrix}x(s)&y(s)\end{bmatrix}^{\top},
$$

并实际执行该路径。理想的系统还应从 Nav2 的全局路径长度、速度模型或实测通行时间生成/更新 $tt_{ij}$，使高层预算与真实可达性一致。

**独立的 Nav2 静态障碍导航测试已完成；但它不能被表述为 GP-GRG-Nav2 已经闭环集成。**

## 7. 传感、GP 更新与滚动时域重规划

到达下一个采样事件并获得有效测量后，更新数据集：

$$
\boxed{
\mathcal D_{k+1}
=\mathcal D_k\cup
\{(\mathbf z_{k+1},y_{k+1})\}
}
$$

固定 kernel 超参数，只重新 conditioning：

$$
(\mu_k,\Sigma_k)
\longrightarrow
(\mu_{k+1},\Sigma_{k+1}).
$$

然后从当前机器人位置、当前时刻和剩余预算重新求解：

$$
\boxed{
P_{k+1}^*\approx
\arg\max_{P\in\mathcal P_{k+1}}
J(P\mid\mathcal D_{k+1})
\quad
\text{s.t.}\quad
T(P)\le T_{\mathrm{remaining}}
}
$$

这就是 receding-horizon informative path planning：只执行当前计划的下一段，在获得真实新信息后再更新模型与路线，而不是在任务开始时一次性锁定整条路线。

完整在线循环应为：

$$
\text{定位}
\rightarrow
\text{GRG 选下一 waypoint}
\rightarrow
\text{Nav2 执行}
\rightarrow
\text{时间戳对齐并采样}
\rightarrow
\text{质量检查}
\rightarrow
\text{GP update}
\rightarrow
\text{replan}.
$$

## 8. 评价指标及其解释

### 8.1 GP 预测质量

对 held-out 集合 $\{(\mathbf z_j,y_j)\}_{j=1}^{N_*}$：

$$
\operatorname{RMSE}
=\sqrt{
\frac1{N_*}
\sum_{j=1}^{N_*}
(\mu(\mathbf z_j)-y_j)^2
},
$$

$$
\operatorname{MAE}
=\frac1{N_*}
\sum_{j=1}^{N_*}
\left|\mu(\mathbf z_j)-y_j\right|.
$$

若报告 95% predictive interval coverage，必须说明区间针对潜在 $f$ 还是含噪观测 $y$；二者方差分别使用 $\sigma_f^2(\mathbf z)$ 与 $\sigma_f^2(\mathbf z)+\sigma_n^2$。

### 8.2 信息与规划质量

- 路线联合 MI：$J(P\mid\mathcal D)$（nats）；
- latent posterior variance 的前后变化；
- 有效测量数量、重复/无效测量数量；
- GRG planning time、递归深度、可行返回；
- 任务时间预算的使用情况。

### 8.3 导航与系统质量

- waypoint/navigation success rate；
- 实际 travel time 与 $tt_{ij}$ 估计误差；
- 碰撞/恢复次数；
- 是否在预算内安全返回；
- 端到端 sensing-to-replanning latency。

## 9. 10.18 Intel Lab 结果：怎样准确表述

以下结果来自已验证的实数据报告；其中**测量值来自 Intel Lab 真实记录，但机器人的移动、任务时钟和路线执行是 replay/simulation**。

| 项目                 |                                                  已验证结果 | 正确解释                                          |
| -------------------- | ----------------------------------------------------------: | ------------------------------------------------- |
| GP 训练/测试数据     |         2,000 条前三日训练记录；90,690 条第四日有效测试记录 | 原始记录及保存的 GP 输入/目标已做 provenance 核验 |
| 第四日整体预测       | RMSE $4.93747\,^{\circ}\mathrm C$；95% 区间 coverage 91.05% | 评价固定 GP 在整日 held-out 数据上的预测与校准    |
| baseline replay      |        采集 7 条真实 Intel 记录；模拟任务时钟 870.26 s 返回 | “真实”指被采样的温度记录；导航与时钟仍是回放模拟  |
| replay held-out RMSE |              $0.48958\rightarrow0.54691\,^{\circ}\mathrm C$ | 该特定 replay 评估集上 RMSE 略有变差              |
| latent variance      |            $1.11523\rightarrow0.43012\,^{\circ}\mathrm C^2$ | 模型不确定性明显下降，但不能据此宣称预测误差改善  |
| GRG 深度测试         |                            depth 1/2/3 完成；depth 5 未运行 | 不应补写或推断 depth 5 性能                       |
| MI 核验              |                  独立实现差异约 $1.4236\times10^{-11}$ nats | 支持 MI 数值实现的一致性，不等于系统级性能证明    |

最关键的实验结论是：

$$
\boxed{
\text{posterior variance 下降}
\centernot\Rightarrow
\text{held-out RMSE 下降}
}
$$

这不是矛盾。可能原因包括固定 kernel 与真实场不完全匹配、MI 只优化模型内的信息量、采样点与有限评估集的空间时间分布不同，以及模型校准误差。因此报告应分别陈述“模型认为不确定性下降”和“外部 ground truth 上的预测误差变化”。

## 10. 已验证、设计边界与尚未实现

| 状态                 | 内容                                                                                                                                                                                                     |
| -------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **已验证**           | Intel Lab 数据 provenance；固定超参数 GP conditioning；3D ARD RBF covariance 链；sampling-event joint MI 数值；GP-GRG 离线 replay；递归深度 1/2/3；真实记录上的预测/方差指标；独立 Nav2 静态障碍导航测试 |
| **回放中的设计选择** | kernel 超参数固定；仿真栅格旅行时间矩阵；模拟任务时钟；online computation 在任务时钟中计为零                                                                                                             |
| **尚未集成/实现**    | 真实机器人上的 GP-GRG-Nav2 闭环；Nav2 在线生成或校准 $TT$；机器人传感器实时采样、时间同步与质量检查；测量后自动 GP update 并触发 GRG；真实障碍与延迟下的端到端安全返回验证                               |

因此，目前最准确的系统表述是：

> WELL-E 已验证 GP、sampling-event MI、GRG replay 和独立 Nav2 导航这些关键组成部分；完整的在线 GP-GRG-Nav2 receding-horizon 闭环仍是待完成的系统集成工作。

---

## 11. 最终逻辑

$$
\boxed{
\underbrace{\mathcal D}_{\text{真实离散测量}}
\xrightarrow{\mathrm{GP}}
\underbrace{(\mu,\Sigma)}_{\text{预测与相关不确定性}}
\xrightarrow{\mathrm{MI}}
\underbrace{J(P\mid\mathcal D)}_{\text{联合采样价值}}
\xrightarrow{\mathrm{GRG}}
\underbrace{P^*}_{\text{预算内高层事件顺序}}
\xrightarrow{\mathrm{Nav2}}
\underbrace{(\mathbf z_{\mathrm{new}},y_{\mathrm{new}})}_{\text{执行与传感}}
\xrightarrow{\mathrm{update/replan}}
\mathcal D_{\mathrm{new}}
}
$$

Kernel 是这条链的核心连接件：它决定 GP 如何传播测量影响，GP posterior covariance 决定 MI 如何衡量联合信息价值，而 MI 又决定 GRG 在时间预算内偏好哪些采样事件。系统评价则必须把模型内的信息收益、外部 ground-truth 预测误差、规划代价和真实导航表现分开报告。
