# WELL-E 项目核心逻辑

## 0. 算法的输入与输出

目标是在给定起点、终点和时间预算的条件下，选择一条路线，使沿途预计取得的温度测量对时空场最有价值。算法的输入包括初始观测、时空 GP 模型、可行路径及其时间参数化、采样时钟，以及选定的信息目标。输出包括预算内的路线与时间安排，以及加入实际测量后的温度场后验。

整个计算链可以写成：

$$
\text{已有观测}
\longrightarrow \text{GP 后验}
\longrightarrow \text{候选路线的时空采样事件}
\longrightarrow \text{信息收益}
\longrightarrow \text{GRG 路线搜索}.
$$

执行时，实际观测进入下一次 GP 更新。路线评分所需的是候选测量的时空输入与噪声方差。实际测得的温度值在更新后验均值时才进入计算。

## 1. 温度场与离散观测

设 $\mathcal W\subseteq\mathbb R^2$ 为空间区域，$\mathcal T\subseteq\mathbb R$ 为时间区间，重建域为 $\Omega=\mathcal W\times\mathcal T$。用 $T(\mathbf z)$ 表示未知的潜在温度，输入为 $\mathbf z=(x,y,t)^\top\in\Omega$。沿轨迹运动时，传感器在离散时刻取样。若第 $i$ 次测量发生在 $t_i$，轨迹在该时刻的位置为 $(x_i,y_i)$，则模型为

$$
\mathbf z_i=(x_i,y_i,t_i)^\top,\qquad
\widetilde T_i=T(\mathbf z_i)+\epsilon_i,\qquad
\epsilon_i\overset{\mathrm{iid}}{\sim}\mathcal N(0,\sigma_n^2),
\quad \sigma_n^2>0.
$$

假定测量噪声与潜在场独立。因此，观测误差之间独立，潜在温度之间仍可通过 GP kernel 相关。两次独立测量可以拥有相同的输入，此时它们共享该输入处的潜在温度，各自带有独立噪声。以观测身份区分记录，使用 $X=(\mathbf z_i)_{i=1}^N$ 和 $\widetilde{\mathbf T}_X=(\widetilde T_i)_{i=1}^N$ 表示已观测的数据 $\mathcal D$。

## 2. GP 先验、核函数与超参数学习

### 2.1 GP 先验的有限维含义

GP 先验 $T\sim\mathcal{GP}(m,k)$ 的含义是，对任意有限输入列表 $U=(\mathbf u_i)_{i=1}^p$，潜在温度向量 $\mathbf T_U=(T(\mathbf u_i))_{i=1}^p$ 服从联合高斯分布：

$$
\mathbf T_U\sim\mathcal N(\mathbf m_U,K_{UU}),\qquad
[\mathbf m_U]_i=m(\mathbf u_i),\qquad
[K_{UV}]_{ij}=k(\mathbf u_i,\mathbf v_j).
$$

对任意输入 $\mathbf z,\mathbf z'$，$m(\mathbf z)=\mathbb E[T(\mathbf z)]$，$k(\mathbf z,\mathbf z')=\operatorname{Cov}(T(\mathbf z),T(\mathbf z'))$。前者给出条件化前的温度基线，可以取常数 $m_0$。后者给出潜在温度如何共同变化，也确定任意两个输入列表 $U,V$ 之间的交叉协方差矩阵 $K_{UV}$。

### 2.2 三维 ARD RBF kernel

对两个空间坐标和一个时间坐标分别设置长度尺度：

$$
k(\mathbf z,\mathbf z')=
\sigma_f^2\exp\!\left[-\frac12
\left(
\frac{(x-x')^2}{\ell_x^2}+
\frac{(y-y')^2}{\ell_y^2}+
\frac{(t-t')^2}{\ell_t^2}
\right)\right].
$$

$\ell_x,\ell_y,\ell_t>0$ 分别与对应坐标具有相同单位，故指数中的三项均无量纲。若两点只在某一维相差该维的一个长度尺度，先验相关系数为 $e^{-1/2}$。长度尺度越大，固定间隔下相关性越强。$\sigma_f^2>0$ 等于 $k(\mathbf z,\mathbf z)$，是潜在温度的先验方差。$\sigma_n^2$ 来自观测模型，单次含噪观测的先验方差为 $\sigma_f^2+\sigma_n^2$。训练代码沿用 GP-Planning 的空间 ARD RBF 与时间 RBF 相乘的写法。两个指数相乘等于指数内的空间项和时间项相加，正好得到上式。

### 2.3 从初始观测估计超参数

仓库采用常数均值 $m(\mathbf z)=m_0$，以训练集的各输入维度均值和标准差标准化坐标，并对观测温度作同样的变换。本节暂以 $X$ 表示拟合所用的 $N$ 个训练输入。设 $\theta=(m_0,\sigma_f^2,\ell_x,\ell_y,\ell_t,\sigma_n^2)$。根据独立高斯噪声模型，积分消去 $\mathbf T_X$ 与计算两个独立高斯向量的和等价，得到 $\widetilde{\mathbf T}_X\mid X,\theta\sim\mathcal N(\mathbf m_X,C_\theta)$，其中 $C_\theta=K_{XX}+\sigma_n^2 I_N$。令 $\mathbf r=\widetilde{\mathbf T}_X-\mathbf m_X$ 为观测相对先验均值的残差。将高斯密度取自然对数，得到

$$
\log p(\widetilde{\mathbf T}_X\mid X,\theta)=
-\tfrac12\mathbf r^\top C_\theta^{-1}\mathbf r
-\tfrac12\log\det C_\theta-\tfrac N2\log(2\pi).
$$

第一项衡量残差在协方差尺度下的大小，第二项来自高斯密度的归一化因子，第三项只随样本数变化。若 $C_\theta=LL^\top$ 是 Cholesky 分解，解 $L\mathbf b=\mathbf r$ 后即可用 $\mathbf b^\top\mathbf b$ 计算二次型，用 $2\sum_i\log L_{ii}$ 计算对数行列式。训练代码对负边缘对数似然做自动微分，以 Adam 更新均值和受正值约束的 kernel、噪声参数。

把拟合结果转换回物理单位时，各长度尺度乘以对应输入的训练标准差，信号和噪声方差乘以训练温度标准差的平方。均值乘训练温度标准差再加训练温度均值。这些变换使同一个模型可以在物理坐标和温度单位下直接用于路线评分。在线阶段可加载已拟合好的参数，再以初始观测 $\mathcal D_0$ 做条件化。拟合数据负责估计参数，$\mathcal D_0$ 决定规划开始时的后验。后续步骤固定已拟合的参数。

## 3. 从联合高斯推出 GP 后验

取任意查询输入 $U$，并记 $C=K_{XX}+\sigma_n^2 I_N$。独立观测噪声只进入观测块，因此

$$
\begin{bmatrix}\widetilde{\mathbf T}_X\\\mathbf T_U\end{bmatrix}
\sim\mathcal N\!\left(
\begin{bmatrix}\mathbf m_X\\\mathbf m_U\end{bmatrix},
\begin{bmatrix} C&K_{XU}\\K_{UX}&K_{UU}\end{bmatrix}
\right).
$$

把查询温度写成一项由观测线性预测的量和一项剩余量。令 $G=K_{UX}C^{-1}$ 及 $\mathbf e=\mathbf T_U-\mathbf m_U-G(\widetilde{\mathbf T}_X-\mathbf m_X)$。于是 $\operatorname{Cov}(\mathbf e,\widetilde{\mathbf T}_X)=K_{UX}-GC=0$。两者联合高斯，零交叉协方差给出独立性。固定观测值之后，剩余量仍为零均值，其协方差为 $K_{UU}-K_{UX}C^{-1}K_{XU}$。因此，后验均值和任意两个查询列表间的后验协方差为

$$
\begin{aligned}
\boldsymbol\mu_{\mathcal D}(U)
&=\mathbf m_U+K_{UX}C^{-1}(\widetilde{\mathbf T}_X-\mathbf m_X),\\
\Sigma_{\mathcal D}(U,V)
&=K_{UV}-K_{UX}C^{-1}K_{XV}.
\end{aligned}
$$

第二式可由对拼接列表 $(U,V)$ 计算后验再取对应块得到。记 $v_{\mathcal D}(q)=\Sigma_{\mathcal D}(q,q)$。它给出潜在温度的后验方差，$\sqrt{v_{\mathcal D}(q)}$ 是后验标准差。若预测一批新的含噪读数，则其协方差为 $\Sigma_{\mathcal D}(U,U)+\sigma_n^2 I$。

计算时复用 $C=LL^\top$。先解 $L\mathbf b=\widetilde{\mathbf T}_X-\mathbf m_X$，再对查询列表解 $LB_U=K_{XU}$。这样均值是 $\mathbf m_U+B_U^\top\mathbf b$，协方差是 $K_{UV}-B_U^\top B_V$。后验协方差由输入位置、时间和固定超参数决定。后验均值还取决于观测残差。

## 4. 同一批候选测量如何产生三种路线收益

### 4.1 候选事件的条件协方差与方差下降

记 $S=S(P)$ 为候选路线 $P$ 的预计采样事件列表，构造方法见 §5.2。取固定时空评价列表 $Q$。若 $S$ 为空，三种目标的收益均为零。其他情况下，在同一个当前后验下定义

$$
M_S=\Sigma_{\mathcal D}(S,S)+\sigma_n^2 I_{|S|},\qquad
H=\Sigma_{\mathcal D}(Q,S).
$$

$M_S$ 是未来含噪读数的联合协方差，非对角元素表示事件之间的相关性。$H$ 的第 $q,s$ 项表示评价点与候选事件的潜在温度如何共同变化。未来观测噪声独立，故它也等于 $\mathbf T_Q$ 与未来读数 $\widetilde{\mathbf T}_S$ 的条件交叉协方差。

将 §3 的条件高斯公式再用一次，以当前后验为起点，以未来读数为条件。更新后的 $Q$ 上协方差等于当前协方差减去

$$
\Delta_Q(S\mid\mathcal D)=H M_S^{-1}H^\top.
$$

因为 $\sigma_n^2>0$，$M_S$ 正定。令 $M_S=L_SL_S^\top$，解 $L_S Z=H^\top$，便得到 $\Delta_Q=Z^\top Z$。所以每个评价点 $q$ 的方差下降量 $d_q(S)=[\Delta_Q]_{qq}=\sum_j Z_{jq}^2\ge 0$。这个量对所有可能的未来读数都相同，可以在采样前直接算出。

只有一个候选事件 $s$ 时，矩阵式退化为

$$
d_q(\{s\})=
\frac{\Sigma_{\mathcal D}(q,s)^2}{v_{\mathcal D}(s)+\sigma_n^2}.
$$

分子表示事件对目标位置的关联强度，分母是该事件含噪读数的方差。若有多个事件，联合矩阵 $M_S$ 同时处理它们之间的冗余。

### 4.2 MI：沿途测量提供多少联合信息

MI（Mutual Information）衡量未来含噪读数 $\widetilde{\mathbf T}_S$ 对同一批事件上的潜在温度 $\mathbf T_S$ 提供多少信息。由条件互信息定义，

$$
J_{\mathrm{MI}}(S\mid\mathcal D)
=h(\widetilde{\mathbf T}_S\mid\mathcal D)
-h(\widetilde{\mathbf T}_S\mid\mathbf T_S,\mathcal D).
$$

对 $m$ 维高斯向量，微分熵为 $\tfrac12\log((2\pi e)^m\det\Sigma)$。上式第一项的协方差是 $M_S$。给定 $\mathbf T_S$ 后，第二项的协方差只剩 $\sigma_n^2 I_m$，其中 $m=|S|$。相减后公共常数抵消：

$$
J_{\mathrm{MI}}(S\mid\mathcal D)
=\tfrac12\bigl[\log\det M_S-m\log\sigma_n^2\bigr]
=\tfrac12\log\det\!\bigl(I_m+\sigma_n^{-2}\Sigma_{\mathcal D}(S,S)\bigr).
$$

其中 $\sigma_n^{-2}=1/\sigma_n^2$。直接复用 $L_S$ 的对角元素即可计算对数行列式。协方差的每个特征值 $\lambda_j$ 对 MI 贡献 $\tfrac12\log(1+\lambda_j/\sigma_n^2)$，故相关性通过联合矩阵进入评分，收益以 nats 计。

原 GP-Planning 在固定候选节点全集 $V$ 上计算 $I(\widetilde{\mathbf T}_S;\mathbf T_V\mid\mathcal D)$，即观测前后 $\mathbf T_V$ 的熵差。若 $S$ 的潜在输入全都包含在 $V$ 内，则给定 $\mathbf T_V$ 已知 $\mathbf T_S$，读数的条件熵只由独立测量噪声决定，因此两种写法等价。沿边事件可以落在任意位置与时刻，当前 MI 直接使用 $\mathbf T_S$ 作为目标变量。EMSE 和 WIVR 使用固定的评价列表 $Q$。

### 4.3 EMSE：评价域的平均平方误差期望下降多少

对查询点 $q$，任取预测值 $a$。把 $T(q)-a$ 拆成 $T(q)-\mu_{\mathcal D}(q)$ 和 $\mu_{\mathcal D}(q)-a$。展开平方后，由于 $\mathbb E[T(q)-\mu_{\mathcal D}(q)\mid\mathcal D]=0$，交叉项消失，于是

$$
\mathbb E[(T(q)-a)^2\mid\mathcal D]
=v_{\mathcal D}(q)+(\mu_{\mathcal D}(q)-a)^2.
$$

取 $a=\mu_{\mathcal D}(q)$ 可最小化期望平方误差，其最小值为后验方差。采样后使用更新后的后验均值，最小期望平方误差的下降量就是 §4.1 的 $d_q(S)$。

为计算时空域的平均下降量，将空间域划分为有效面积 $a_i>0$ 的单元，用 $\mathbf p_i$ 代表第 $i$ 个单元。在相同的 $L$ 个等间隔评价时刻 $t_\ell^Q$ 上，用 $q_{i\ell}=(\mathbf p_i,t_\ell^Q)$ 构成 $Q$。用代表点近似空间积分，并以等权近似时间平均，得到

$$
J_{\mathrm{EMSE}}(S\mid\mathcal D)
=\frac{\sum_i a_i\sum_{\ell=1}^L d_{q_{i\ell}}(S)}
       {L\sum_i a_i}.
$$

单位面积的重要性相同，边界单元按其有效面积参与平均。若所有单元等面积，该式就是 $\operatorname{tr}(\Delta_Q)/|Q|$。路线搜索最大化 EMSE 的下降量，等价于在同一当前后验与同一评价域下最小化采样后的剩余 EMSE。这里的期望由 GP 后验定义。

GP-Planning 的单事件 EMSE 公式对条件方差下降求和。评价列表固定且等权时，除以评价点数只改变分值的比例。按有效面积加权则对应不同面积单元对空间平均的贡献。

### 4.4 WIVR：带空间重要性的方差下降量

WIVR（Weighted Integrated Variance Reduction）在同一个 $Q$ 上为每个评价点指定非负权重 $w_{i\ell}$。只要权重总和为正，归一化收益就是

$$
J_{\mathrm{WIVR}}(S\mid\mathcal D;w)=
\frac{\sum_{i,\ell} w_{i\ell}\,d_{q_{i\ell}}(S)}
     {\sum_{i,\ell} w_{i\ell}}.
$$

权重加在希望减少误差的评价点上。一个候选事件即使位于低权重区域，也可以通过与高权重区域的后验交叉协方差取得较大的收益。取 $w_{i\ell}=a_i$ 时，公式与 §4.3 的面积平均 EMSE 相同。两种收益都具有温度平方的单位。

GP-Planning 的 WIVR 使用按距池边距离衰减的固定空间权重。Adaptive WIVR 保留同样的加权方差下降结构，在第 $k$ 轮开始时使用当前数据 $\mathcal D_k$ 与权重参考时刻 $t_k$ 计算

$$
u_i^{(k)}=v_{\mathcal D_k}((\mathbf p_i,t_k)),\qquad
w_{i\ell}^{(k)}=a_i u_i^{(k)} \quad (\ell=1,\dots,L).
$$

$u_i^{(k)}$ 是该空间单元的单位面积重要性，乘 $a_i$ 后得到离散积分中的总权重。同一空间单元在各评价时刻共享这个权重，方差下降 $d_{q_{i\ell}}(S)$ 仍通过 GP 随评价时间变化。如果所有 $u_i^{(k)}$ 都低于数值容差，使用 $w_{i\ell}=a_i$。本轮所有候选路线及递归子问题使用同一份权重。实际 GP 更新后可用新后验计算下一轮权重。

### 4.5 条件增益怎样在子路线之间传递

设 $A$ 是已经排入候选前缀的事件，$B$ 是拟追加的事件，两者按事件身份去重。用 $\Sigma_{\mathcal D\mid A}$ 表示对 $A$ 的假设读数作条件化后的协方差。它由 §4.1 的公式算得，所需输入只有事件位置、时间和噪声方差。定义追加收益 $J(B\mid\mathcal D;A)=J(A\cup B\mid\mathcal D)-J(A\mid\mathcal D)$。这个差值可直接用 $\Sigma_{\mathcal D\mid A}$ 给 $B$ 评分。

对 MI，联合熵按链式法则拆成前缀读数的熵与给定前缀后的后缀读数条件熵，独立观测噪声的熵也按批次相加。对 EMSE 和固定权重的 WIVR，采样前后方差之差可先减去前缀更新后的方差，再加上它与最终方差之差，因此逐点相消。三种目标都满足

$$
J(A\cup B\mid\mathcal D)
=J(A\mid\mathcal D)+J(B\mid\mathcal D;A).
$$

这一步让递归搜索按前缀已有的信息评估后缀的新增价值。Adaptive WIVR 在整轮冻结权重，才可使用同一组权重得到上述相消关系。

## 5. 从路径几何到时空采样事件

### 5.1 路线、边代价与预算

设候选 Checkpoints 与 Home 构成顶点集合 $V$，有序路线为 $P=(v_0,\dots,v_M)$。路径规划器为可达的有向边 $(i,j)$ 提供路径折线 $g_{ij}$。对路径顶点 $\mathbf a_0,\ldots,\mathbf a_m$，令 $s_0=0$，逐段计算累计弧长 $s_r=s_{r-1}+\|\mathbf a_r-\mathbf a_{r-1}\|$，末项 $s_m=L_{ij}$ 即路径总长。当前沿边实现用名义速度 $v_{\mathrm{nom}}$ 和边时间补偿 $\delta_{ij}$ 估计运动耗时：

$$
d_{ij}=\frac{L_{ij}}{v_{\mathrm{nom}}}+\delta_{ij}.
$$

$d_{ij}$ 与路径几何共同构成边数据。出发时刻为 $t_0^P$ 时，第 $j$ 个 Checkpoint 的预计到达时刻为 $t_j^P=t_0^P+\sum_{r=1}^j d_{v_{r-1},v_r}$。路线须在搜索截止时刻 $t_{\max}^P$ 前到达 Home $h$，亦即 $v_M=h$ 且 $t_M^P\le t_{\max}^P$。$t_{\max}^P-t_0^P$ 是本次可用于导航搜索的预算。在这些约束下，GRG 近似最大化 $J(S(P)\mid\mathcal D)$。实际执行时另以任务时钟检查剩余时间和返航裕度。

### 5.2 同一个采样时钟如何跨边运行

固定采样时钟原点 $\phi$ 与间隔 $\Delta_s>0$，全局 tick 为 $\tau_n=\phi+n\Delta_s$。沿每条正耗时边，只取满足 $t_{j-1}^P<\tau_n\le t_j^P$ 且处于采样窗口内的 tick。这样交界时刻只归属于上一段，每轮从规划起点之后的首个 tick 开始，跨边连续使用同一相位。

对某个落在边 $(i,j)$ 上的 tick，把已走时间比例映射为弧长 $\widehat s=L_{ij}(\tau_n-t_{\mathrm{edge}})/d_{ij}$。其中 $t_{\mathrm{edge}}$ 为该边的预计起始时刻。找出包围 $\widehat s$ 的两项 $s_{r-1},s_r$，在对应非零长度线段内插值：

$$
\beta=\frac{\widehat s-s_{r-1}}{s_r-s_{r-1}},\qquad
\widehat{\mathbf p}_P(\tau_n)=(1-\beta)\mathbf a_{r-1}+\beta\mathbf a_r.
$$

最终将 $(\widehat x_P(\tau_n),\widehat y_P(\tau_n),\tau_n)^\top$ 加入 $S(P)$。这个过程为整条路线生成预计输入，沿途与返航边遵循相同规则。路径形状、预计边耗时和出发时刻都会影响 $S(P)$。执行阶段的采样器按实际位姿与实际时间戳生成观测，可以在运动或等待期间取得读数。GP 输入与预计事件、实际记录、评价时刻共用同一时间基准和单位。

## 6. GRG 如何递归找路线

### 6.1 子问题和预算分割

固定一次搜索的 GP 快照、起始时刻 $t_0^P$、路径图、评分目标和所需权重。递归子问题记为 $\operatorname{Search}(R,g,b,d)$，其中 $R$ 是已经排定的路线前缀，$s=R[-1]$ 是前缀末端，$g$ 是本层目标，$b$ 是到达 $g$ 的截止时刻，$d$ 是剩余递归深度。前缀的完成时刻 $t(R)$ 根据 §5.1 从同一个 $t_0^P$ 累加得到。

先检查直接把 $g$ 接到 $R$ 的路线能否满足 $b$。当前递归以该路线作为可行性入口和基准。当 $d=0$ 时返回基准路线。当 $d=1$ 时，比较直接结束和插入一个可行中间点的路线，保留本层得分最高的可行候选。

当 $d>1$ 时，枚举中间点 $v$，利用两条直达边估计分割时刻的可行区间：

$$
b_{\mathrm{lo}}=t(R)+d_{sv},\qquad
b_{\mathrm{hi}}=b-d_{vg}.
$$

下界是直接从前缀末端到 $v$ 的预计时刻。上界为 $v$ 后至少预留一段直达 $g$ 的时间。若下界超过上界，当前搜索跳过该中间点。其余情况在区间内取 $m$ 个等间隔分割时刻 $b'_j=b_{\mathrm{lo}}+\tfrac{j}{m+1}(b_{\mathrm{hi}}-b_{\mathrm{lo}})$，其中 $j=1,\ldots,m$。对每个 $b'=b'_j$，先计算 $R_1=\operatorname{Search}(R,v,b',d-1)$，再计算 $R_2=\operatorname{Search}(R_1,g,b,d-1)$。第二次调用携带完整的 $R_1$，从 $t(R_1)$ 接续。$b'$ 只约束前段最迟结束的时间。路线事件仍按各边实际分配的预计耗时计算。

### 6.2 前缀条件化如何体现在整条路线的评分中

对某个固定 $R_1$，将所有候选后缀与它拼成整条路线，再按 $J(S(R_2)\mid\mathcal D)$ 比较。§4.5 表明，这个总分等于固定前缀分数加上后缀的条件收益，所以给整条路线联合评分就能实现对后缀的条件比较。连接点在传递前缀时已经存在，后续只追加新边，并在同一时钟上计算该边的事件。

对子路线的选择是贪心的。每个分割先保留一个前段，再用它递归搜索后段，最后将完整路线与直接路线及其他分割的候选比较。根调用以当前位置或当前目标的预计到达位置为起点，以 Home 为终点。有限递归深度、分割离散化和前段贪心选择共同限制了搜索空间，因此返回的是近似路线。

在固定的可选事件全集、固定当前 GP 与独立观测噪声下，MI 具有边际收益递减性质：单事件的条件增益为 $\tfrac12\log(1+v_{\mathcal D\mid A}(s)/\sigma_n^2)$，随已条件化的事件集扩大，方差只会减小。EMSE 和固定权重 WIVR 具有单调性，因为各评价点的方差下降非负。一般 GP 的方差降低目标可能出现互补性，单调性本身不足以证明次模性或推出 GRG 的近似保证。分割与递归结构参考 [Binney、Krause 和 Sukhatme, 2013](https://doi.org/10.1177/0278364913488427)。

## 7. 用实际测量批量更新 GP

设当前数据为 $\mathcal D_k$，新到的一批有效观测为 $\mathcal B_k$。记它们的实际输入为 $B$，读数为 $\widetilde{\mathbf T}_B$。定义这批观测的预测误差和含噪协方差：

$$
\mathbf r_B=\widetilde{\mathbf T}_B-\boldsymbol\mu_{\mathcal D_k}(B),\qquad
M_B=\Sigma_{\mathcal D_k}(B,B)+\sigma_n^2 I_{|B|}.
$$

将 §3 的条件高斯公式应用于当前后验，得到

$$
\begin{aligned}
\boldsymbol\mu_{\mathcal D_{k+1}}(U)
&=\boldsymbol\mu_{\mathcal D_k}(U)
+\Sigma_{\mathcal D_k}(U,B)M_B^{-1}\mathbf r_B,\\
\Sigma_{\mathcal D_{k+1}}(U,V)
&=\Sigma_{\mathcal D_k}(U,V)
-\Sigma_{\mathcal D_k}(U,B)M_B^{-1}\Sigma_{\mathcal D_k}(B,V).
\end{aligned}
$$

第一式用新测量相对旧预测的残差修正均值。第二式从当前协方差扣除已取得的信息。固定模型参数和噪声模型时，先后对 $\mathcal D_k$ 和 $\mathcal B_k$ 条件化，等价于将新记录追加到 $\mathcal D_k$ 后一次条件化。仓库的批量更新采用后一种计算方式：拼接实际输入与读数，重新分解加入噪声的训练协方差。以记录身份判断一条读数是否已纳入，保留具有相同输入的独立读数。

## 8. 后验更新与路线更新怎样衔接

系统使用采样时钟取得离散观测。仓库的沿边闭环按配置的间隔触发 GP 更新，将缓冲区内已收到的观测合并为一批，并用更新后的 GP 快照发起搜索。WIVR 的权重参考时刻 $t_k$ 取本批最后一条测量的时刻。如果机器人正向某个目标运动，下一次搜索从该目标及其预计到达时刻 $t_0^P$ 开始。$t_k$ 决定空间权重，$t_0^P$ 决定预计样本的时间安排。

新的候选路线在当前路径段结束后才进入下一段的选择。选用前核对路线是否对应当前目标、是否已有完成的搜索结果，再用实际任务时间检查下一段与返航是否仍能放进预算。若 $t_{\mathrm{now}}$ 为当前实际时刻，拟执行 $s\to v$，至少需要为

$$
t_{\mathrm{now}}+d_{sv}+d_{vh}+r_{\mathrm{home}}
$$

保留时间，其中 $r_{\mathrm{home}}$ 是额外返航裕度。超出任务截止时刻或触发返航规则时，执行返航分支。导航边代价属于预测量，实际耗时由任务时钟更新。GP 批处理触发、下一段路线启用和采样时钟分别决定何时吸收读数、何时改变路线和何时取得读数。

## 9. 端到端计算顺序

1. 加载拟合得到的常数均值、RBF 参数和噪声，以初始观测构造 GP 后验及其 Cholesky 分解。
2. 固定当前 GP 快照，读取边的路径几何、预计耗时与剩余导航预算。对 Adaptive WIVR，计算并冻结本轮评价权重。
3. GRG 递归枚举中间点与时间分割。对每条可行候选路线，用同一采样时钟生成 $S(P)$，计算联合事件协方差，再用当前选定目标评分。
4. 启用可执行的路径段，同时缓冲带实际位置、时间与温度的观测。到达 GP 更新触发条件时，批量加入有效读数。
5. 需要重规划时，从更新后的 GP 快照开始步骤 2，并按当前目标和剩余时间启用下一段。执行结束后，将剩余有效观测纳入最终后验。

原方法可对照 [GP-Planning 的 GRG 与 MI](../GP-Planning/src/planners.py) 和 [时空 GP 与方差下降目标](../GP-Planning/notebooks/06_temporal_nonstationary.ipynb)。训练过程见 [GP 模型与拟合代码](../Code/ROS_simulation/Trained_Real_Data_Kernel/testing_code/gp_model.py)。[沿边搜索核心](../Code/Verification%20tests/ROS_closed_loop_test/code/src/core.hpp) 提供 MI、EMSE 与 WIVR 三种目标，[闭环调度](../Code/Verification%20tests/ROS_closed_loop_test/code/src/nodes.cpp) 使用 Adaptive WIVR。

---

## 10. Experiment Plan — Continuous Edge Sensing and Real-Data-Informed Synthetic Fields

### 10.1 Scope, research question, and evidence status

**Status: proposed revision; the new continuous-edge/SRF experiment has not been implemented or validated by this document edit.** Preserve the GP–GRG–Nav2–Mission Manager architecture and compare MI, EMSE, and Adaptive WIVR for spatiotemporal temperature reconstruction/interpolation under a common 900 s mission budget.

**Adaptive WIVR status: Design Defined; Implementation Pending.** The variance-based weighting rule is specified in §4.4; neither its implementation nor an RMSE improvement has been validated. Future temperature forecasting is outside the current research scope.

**Research question:** With the same Initial GP, simulated field realization, navigation environment, candidate Checkpoints, continuous sampling rule, and GRG settings, which scoring objective yields the lowest Synthetic Reconstruction RMSE?

| Evidence category | Existing location / scope | How it may be used |
| --- | --- | --- |
| Historical Python experiments | Original 10.18 / 11.2 experiments, currently organized under `Code/Robot_close_loop_system/` and `Code/Policy_compare/` | Preserve code, saved models, logs, and established results; reuse verified components where applicable. |
| Completed node-based Nav2 prototype | `Code/ROS_Nav2_policy_compare/result/synthetic_ros_full_01/` and its saved reports | Evidence for the previous synthetic-record, checkpoint-sampling workflow. It does not validate continuous edge sensing or a GSTools SRF. |
| Proposed revised experiment | Continuous sampling on Nav2 edges, segment-end batch updates, and an independent SRF temperature source | Pending implementation, component tests, integrated MI baseline, and the subsequent three-policy comparison. |
| Physical WELL-E experiment | Future deployment and validation | Pending; neither synthetic navigation results nor ideal-localization tests establish hardware performance. |

Do not overwrite the original 10.18 or 11.2 experiments, the node-based Nav2 prototype, or any completed result. Use a separate implementation/configuration and result namespace for the revised protocol; its exact new location remains to be specified. Retain one experiment README for purpose, objects and dependencies, and one `Comparison.ipynb` for saved results, figures and conclusions.

### 10.2 Real datasets, Initial GP, and common reference frames

- Physical working area: **approximately 97 × 64 m, subject to confirmation from the registered real-world map and valid operating-area mask**. This is an enclosing extent, not a claim that the full rectangle is traversable. Neither the earlier 30 × 30 m Intel simulation nor the 10 × 8 m Nav2 fixture represents the current real site.
- Use available real Lawnmower temperature data for Initial GP training. Retain the planned Dataset 0–3 historical training arrangement when available; do not invent missing files or observations. Label any pilot using fewer datasets separately.
- Fit normalization, RBF GP hyperparameters, and empirical statistics/trends for simulation using historical training data only. Freeze the Initial GP and simulation configuration before comparing policies.
- Reserve **Dataset 4 for independent real-data reconstruction/interpolation validation**, subject to verified spatial support, temporal coverage and a predefined held-out protocol. It must not supply simulated sensor readings, fit the SRF, select its parameters/seeds/trends, tune weighting rules, or train/update the independent validation GP.
- Preserve measured coordinates, dates, timestamps, source identities, units and synchronization assumptions. Register runs to a common metric map frame; a GNSS trajectory is not an occupancy map.
- Define the ROS mission origin and its explicit mapping to the real-data and SRF time references. Do not compress dates or stretch recorded intervals to create apparent temporal coverage.
- Evaluate the frozen real-data model only on a predefined interpolation-valid held-out subset. If Dataset 4 occurs entirely after the training interval or otherwise lacks suitable support, mark independent real-data reconstruction validation **Pending**. Do not automatically substitute future forecasting. Synthetic observations must not enter this independent model.

### 10.3 Checkpoints, feasible paths, and continuous route events

1. Use a 2D map registered to the real WELL-E operating region, with the same physical dimensions and work-area mask.
2. Generate a uniform **1 m base grid**. Reject points outside the work area, inside obstacles, without sufficient robot-footprint/safety clearance, or unreachable through Nav2. Verify connectivity to the start/Home.
3. Initially select **30 candidate Checkpoints plus a separate Home; Home is not included in the candidate count**. This is a provisional final-experiment setting and may be adjusted after pilot verification of spatial coverage, Nav2 reachability, route-event count, and GP/GRG latency and memory. Document the subset-selection rule and coverage; the retained subset may be sparser than 1 m, and a route need not visit every candidate. Freeze the count and candidate set before the formal policy comparison and use the same configuration for all policies.
4. GRG continues to output an ordered Checkpoint Route. A Checkpoint is a route-optimization and segment-boundary reference, not a mandatory stop-and-measure site.
5. Retain Nav2 path geometry and estimate motion timing/progress along each edge. Convert the entire candidate route into expected `(x,y,t)` sampling events using a common **initial 1 s sampling interval** and global sampling phase.
6. Score the joint event collection across the complete route, including the declared return-segment sampling rule. Avoid duplicate events at edge junctions. Do not sum independent edge gains or assume independent 1 Hz observations.
7. Recompute affected event predictions when navigation geometry, timing, current pose or the task schedule changes. A pairwise travel-time matrix alone cannot determine where measurements will be taken along a detour.
8. Keep the **1 m Checkpoint base grid**, **10 m online WIVR/EMSE spatial scoring grid**, and **0.1 m / 0.5 s SRF visualization or offline evaluation references** separate. Their purposes differ; scoring representatives need not coincide with route Checkpoints.

Sampling during navigation, including return motion, is the proposed default. Rules for samples produced during planning pauses, waiting, recovery and target switching must be frozen and tested. Segment-based Nav2 goal handling may still slow or stop the robot; continuous sensing does not by itself establish uninterrupted motion.

### 10.4 Independent GSTools SRF temperature source

Construct an on-demand simulated temperature field

$$
T_{\mathrm{sim}}(x,y,t;\eta),
$$

where $\eta$ identifies one fixed random-field realization. Use a physical extent of **approximately 97 × 64 m, subject to confirmation from the registered real-world map and valid operating-area mask**, together with the common coordinate/time conventions.

An initial candidate is a **Matérn covariance model** for a spatially/temporally scaled SRF, with an empirical mean or deterministic trend informed by historical Lawnmower data. The exact spatiotemporal construction, spatial and temporal scales, smoothness, variance, mean/trend and units remain **Pending**. Time in seconds must not be silently treated as another spatial coordinate in metres. Independent 2D redraws at successive timestamps are not a validated temporal model.

Calibrate plausible temperature range, spatial variation scales and temporal behavior from historical data; characterize the resulting distribution rather than assuming every random realization remains inside the historical minimum and maximum. The generator remains separate from the planning GP: the proposed Matérn SRF is the simulation source, while the planner retains the RBF GP. Do not use the GP Posterior Mean as Ground Truth or claim the generated field is the actual measured field.

For a given seed, every policy must encounter the same field realization, trend and environment. Its values at unmeasured inputs remain inaccessible to online planning. SRF parameters and seed selection must not be tuned against Dataset 4 or synthetic evaluation RMSE.

**On-demand and memory requirements**

- Evaluate the field only at requested sensor or offline evaluation inputs. Accept actual continuous coordinates and timestamps; do not require them to lie on a dense lattice.
- Do not precompute or retain a complete 3D space–time temperature array. Retain the reproducible generator state, required measurements, configurations, GP snapshots and necessary statistics.
- Treat **0.1 m spatial resolution** and **0.5 s temporal resolution** as initial visualization, discrete-evaluation or output references, not limits on SRF query coordinates or the 1 s sensor schedule.
- Generate large evaluation inputs in fixed indexed batches and accumulate error statistics without materializing the whole temperature volume. Batch size, peak memory, runtime and the generator's storage behavior require measurement.
- Test repeated queries, reversed/shuffled query orders, different batch partitions, and a fresh generator reconstructed from the same saved configuration. Each must reproduce the same noise-free field at identical inputs within a documented numerical tolerance. Do not reinitialize a different realization at each request.
- If sensor noise is enabled, use a separate reproducible noise mechanism with its own seed and stable measurement identities. Evaluation queries and batching must not consume or alter the sensor-noise stream. Define whether repeated physical measurements receive independent noise; keep noise-free field truth separate from noisy observations.

### 10.5 Continuous-sensing ROS 2 / Gazebo mission

Establish an integrated **MI baseline first**, using the Gazebo Bunker Mini and Nav2, before substituting EMSE and Adaptive WIVR.

1. Load the same saved Initial GP, registered map, candidate Checkpoints, Home, SRF configuration/seed, noise configuration, and mission timing setup.
2. Obtain feasible Nav2 paths and expected motion schedules; predict continuous route events and run MI-based GRG.
3. Execute the next Checkpoint-to-Checkpoint segment through Nav2. Let the simulated body move through the physics/controller stack; do not substitute direct pose changes for execution.
4. While navigating, initially sample every 1 s using the **actual simulated pose and timestamp** to query the fixed SRF. Record the noise-free source value and noisy observation separately when applicable, with source/measurement identities and quality flags.
5. Buffer valid observations. Initially, after each completed segment, perform a **batch GP posterior update**; for Adaptive WIVR, refresh the 10 m grid weights from the updated posterior at the current time. Update remaining time and path/event estimates, freeze weights for the next GRG search, and replan. MI/EMSE skip adaptive weighting.
6. Handle navigation failures independently from measurement validity: valid samples already acquired on a partially executed edge remain real observations of the simulated field. Specify and test their batch-update handling; do not fabricate samples for missing positions or times.
7. Stop extending the sampling route early enough to execute Home and the required final online update within the budget. Record actual return success, final pose, duration and termination reason.

There is **no fixed per-checkpoint sensing dwell** and no Dataset 4 recorded-data Acquisition Pool in this revised mission. The actual sensor response, viable 1 Hz rate, GP update frequency, replanning frequency, buffering policy, Nav2 target-switching behavior and computation/motion overlap remain to be validated.

### 10.6 Mission budget and stopping rule

Use a common **900 s** mission budget:

$$
\tau_{\mathrm{motion}}
+\tau_{\mathrm{planning}}
+\tau_{\mathrm{other}}
\leq 900\,\mathrm{s}.
$$

Motion includes navigation and return, with temperature sensing performed concurrently. Planning includes online GP updates, Adaptive WIVR weight calculation where applicable, and GRG work that actually occupies mission time. Other includes necessary non-overlapping waiting, switching or recovery overhead. Define a non-overlapping time ledger and use actual elapsed ROS simulation time as the authoritative budget; do not add a separate sensing duration to motion, or count overlapping computation twice. Initial historical training and final offline evaluation are outside the mission clock.

Before accepting another segment, require sufficient remaining time for the segment, exposed online update/planning work, return, necessary extra overhead, and a predeclared safety margin. Include the final required online batch update in this accounting. Recheck after delays, failures, path changes and computation. Stop when no admissible continuation preserves return feasibility; do not use Ground Truth, reconstruction RMSE or a target sample count as the stopping rule.

Record any overrun, failed return or implementation watchdog as such. An early feasible return is allowed; do not invent measurements to fill unused budget. Exact reserve values and scheduling behavior are Pending experimental choices.

### 10.7 Policies, baselines, and fair comparison

| Method | Objective / role |
| --- | --- |
| History Only | Frozen Initial GP with no new simulated observations; the same evaluation field/grid/time interval. |
| GRG + MI | Joint conditional information gain from all predicted along-route events, accounting for correlation and redundancy. |
| GRG + EMSE | Average posterior-variance reduction over the common fixed spatiotemporal scoring domain, with effective-area correction. |
| GRG + Adaptive WIVR | Posterior variance reduction with dynamically updated uncertainty-based spatial weights over the same domain. |

Freeze and share the Initial GP, RBF hyperparameters, candidate set, start/Home, navigation map, Gazebo model, Nav2/localization configuration, SRF realization/seed/trend, noise protocol, continuous sensing interval/phase, batch-update and replanning rules, mission time mapping, 900 s budget, return rule, GRG settings and reconstruction evaluation protocol. Only the scoring objective and its necessary weight computation differ; account for their actual online costs.

**Defined Adaptive WIVR configuration**

- Partition the approximately 97 × 64 m enclosing region into **10 × 10 m** cells: approximately 70 before the valid operating-area mask. Keep one representative point inside each valid cell/area intersection, with effective area $a_i>0$. Fix the grid origin, mask and representative-point rule before runs; registered details and the actual valid count remain to be confirmed.
- Use a fixed scoring set $Q=Q_s\times\mathcal T_Q$, with the same spatial representatives, mask, evaluation times and effective-area treatment for EMSE and WIVR. Predeclare the common equally spaced temporal scoring grid; no separate temporal importance function is added.
- At planning iteration $k$, query the current 3D GP at $(\mathbf p_i,t_k)$ and compute $w_i^{(k)}=a_i\sigma_{\mathcal D_k}^{2}(\mathbf p_i,t_k)$. Reuse this spatial weight at every scoring time for that cell. No regional variance-averaging algorithm is introduced.
- Compute initial weights from the Initial GP; refresh only after actual Batch GP Updates. Freeze weights across all routes, hypothetical observations, gain calculations and recursive subproblems in one GRG search. Model-based conditioning on a proposed route must not refresh weights within that search.
- Use the normalized WIVR objective from §4.4 without a second weight normalization. If all current variances are near zero, fall back to effective-area weights $a_i$; predeclare and validate the numerical threshold.
- For the Uniform Spatial Weighting check, set per-unit-area importance constant, so weights are proportional to $a_i$. This must match area-averaged EMSE numerically, including boundary cells; with equal-area cells it reduces to the ordinary arithmetic-mean identity.
- Weights use only the online GP posterior, not unseen SRF temperatures, offline RMSE or Dataset 4 labels. Freeze the adaptive rule before evaluation, not the weight values for the whole mission. The earlier pool-distance rule and upper-region 3:1 fixture remain historical examples.

Use the same continuous-event construction at every GRG recursion level. Cache keys/invalidation must distinguish objective, GP version, event coordinates/times, grid/mask/area configuration, frozen weight version and affected path/timing state. Do not allow an Adaptive WIVR recursive call or shared cache to fall back to, or mix with, default MI. Checkpoints alone are insufficient cache identities.

Use multiple predeclared random-field seeds where feasible. Within each seed, pair all policies with the same realization and conditions; reset the simulator, initial pose/time, GP data, sampler phase, noise state, buffer and initial adaptive-weight state before every mission. Do not inherit observations or weights from another policy. Report paired seed-wise outcomes and variation; superiority remains unvalidated until these runs are completed.

### 10.8 Synthetic reconstruction and separate real-data validation

Define a common fixed spatiotemporal evaluation grid over the predeclared work-area domain and mission interval, independent of policy trajectories. Freeze its map mask, time origin/range, spatial/temporal resolution and batching protocol before evaluation. Do not restrict it to the convex hull of measurements acquired by a particular policy.

Use noise-free SRF values as Synthetic Ground Truth:

$$
\operatorname{RMSE}_{\mathrm{syn}}
=\sqrt{\frac{1}{N_{\mathrm{eval}}}
\sum_{j=1}^{N_{\mathrm{eval}}}
\left[\mu_{\mathrm{final}}(q_j)-T_{\mathrm{sim}}(q_j;\eta)\right]^2}.
$$

Compute large-grid metrics in batches. Compare History Only, MI, EMSE and Adaptive WIVR at identical offline evaluation inputs for each seed. The offline grid may be denser than the 10 m online scoring grid; explicitly record both. Online weighting must never read SRF Ground Truth or offline RMSE. Keep a common target interval despite different return times; separately flag any unsupported late-time queries as extrapolation, not interpolation evidence. Future forecasting is not the current research objective.

Additional outputs: MAE, posterior variance, error versus mission time and successful measurement count, spatial coverage versus temporal revisits, actual Nav2 trajectories, estimated/actual travel time, GP-update/GRG latency, peak memory, actual sampling interval, mission-time ledger, failure/recovery events and return success. Reconstruct curves from saved GP snapshots and logs; evaluation truth never feeds online choices.

Maintain separate result categories for **Synthetic Reconstruction RMSE**, **independent real-data reconstruction/interpolation validation**, and **future physical-robot experiments**. Dataset 4 remains an independent real-data source, but validation is **Pending** unless the frozen real-data training model and held-out inputs satisfy the stated spatial/temporal support protocol. Do not treat a later collection automatically as interpolation or replace missing coverage with a forecasting study. None of the new results are assumed complete.

### 10.9 Staged validation, implementation trace, and reports

1. **Review/reuse:** identify existing GP, GRG, Nav2, logging and numerical components. Preserve historical records and document which old assumptions are superseded.
2. **Component checks:** test Checkpoint filtering, path-to-event timing, global sampling phase/junction deduplication, correlated joint scoring, batch updates, recursive objective propagation, cache invalidation and non-overlapping budget accounting.
3. **SRF checks:** fit/check historical-data-informed parameters; test fixed-realization consistency, query order and batching, noise separation, on-demand latency and memory. Freeze the accepted configuration before comparisons.
4. **MI integration:** demonstrate actual Nav2 motion with continuous measurements, multiple segment-end GP updates/replans, failure handling and timely physical Home arrival in simulation.
5. **Policy comparison:** change only the scoring objective and its necessary Adaptive WIVR weight calculation; run the paired seed protocol under the same conditions.
6. **Reporting:** produce tables and figures from saved outputs; report independent real-data reconstruction/interpolation only when the held-out dataset and spatial/temporal support protocol are available.

**Adaptive WIVR-specific validation plan — all checks below are Pending, not passed results:**

1. Check 10 m grid generation, registered origin, valid operating-area mask, representative points and actual valid cell count.
2. Check current GP posterior variance at every scoring representative, including units and numerical nonnegativity.
3. Check that higher variance produces higher **per-unit-area** importance; distinguish this from total cell weight when cell areas differ.
4. Check weight refresh after each actual Batch GP Update and the near-zero-variance area-only fallback.
5. Check that weights remain identical for all routes and recursive subproblems within one GRG search.
6. Check numerical equality of Uniform Spatial Weighting and EMSE using identical areas, query domain and scoring times.
7. Check that WIVR scores predicted **continuous along-path sampling events**, not Checkpoints alone.
8. Check effective areas and representative-point validity for incomplete boundary cells.
9. Measure 10 m grid weight/scoring overhead; plan a paired **10 m versus 5 m** sensitivity comparison.
10. Check recursive objective propagation and cache isolation so WIVR never mixes with default MI.

Document each major calculation with its actual repository-relative file/function, input/output, equation, units, caller and downstream interface. Specifically trace Initial GP → current variance/adaptive weights (WIVR only) → path/timing → continuous events → joint score → GRG route → Nav2 execution → SRF observation → batch GP update → weight refresh/replanning → return. Use Pending implementation entries instead of inventing functions or claiming an interface already works.

Save full configuration/version/seed records, registered coordinate/time conventions, selected Checkpoints, predicted paths/events and actual trajectories, every valid/rejected observation, batch membership, GP snapshots, online computation and memory measurements, navigation outcomes and exact termination reasons. Also save the scoring-grid mask/representatives/effective areas, per-iteration variances and weight vectors, frozen weight versions, fallback events and weight-calculation latency. Record enough generator state to reproduce offline truth without storing the full 3D field.

Keep one README for the experiment purpose, test objects and dependencies, and one `Comparison.ipynb` for the actual results and concise conclusions. The notebook should include the four-method RMSE/MAE summary, RMSE versus mission time/sample count, variance, actual trajectories, sampling coverage/revisits, estimated versus actual travel time, GP-update/GRG cost and return/budget outcomes. New experiment tables remain **TBD** until the revised runs are completed; do not relabel previous node-based results as continuous-SRF results.

### 10.10 Pending decisions and completion criteria

| Item | Status / required evidence |
| --- | --- |
| Real sensor response and proposed 1 Hz measurement | **Pending:** response/delay, synchronization, valid sample rate and noise characterization. |
| 1 m grid, initial 30 candidates plus separate Home, and GRG cost | **Pending:** subset/coverage rule and validation of the provisional count against route-event count, GP size, latency and memory. The count may be adjusted after pilot verification; freeze the common candidate count/set before formal policy comparison. |
| SRF calibration from real data | **Pending:** Matérn/spatiotemporal form, units, range/variance, spatial/temporal scales, empirical trend and parameter-fit checks using historical data only. |
| On-demand SRF consistency and performance | **Pending:** same realization under repeat/order/batch/restart tests, reproducible noise, query latency and peak memory. |
| Adaptive WIVR | **Design Defined; Implementation Pending:** 10 m grid; current GP posterior-variance weights; representative-point variance with effective-area correction; no separate temporal weighting; refresh after each Batch GP Update; frozen within iteration. Implementation, numerical tests and RMSE improvement remain Pending. |
| Scoring-grid sensitivity | **Pending:** 10 m versus 5 m comparison of scoring behavior, runtime, memory and reconstruction outcomes. |
| Continuous ROS 2 system integration | **Pending:** path-to-event builder, continuous sampler, buffering, batch updates, actual replanning rate, target switching, budget and failure/return checks. Existing node-based tests do not complete this item. |
| Physical-robot validation | **Pending:** real sensor/navigation interfaces, localization, environment sensing and actual WELL-E trials. |
| Dataset availability and registration | **Pending:** available Dataset 0–3 training set, independent Dataset 4, confirmed approximately 97 × 64 m registered extent/mask, common map/time references and an interpolation-valid real-data holdout protocol. No automatic future-forecasting substitution. |
| References, paper notation and LaTeX | **Pending:** bibliography completion, formal manuscript symbol harmonization and publication formatting; this note revision is not a completed paper-format/reference update. |

The Adaptive WIVR weighting design is defined; its implementation and validation are pending. A complete reproducible experimental proposal additionally requires the outstanding dataset, map, timing, SRF and numerical configurations to be recorded. A working continuous-edge baseline requires runnable code and saved execution evidence; a completed policy comparison additionally requires the fixed SRF/evaluation protocol and paired policy results. This document edit does not establish either milestone.
