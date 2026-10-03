# 4.5 消融实验替换稿与图注

## 固定版本

- 最终模型：`adaptive-attention-v1`
- 模型表达：物理三次退化基线 + 残差时序网络 + 退化状态引导的曲线注意力 + 生命周期自适应 MMD
- 数据协议：Batch1-8 为源域训练，Batch9 用于目标域适配与检查点选择，Batch10 仅作为最终盲测报告集
- 注意：`late-aware-v2` 仅作为诊断尝试，不进入正文最终模型

## 4.5 消融实验

为避免消融实验停留在简单删减模块的层面，本节将模型拆解为三个可检验的建模假设：退化状态是否提供跨阶段索引，局部曲线形态是否提供电压退化证据，自适应聚焦强度是否能够调节不同寿命阶段的时间依赖。所有对照实验均保持 Batch1-8、Batch9 和 Batch10 的划分不变，其中 Batch10 只用于最终评价，不参与模型选择。

首先，对注意力结构进行等条件对比。所提出的退化状态引导曲线注意力在 3 个随机种子下取得 0.009906 的平均 RMSE，对应 0.991% SOH，平均 MAE 为 0.006189。原始拼接式注意力的平均 RMSE 为 0.010381，对应 1.038% SOH。两者差距并不表现为单一批次上的大幅跃迁，而是表现为误差中心和种子波动的同步收缩，说明新的注意力结构主要改变了特征聚合方式，而不是依赖额外输入信息获得偶然优势。

退化状态相关的对照用于回答一个更具体的问题：模型是否只是记住循环编号，还是能够利用由物理基线给出的退化位置。当移除退化状态后，平均 RMSE 增加至 0.010790；仅保留循环位置时，平均 RMSE 为 0.010391；打乱退化状态后，平均 RMSE 为 0.010091。该结果说明，退化状态对性能的贡献是温和但可观的，其价值主要体现在为注意力分配提供与寿命进程一致的条件变量，而不是简单替代时间索引。

局部曲线形态的对照进一步检验了电压轨迹信息的作用。去除曲线 token 后，平均 RMSE 上升至 0.010952，表明仅依赖全局时序状态不足以稳定刻画 SOH 退化过程。值得注意的是，仅使用 voltage token 的诊断模型在 Batch10 上得到 0.009141 的平均 RMSE，低于完整曲线 token 版本。该结果提示原始电压轨迹本身已经携带主要的局部退化信息，斜率和曲率展开并非在当前数据规模下带来单调收益。因此，本文保留完整曲线 token 的依据并非 Batch10 上的事后筛选，而是其与局部形态可解释性相一致；该诊断结果也为后续扩大电池数量后的形态特征筛选提供了边界依据。

自适应聚焦强度的作用由 fixed-temperature 对照验证。当注意力温度固定后，平均 RMSE 增加至 0.011829，且种子间标准差扩大至 0.002122。相比之下，完整模型的标准差为 0.000557。该结果表明，退化状态引导注意力的关键并不只是重新加权输入特征，而是在不同退化阶段调整注意力分布的集中程度。也就是说，早期缓慢退化、中期转折和后期快速衰退并不适合共享完全相同的聚焦尺度。

在领域适配部分，本文重新在新注意力结构下补充了 static-MMD 和 w/o adaptive MMD 对照，以避免沿用旧注意力实验造成结论混杂。完整模型的平均 RMSE 为 0.009906，static-MMD 为 0.010224，移除自适应 MMD 后为 0.010830。该结果说明 MMD 的作用不是直接主导预测精度，而是在目标域适配过程中抑制源域和目标域表征偏移；同时，生命周期自适应权重较固定 MMD 更符合不同退化阶段的分布差异。

最后，物理基线对照显示了结构先验的基础作用。移除三次退化基线后，平均 RMSE 增加至 0.016634，平均 MAE 增加至 0.011158，并且 early、middle 和 late 三个阶段的 RMSE 均明显升高。该结果说明，数据驱动残差网络并未完全替代物理退化趋势；相反，物理基线先给出低频寿命趋势，注意力网络再学习局部残差，是当前模型稳定性的主要来源。

综上，消融结果支持以下结论：物理基线提供全局退化骨架，自适应 MMD 缓解跨批次表征偏移，退化状态引导的曲线注意力则负责在寿命进程中动态选择局部曲线证据。三者对应的不是并列堆叠模块，而是退化趋势、域间适配和局部聚焦三个不同层面的建模约束。

## 图题与图注

**Fig. X. Ablation and static-MMD control of the adaptive-attention SOH model.**  
The plot summarizes the change in RMSE and MAE relative to the proposed model on the Batch10 blind test set. Markers denote the mean over three random seeds, and horizontal intervals denote one standard deviation. Positive values indicate larger error than the proposed model. The controls are grouped by MMD adaptation, degradation-state conditioning, local curve tokens, adaptive focus temperature, and physical degradation prior.

**Fig. X. Stage-aware RMSE profile.**  
The radar plots summarize five RMSE-related axes on Batch10, including early-stage RMSE, middle-stage RMSE, late-stage RMSE, overall RMSE, and macro RMSE. The pentagonal representation avoids the triangular geometry caused by using only three degradation stages, while retaining the stage-wise interpretation. Black vertex labels denote the corresponding values in % SOH, and a smaller polygon indicates lower prediction error.

**Fig. X. Multi-seed stability.**  
The violin plot shows cell-wise RMSE distributions across Batch10 cells and three random seeds. Black diamonds indicate seed-level pooled RMSE values, and horizontal bars indicate the median cell-wise RMSE. This plot separates average accuracy from seed stability and cell-to-cell dispersion.

## 可放在正文图注后的简短解释

消融实验并非仅用于说明某一模块是否“有用”，而是用于识别不同建模假设的误差来源。退化状态相关对照检验寿命阶段索引是否有效，曲线 token 对照检验局部电压形态是否携带可迁移退化信息，fixed-temperature 对照检验注意力聚焦强度是否需要随退化阶段变化。该设计使注意力机制的贡献从经验拼接转化为可解释的状态条件化聚合过程。
