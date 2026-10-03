# 数据契约与历史边界

所有 `data/` 的现成数值来自当前项目落盘结果或冻结模型推理，不是按参考图片手工猜测。历史文件的绝对来源、包内路径、字节数和哈希在 `provenance.json`。图片风格可迁移，实验值不能作为新论文的证据。

## CSV 输入

| 文件 | 必需列与用途 | 单位/粒度 |
| --- | --- | --- |
| `fig4_5b_stage_profile_pentagon_source.csv` | `label, source, axis, rmse_pct` | 每方法×轴一行，已乘100，不再乘100 |
| `fig4_5a_effect_map_source.csv` | `label, category, mean_rmse_pct, std_rmse_pct, mean_mae_pct, std_mae_pct, delta_rmse_pct, delta_mae_pct, order` | 每方法一行，已是 % SOH |
| `fig4_5c_multiseed_cell_violin_source.csv` | `label, seed, battery_index, cell_rmse_pct, cell_mae_pct` | 每方法×seed×电芯一行，已是 % SOH |
| `fig4_5c_multiseed_seed_points_source.csv` | `label, seed, pooled_rmse_pct, macro_rmse_pct, mae_pct` | 每方法×seed一行，已是 % SOH |
| `attention_mechanism_seed42_per_cell_long.csv` | `battery_index, method, label, rmse, mae` | 每方法×电芯一行，原始 SOH 比例误差，画图乘100 |
| `attention_ablation_all_evidence_aggregate.csv` | `method, mean_rmse, std_rmse`，另有其他统计列 | seed 聚合，原始 SOH 比例误差，画图乘100 |
| `adaptive_attention_v1_seed42_target_test_predictions.csv` | `battery_index, cycle, true_soh, adaptive_pred, original_pred` | 每循环一行；曲线 y 使用 SOH 比例，不乘100 |
| `mechanism_tables/*.csv` | 阶段 profile、温度/熵分箱、加权中心、可选点云 | 历史导出的辅助表，仅解释与核验 |

`fig4_5b_stage_radar_source.csv` 和 `attention_mechanism_seed42_stage_interpretability_long.csv` 为辅助/历史表，默认入口没有将三角雷达再生成一次。

## 指标语义

- Pooled RMSE：把某 split 中循环样本合并后的 RMSE。
- Macro RMSE：按项目落盘定义，电芯级 RMSE 的等权平均，不等于 pooled RMSE。
- Stage RMSE：按原项目的阶段划分对样本分组后求误差。机制缓存中 `p_norm=cycle/3000`，分界 1/3 与 2/3；不能擅自解释为“每个电芯生命周期各三等份”。
- 图中 `Overall` 与 `Macro` 是汇总统计，不是附加退化阶段。
- `delta_rmse_pct = mean_rmse_pct(method) - mean_rmse_pct(Proposed)`，负值意味着该对照在此指标上更低。
- 差值图的误差线是原表各方法标准差，**不是差值标准差、置信区间或显著性证据**。若要配对区间，需从同 seed 成对差值重新计算，并明确统计方法。
- 默认小提琴每方法有 3 个 seed、8 个电芯，共 24 个 cell-seed 条目；共享电芯使观测有关联。不能称为 24 个独立 seed。

## 缓存 NPZ

`attention_mechanism_cache.npz` 仅包含画图所需数值，无 pickle 对象，载入使用 `allow_pickle=False`。同行 JSON 记录实际 checkpoint 哈希、模型来源、设备、样本数、阶段定义及局限。

| Key | Shape/意义 |
| --- | --- |
| `stages` | 3 个阶段名，Early/Middle/Late |
| `representative_curve` | 3×T×3，三阶段代表样本的电压、差分、二阶差分 |
| `representative_attention` | 3×T，对应 token 权重 |
| `representative_indices/battery/p_norm` | 代表样本身份及归一化循环 |
| `tokens` | T 个 token 序号 |
| `profiles` | 3×T，各阶段平均注意力权重 |
| `attention_vmax` | 历史全样本权重 99.5% 分位数，用于曲线颜色范围 |
| `focus_bins` | K×4：循环中点、平均温度、平均熵、样本数 |
| `late_center` | 三种曲线特征的晚期注意力加权中心 |
| `vs_x, vs_y, vs_z` | 电压×斜率响应面网格，均为 120×120 |
| `vc_x, vc_y, vc_z` | 电压×曲率响应面网格 |
| `sc_x, sc_y, sc_z` | 斜率×曲率响应面网格 |

## 响应面解释

此包重现历史 `score_surface` 的定义。它固定晚期平均状态与注意力汇聚的 context，将这个 pooled context 输入 `hidden_proj`，在两种曲线特征上扫网格；第三种特征固定在晚期加权中心。每个面单独 min-max 归一化到 [0,1]。

因此它是条件兼容性示意，并非逐 token 隐状态上的完整真实注意力概率。颜色不能叫“参数优化 RMSE”、概率、统计显著性或因果效应；不同面的绝对响应强弱不能直接比较。人工网格组合也未必都对应可实现的电压曲线。迁移到新模型时要明确自己的条件化变量，并重新计算。

历史分箱为左右半开，最后一个箱也未包含恰等于最大边界的点；本包缓存保留这个事实，以免打包过程悄悄更改历史数值。新分析可以修复，但须重新生成并说明版本差异。

## 历史雷达与便携模板

历史 `normalized()` 使用 `r=0.18+0.77*value/vmax`，且每个面板单独确定 vmax。历史图原貌保留在 originals。便携模板采用 `r=value`、共同上限，减少跨面板形状误读；CSV 数值未改变。因此两套图片是同源数据、相同视觉语言，不是像素级相同图片。

## 原模型边界

历史最终模型：physical cubic degradation base + residual temporal network + degradation-state-guided curve attention v1 + life-cycle adaptive MMD；标识为 `adaptive-attention-v1`。`late-aware-v2` 未自动升级为最终模型。

Batch1–8 为源训练，Batch9 为目标适配/检查点选择，Batch10 为最终盲测报告。此包只冻结推理展示，不用 Batch10 选参数、选 seed 或训练。某些对照优于完整模型的数值照实保留，不能删掉 voltage-only token 来宣称所有特征扩展都必然更好。

## 可选重新提取

只有需要重新追溯原 checkpoint 时使用：

```powershell
python scripts/export_mechanism_cache.py --project-root 'C:\Users\ROG\Desktop\paper'
```

这一步需要原模型、processed_clean 数据及 PyTorch，执行冻结 eval 推理；不是训练。默认绘图不依赖它。导出脚本将 MKL 设置为 sequential 以避免本机两份 OpenMP runtime 冲突，没有启用忽略重复 runtime 的不安全开关。
