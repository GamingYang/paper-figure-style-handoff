# 图形索引

所有便携绘图函数在 `scripts/render_figures.py`，共享风格在 `scripts/figure_style.py` 与 `style.json`。建议先看 `qa/gallery.png`，再看需要的独立高分辨率图。

| 要复现的形式 | 默认示例输出 `outputs/` | Python 函数 | 主要数据 |
| --- | --- | --- | --- |
| 三组五边形雷达横版 | `radar_horizontal.*` | `radar(layout='horizontal')` | `fig4_5b_stage_profile_pentagon_source.csv` |
| 同样三组雷达竖版 | `radar_vertical.*` | `radar(layout='vertical')` | 同上 |
| 注意力曲线+profile+双轴+三响应面 | `attention_mechanism.*` | `mechanism()` | `attention_mechanism_cache.npz` |
| 消融/静态 MMD 差值图 | `ablation_effect_map.*` | `effect_map()` | `fig4_5a_effect_map_source.csv` |
| 电芯×seed 小提琴与 seed 点 | `cell_seed_violin.*` | `violin()` | 两份 `fig4_5c_*_source.csv` |
| 柱顶数值避开误差线 | `bar_error_labels.*` | `bar_summary()` | `attention_ablation_all_evidence_aggregate.csv` |
| 每电芯 RMSE/MAE 成对柱图 | `per_cell_error.*` | `per_cell_error()` | `attention_mechanism_seed42_per_cell_long.csv` |
| 2×4 电芯 SOH 预测曲线 | `prediction_curves.*` | `prediction()` | `adaptive_attention_v1_seed42_target_test_predictions.csv` |
| 每电芯极坐标柱图小多图 | `per_cell_polar_bars.*` | `polar_cells()` | 每电芯误差长表 |

`*` 默认包括 PNG/PDF/SVG，另外有 `.layout.json`，TIFF 可按命令导出。

## 原代码快照

| `originals/scripts/` 文件 | 保存原因 | 执行注意 |
| --- | --- | --- |
| `plot_stagewise_pentagon_radar_reference_style.py` | 最后横竖版雷达的历史生成逻辑 | 有旧项目路径、径向偏移和分面标尺 |
| `plot_section4_5_new_attention_ablation_figures.py` | 差值图、小提琴图及旧雷达的历史总入口 | 会读取旧项目工作簿并输出多种历史图，不是迁移入口 |
| `plot_adaptive_attention_mechanism_response_surface.py` | 注意力线段、双轴、关闭黑点云后的响应面 | 需要 PyTorch、原始 split 与 checkpoint |
| `visualize_attention_mechanism_ablation_v1.py` | 每电芯图、曲线、误差线标签修正、阶段解释图 | 部分仍是早期字体和总标题；只作追溯 |

4 个程序均按原字节复制，SHA-256 见 `provenance.json`。历史代码不是未来要求的优先来源。`render_figures.py` 对其中一个模块仅导入方法顺序、标签和色板，未运行其 main，也没有触发原路径 I/O。

## 历史图片

`originals/figures/` 保留 10 张原 PNG，包含两版雷达、差值图、小提琴图及最初的六张 attention 消融/机制图；另有 Section 4.5 四份 PDF。这些是历史实物参考，不标注为全部最终验收。

## 选图理由

- 要比较某个机理干预的误差变化：差值图，零线与完整模型对应。
- 要看阶段/多维误差结构：分组雷达，维度定义与单位统一。
- 要看真实重复/电芯异质性：violin + 原始点，图注明确统计单位。
- 要看模型是否跟住 SOH 曲线：多电芯预测曲线，不能只给汇总 RMSE。
- 要展示“注意哪里、聚焦多尖锐、对特征组合怎样响应”：六面板机制图；不要把响应面包装成误差参数优化。
- 要比较单电芯多个模型：极坐标小多图是参考形式之一，当前图用数字键减轻模型名字拥挤，图例一一对应。

样例机制图的阶段 profile、温度和熵已包含于六面板中，历史独立 `fig05_seed42_attention_stage_interpretability.png` 作为辅助参考保留，没有再复制成一套默认输出。
