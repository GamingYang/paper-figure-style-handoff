# 论文绘图风格复现与新窗口交接包

整理日期：2026-10-03。来源：本窗口锂电池 SOH 论文多轮绘图及用户修改意见。

这个文件夹同时保存“当时生成了什么”和“新论文怎样复用”。优先把本文件、`SKILL.md`、`references/USER_REQUIREMENTS.md` 交给新窗口。默认绘图入口只读取包内 CSV/NPZ，不需要旧工程路径、GPU 或重新训练。

## 新窗口先读

1. `SKILL.md`：简短的模型执行规则，引用文件夹即可使用，不必安装。
2. `references/USER_REQUIREMENTS.md`：用户原话摘录、最后确认的要求、适用边界。
3. `references/STYLE_GUIDE.md`：配色、字体、编号、避让和图形选择的具体做法。
4. `references/DATA_AND_PROVENANCE.md`：样例数据字段、单位、统计层级和历史图的解释边界。
5. `references/FIGURE_CATALOG.md`：图形与代码、数据、历史图片的对应关系。

新窗口可直接使用这段提示词，另有独立文件 `NEW_WINDOW_PROMPT.md`：

> 请先阅读本文件夹的 README.md、SKILL.md 和 references/USER_REQUIREMENTS.md，按其中最后确认的要求生成本次论文图片。使用 scripts/ 的 Python 模板和 style.json，先检查我的新数据字段、单位与统计层级，再替换示例数据。所有英文和数字使用 Times New Roman；数值为黑色；子图编号与下方横轴说明放在一起，例如 a) Token position；双纵轴的轴、刻度和标题为黑色，曲线保留彩色并用图例区分；图内不放总标题和长篇解释；图例、数值、误差线、色条和相邻子图不得重叠。雷达图优先分组、低透明度填充及五个有实际意义的维度，不得为了五边形伪造维度。完成后实际运行程序，查看导出的 PNG/PDF，修复排版，并交付代码、源数据、图片和独立图注。

## 一键复现

本机验证环境：Windows，`D:\Anaconda\python.exe`，Python 3.13.5，Matplotlib 3.10.0、NumPy 2.1.3、pandas 2.2.3、Pillow 11.1.0。Times New Roman 从 Windows 字体目录读取，包内不分发字体文件。

PowerShell，在任意工作目录运行：

```powershell
& 'D:\Anaconda\python.exe' 'C:\Users\ROG\Desktop\paper\codex\paper-figure-style-handoff-20261003\scripts\render_figures.py'
```

移动整个文件夹后，在包目录执行：

```powershell
python scripts/render_figures.py
```

默认生成 9 张示例图，每张有 600 dpi PNG、PDF、可编辑文字的 SVG，以及布局筛查 JSON。输出在 `outputs/`。同名运行会更新包内输出，不会写回原项目。已有示例图可直接查看。

只画横版雷达图，或只画机制图：

```powershell
python scripts/render_figures.py --figure radar --layout horizontal
python scripts/render_figures.py --figure mechanism
```

使用新数据目录，指定独立输出，并额外导出 TIFF：

```powershell
python scripts/render_figures.py --data-dir ./my_data --out-dir ./my_figures --figure radar --layout horizontal --formats png pdf svg tiff
```

新环境安装依赖的命令是 `python -m pip install -r requirements.txt`。本次未更改全局 Python 环境。字体不存在时程序明确报错，不会静默换字体；先在自己的系统合法安装 Times New Roman。SVG 交给他人打开时也需要此字体，PDF 可用于检查嵌入效果。

## 文件夹内容

```text
paper-figure-style-handoff-20261003/
  README.md                         总入口和复现命令
  SKILL.md                          可直接交给新窗口的 skill 入口
  NEW_WINDOW_PROMPT.md               可复用交接提示词
  style.json                        字体、色板、透明度和导出参数
  requirements.txt                  轻量绘图依赖
  references/                       原话、规范、数据说明、图形索引、检查表
  scripts/
    figure_style.py                 共用字体、编号、数值避让和导出
    render_figures.py               8 类图形，雷达横竖两版，共 9 张
    export_mechanism_cache.py       可选：从冻结模型提取机制图缓存
    bundle_sources.py              可选：从原工程收集历史文件
    verify_bundle.py               数据、布局、文件哈希和预览检查
    smoke_portability.py           搬到新目录后实际重画验证
    package_zip.py                 将交接包压缩，跳过 Python 缓存
  data/                             真实历史实验 CSV 和机制图缓存
  originals/
    scripts/                        4 个历史 Python 程序的原样快照
    figures/                        历史图 PNG 和部分 PDF
    notes/                          历史图注和论文相关说明
  outputs/                          可移植模板实际重画的图与布局报告
  qa/                               验证报告和总览图
  provenance.json                   历史来源路径、目标路径、SHA-256
  MANIFEST_SHA256.json               文件完整性清单
```

## 怎样复用到新论文

**先选择要复用的图形，随后更换数据与语义。** 配色、字体、线宽、留白、编号和避让方法可以迁移，电池的阶段定义、模型名称、指标值不能自动沿用到新论文。

- 改视觉参数：编辑 `style.json`。
- 改雷达图维度和分组：编辑 `scripts/render_figures.py` 中的 `RADAR_AXES`、`RADAR_GROUPS`，并同步 CSV。
- 改模型配色：在 `style.json` 的 `palette` 中使用与数据完全一致的模型名称。
- 改数据：保留数据契约中的字段名，或在读取处明确映射；用 `--data-dir` 和 `--out-dir` 保留原样例。
- 改版式：优先更改 `figsize`、GridSpec、边距及图例区域，调整后重新检查，不用缩小全部文字掩盖拥挤。
- 改机制图：替换缓存中的实际曲线、注意力、温度、熵、网格坐标及响应值。新模型需要从新模型导出，不得把本包响应面当成新模型证据。

默认模板采用最后确认的字体和底部编号规范；历史代码/图片原样存档，某些仍有早期字体、总标题、旧图形等。**原样存档不表示全部格式已经被用户认可。** 迁移时以 `references/USER_REQUIREMENTS.md` 的优先级为准。

便携雷达模板保留五边形、分组和彩色线条的视觉语言，并明确使用真实零点与跨面板统一标尺。历史雷达程序有径向偏移和分面标尺，见数据说明；需要核对当时原貌请看 `originals/figures/`，不要混称两者像素完全一致。

## 数据与解释

样例来自已落盘实验，最终模型为 `adaptive-attention-v1`。3 个 seed 为 42、7、123。这里没有开展新训练，没有合成更多 seed，也没有给新模型伪造参数搜索结果。

小提琴图的细点是“电芯 × seed”的误差，黑色菱形是 seed 级 pooled RMSE。24 个电芯误差条目不等于 24 个训练 seed。误差差值图的横向误差线继承各方法自己的 seed 标准差，不是配对差值的置信区间。

注意力响应面缓存保持历史计算定义，属于冻结模型的条件兼容性示意；颜色不是 RMSE，也不是归一化到总和为 1 的注意力概率。详见 `data/attention_mechanism_cache.json`。源电池阶段与测试集使用边界仅供历史复现。

## 验证与交接

```powershell
python scripts/verify_bundle.py
```

`qa/validation.json` 保存文件、数据及布局检查结果；`qa/gallery.png` 是新版模板的预览总览。程序筛查无法证明所有曲线和文字均无遮挡，因此需按 `references/QUALITY_CHECKLIST.md` 实际查看图片。替换数据、调整长标签、换字体或改变最终印刷尺寸后都应重查。

`qa/portability.json` 记录已在独立临时目录重画全部 9 张图通过；`qa/REVIEW.md` 记录本次人工检查和 PDF 字体核查。原对话的部分临时上传截图路径现已失效，未伪造这些文件；对应风格要求已写入文字，实际生成的历史图完整保留在 originals/figures/。

维护包内容之后，重新记录清单并打包：

```powershell
python scripts/verify_bundle.py --write-manifest
python scripts/package_zip.py
```

可选的 `bundle_sources.py` 和 `export_mechanism_cache.py` 用于追溯原工程，依赖原数据/模型且不属于日常画图入口。普通复现不需要执行它们。完整夹可整体拷贝；同级 ZIP 只是这一文件夹的便携副本。
