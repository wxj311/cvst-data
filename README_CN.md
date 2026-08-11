# CVST-SII 完整 Jupyter/Python 分析代码

本项目针对您压缩包中实际存在的图表清单重建：

- 正文图：Fig1-Fig6（6张）
- 补充图：eFig01-eFig60（60张）
- 正文表：Table1-Table5（5个）
- 补充表：TableS01-TableS20（20个）

合计：66张图 + 25个表。所有结果均从 `data.csv` 重新计算，不直接读取压缩包内已经生成的 CSV 数值作为分析结果。

## Jupyter 推荐用法

1. 将整个文件夹复制到本地。
2. 建议创建 Python 3.11/3.12 环境。
3. 安装依赖：`pip install -r requirements.txt`
4. 启动 Jupyter：`jupyter lab`
5. 打开 `CVST_complete_run.ipynb`
6. 依次运行所有单元格。

输出位于：

- `outputs/figures/`：每张图同时保存 PDF + PNG
- `outputs/tables/`：每个表同时保存 CSV + XLSX
- `outputs/metadata/output_manifest.csv`：逐项检查是否全部生成

## 正式参数

代码默认：

- 10-seed LASSO：0, 7, 13, 21, 42, 99, 123, 777, 2024, 314
- C 网格：10^-4 到 10^2，共100个
- LASSO 内层：10-fold AUROC
- 共识阈值：>=8/10 seeds
- 最终三变量模型：SII + platelet + intracerebral haemorrhage，全部标准化
- Bootstrap：1000次分层重抽样
- 固定特征机器学习：3×10 repeated stratified CV
- LR/RF/GB/SVM-R/SVM-L 参数按材料设置
- RF/GB解释：不依赖 SHAP 包，使用树路径（Saabas-style）贡献；LR使用精确线性贡献
- LIME：500 Gaussian perturbations + Gaussian kernel + weighted local linear surrogate

## 快速测试

若只是检查本机是否可跑，可把 `cvst_complete_analysis.py` 中 `FAST_MODE = False` 改为 `True`。正式分析请保持 `False`。

## 关于材料版本

您压缩包中的“数据图表及文章规划”实际包含 eFig01-eFig60 和 TableS01-S20；而后续稿件文字中存在另一套补充材料编号。因此本项目以压缩包中真实文件清单作为“不漏图表”的执行清单，同时统计模型遵循稿件方法学：P<0.10预筛、10-seed LASSO共识、重复交叉验证、校准、DCA、SII四分位、RCS、模型归因和LIME。
