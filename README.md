# 六类动物识别课程作业

本项目比较两种动物识别方法：

1. 基于事实、IF–THEN 规则和正向链推理的产生式系统；
2. 基于 ImageNet 预训练 ResNet18 的 PyTorch 图像分类模型。

统一的六个目标类别是：猫、狗、鸡、马、牛、羊。英文标签与
`Rapidata/Animals-10` 数据集一致。

## 当前完成度

- 已完成：产生式系统、规则库、命令行演示、自动测试、符号压力测试及量化图表；
- 已完成：Animals-10 六类图片下载、固定的训练/验证/测试划分和数据预览；
- 已完成：ResNet18 ImageNet 预训练权重加载、数据增强、类别加权损失、两阶段迁移
  学习、验证集选模、测试评价、学习曲线和混淆矩阵；
- 已完成：产生式系统与 CNN 的统一指标汇总。两者类别相同但输入和测试集不同，
  对比用于方法分析，不解释为同一批样本上的严格竞赛。

## 项目结构

```text
animal-classification/
├─ notebooks/
│  ├─ 01_production_system.ipynb     # 产生式推理、测试和量化评价
│  ├─ 02_dataset_check.ipynb         # CUDA、数据统计和图片预览
│  └─ 03_cnn_training.ipynb          # ResNet18 训练、测试与方法对比
├─ src/production_system/
│  ├─ engine.py                      # 正向链推理引擎
│  ├─ knowledge_base.py              # 特征、规则和六类标准事实
│  └─ cli.py                         # 独立的命令行演示入口
├─ src/labels.py                      # 两种方法共享的六类标签与中英文映射
├─ src/cnn/                           # 数据、模型、训练、评价和实验流水线
├─ src/comparison.py                  # 两种方法的统一指标表和图
├─ scripts/                           # 数据下载与 CNN 训练入口
├─ tests/                             # 产生式和 CNN 自动测试
├─ docs/CNN_IMPLEMENTATION_PLAN.md    # CNN 实现、复现步骤和验收条件
├─ checkpoints/cnn/                   # 最佳 ResNet18 权重（不提交 Git）
├─ outputs/                           # 两种方法的指标与图表（不提交 Git）
└─ requirements.txt                   # Python 依赖
```

`src/production_system` 不是 Notebook 的重复代码：Notebook、CLI 和测试都导入同一套
推理实现，从而避免在多个位置复制规则。Notebook 中保留的是实验编排、表格和绘图代码。

## 运行检查 Notebook

`notebooks/01_production_system.ipynb` 只包含产生式推理、异常与冲突案例、自动
测试和符号压力测试。`notebooks/02_dataset_check.ipynb` 只包含 PyTorch/CUDA 实算、
数据集统计和图片预览；`notebooks/03_cnn_training.ipynb` 展示 ResNet18 训练、测试
及方法对比。在 VS Code 中选择项目 `.venv` 内核后，可以分别重新运行。

## 运行 CNN

在项目根目录执行完整训练与测试：

```powershell
.\.venv\Scripts\python.exe scripts\train_cnn.py --force
```

不带 `--force` 时会复用已有最佳权重并重新评价。训练采用 ImageNet-1K 预训练
ResNet18：先冻结卷积骨干训练分类头 2 个 epoch，再解冻全网络微调最多 6 个 epoch，
以验证集 Macro F1 保存最佳权重。当前最佳模型在 2,265 张测试图片上的 Accuracy 为
0.9695，Macro F1 为 0.9652。

## 运行产生式系统

在项目根目录执行：

```powershell
.\.venv\Scripts\python.exe -m src.production_system.cli --example 猫
```

不带参数时进入交互式特征选择：

```powershell
.\.venv\Scripts\python.exe -m src.production_system.cli
```

运行自动测试：

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

## 数据集

数据来自 Hugging Face 上经过人工复核的 `Rapidata/Animals-10`，数据集卡标注
GPL-2.0。下载脚本只导出猫、狗、鸡、马、牛、羊，并用稳定哈希生成约
70%/15%/15% 的训练、验证和测试清单：

```powershell
.\.venv\Scripts\python.exe scripts\download_animals10.py
```

图片保存在 `data/animals10/images`，划分及标签保存在
`data/animals10/metadata.csv`。一张图片只保存一次，划分之间不会复制。

当前六类共 14,774 张图片。类别数量并不均衡，CNN 使用训练集样本数计算交叉
熵类别权重；验证集和测试集保持原始分布。

产生式系统是在六个目标类别构成的封闭世界中推理。例如“家养、有鸡冠的
鸟类是鸡”并不是对所有鸟类的完整生物学定义，而是课程实验知识库中的识别
规则。报告中应明确说明这个适用范围。

Animals-10 的样本字段只有图片和动物类别，不包含“有毛发”“有蹄”“反刍”等
属性。本项目的产生式规则来自常识性动物知识和经典专家系统规则结构，由人工
针对六个目标类别整理；规则不是从 Animals-10 训练、统计或自动生成的。该数据集
在产生式方法中只用于统一最终类别名称。

## 参考资料

- PyTorch Transfer Learning Tutorial:
  https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
- Rapidata Animals-10:
  https://huggingface.co/datasets/Rapidata/Animals-10
- 产生式动物识别参考：
  https://github.com/xiaonan-x/animal-recognition-system
- 文本规则库与正向推理参考：
  https://github.com/yixiangxi/Generative_Animal_IdentificationExpert_System

本项目的推理引擎、知识库和测试为独立实现，参考项目仅用于理解经典动物识别
规则及正向链工作流程。
