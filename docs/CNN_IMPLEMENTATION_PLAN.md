# CNN 图像分类实现与复现方案

## 目标与边界

使用 Animals-10 中的猫、狗、鸡、马、牛、羊图片完成六分类。产生式系统接受人工
符号特征，CNN 接受图像；二者类别名称一致，但输入空间和测试样本不同，因此报告
中不能把两个 Accuracy 简单解释为同一测试集上的直接胜负。

## 已实现结构

```text
src/cnn/
├─ config.py
├─ dataset.py
├─ model.py
├─ train.py
├─ evaluate.py
└─ pipeline.py
notebooks/03_cnn_training.ipynb
tests/test_cnn.py
checkpoints/cnn/
└─ best_resnet18.pt
outputs/cnn/
├─ history.csv
├─ test_metrics.csv
├─ per_class_metrics.csv
├─ learning_curves.png
└─ confusion_matrix.png
```

## 实现步骤

1. **数据层**：读取 `data/animals10/metadata.csv`；训练集使用随机裁剪、翻转等增强，
   验证集和测试集只做确定性缩放、中心裁剪及预训练权重要求的归一化。
2. **可复现性**：固定 Python、NumPy、PyTorch 和 DataLoader 随机种子；类别顺序复用
   `src.labels.LABELS_EN`；训练、验证、测试严格使用现有 `split` 字段。
3. **模型层**：优先采用 torchvision ResNet18，并显式使用 ImageNet 默认权重；将最后
   全连接层替换为六分类输出。先冻结骨干训练分类头，再用较小学习率微调整体网络。
4. **训练层**：使用交叉熵；针对类别不均衡只在训练集计算类别权重或加权采样；每个
   epoch 保存 train/validation loss、accuracy 和 macro F1，以 validation macro F1
   选择最佳权重，并加入早停机制。
5. **测试层**：训练和调参结束后只加载最佳权重评估一次 test 集，输出 Accuracy、
   Macro Precision/Recall/F1、Weighted F1、每类指标和行归一化混淆矩阵。
6. **报告层**：绘制 train/validation loss、accuracy 曲线以及测试混淆矩阵；与产生式
   方法统一展示 Accuracy、Macro F1 和每类指标，同时明确两者使用不同输入测试集。

## 验收条件

- 数据划分无交叉，类别索引在训练、验证、测试中一致；
- 默认运行路径不会在测试集上选模型或调参；
- 最佳权重、配置、训练历史和测试预测均能落盘复现；
- CPU 可以完成推理，CUDA 可用于训练；
- Notebook 从头运行时只调用 `src/cnn`，不含另一套重复训练实现；
- 自动测试至少覆盖数据读取、输出维度、单批次前向/反向和权重重载。

## 已完成实验

- 训练集 10,282 张、验证集 2,227 张、测试集 2,265 张；
- 2 个 epoch 冻结骨干训练分类头，6 个 epoch 解冻微调；
- 最佳模型为第 7 个 epoch，验证集 Macro F1 为 0.9740；
- 测试集 Accuracy 为 0.9695，Macro F1 为 0.9652；
- 训练历史、测试预测、逐类指标和图表均保存在 `outputs/cnn/`。
