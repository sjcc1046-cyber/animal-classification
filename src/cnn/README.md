# CNN 模块边界

此目录包含已经完成的 ResNet18 迁移学习图像分类方法。代码按职责拆分，避免把
可复用实现堆入 Notebook：

- `dataset.py`：读取 `metadata.csv`、图像变换、`Dataset` 和 `DataLoader`；
- `model.py`：构建六分类模型、加载 torchvision 预训练权重、冻结/解冻骨干；
- `train.py`：单个 epoch 的训练与验证、最佳权重保存、历史记录；
- `evaluate.py`：测试集预测、分类指标、混淆矩阵和结果导出；
- `config.py`：随机种子、图像尺寸和训练超参数；类别顺序复用 `src.labels`；
- `pipeline.py`：串联数据、训练、最佳权重加载、测试和图表输出。

`notebooks/03_cnn_training.ipynb` 只负责编排实验、展示输出和解释结果，不重复
上述模块的实现。最佳权重写入 `checkpoints/cnn/`，指标和图片写入 `outputs/cnn/`。
