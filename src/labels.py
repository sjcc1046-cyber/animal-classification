"""Shared class labels used by both animal-classification methods."""

LABELS_EN = ("cat", "dog", "chicken", "horse", "cow", "sheep")
LABELS_ZH = ("猫", "狗", "鸡", "马", "牛", "羊")
EN_TO_ZH = dict(zip(LABELS_EN, LABELS_ZH, strict=True))
ZH_TO_EN = dict(zip(LABELS_ZH, LABELS_EN, strict=True))

