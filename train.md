# 双向扫描（水平扫描）+风格注入


> **Abstract:**
>1. Style-Adaptive LayerNorm (AdaIN 变体)
>
> 我们不再修改 SS2D 内部，而是修改 VSSBlock 中的 归一化层 (Norm)。
我们将普通的 LayerNorm 替换为 “风格自适应 LayerNorm”。


## 实验结果一
Method: HWV_V1.3, Epoch: 5500, FID: 12.596328735351562, KID: None, HWD: None, GS: None.
2026-01-12 HWV_V1.3 CVL_fid Epoch: 9500, Split: test, FID: 14.865280151367188, KID: None, HWD: None, GS: None.


## 实验结果二
