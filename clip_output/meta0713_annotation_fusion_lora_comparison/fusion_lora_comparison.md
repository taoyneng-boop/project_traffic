# Multi-Annotation Fusion + CLIP ViT-B/32 LoRA Comparison

All runs use rank=8, alpha=16, lr=1e-4, seed=42, epoch=12.

| Method | Acc | R@1 | R@5 | R@10 | Mean Rank | Loss |
|---|---:|---:|---:|---:|---:|---:|
| original_fixed120 | 42.39% | 11.96% | 38.04% | 54.35% | 13.34 | 1.1604 |
| fusion_concat | 40.22% | 21.74% | 51.09% | 69.57% | 9.74 | 0.8743 |
| fusion_structured | 40.22% | 18.48% | 50.00% | 69.57% | 9.21 | 0.8689 |
| fusion_normalized | 34.78% | 35.87% | 77.17% | 90.22% | 4.26 | 0.7222 |