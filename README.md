# Exploring the Efficacy of the WTA-CRS Method on Vision Transformers

**Author:** Roger Dai  
**Date:** September 5, 2024

---

## Abstract

With the rapid growth in model size, fine-tuning large pre-trained models has become increasingly difficult due to their extensive memory usage.

Following the paper *“Winner-Take-All Column Row Sampling for Memory Efficient Adaptation of Language Models”*, this project investigates the application of the Winner-Take-All Column Row Sampling (WTA-CRS) method on the Vision Transformer (ViT) for image classification.

We measure its impact on memory efficiency and classification accuracy using CIFAR-10 and CIFAR-100 datasets.

---

## Introduction

Large pre-trained Transformer models require substantial GPU memory for both forward and backward passes.

The WTA-CRS method replaces costly GEMM operations in the backward pass with a sampling strategy that reduces memory spikes.

While it proved effective in large language models (T5, BERT), its efficacy on Vision Transformers remains unexplored.

This work adapts WTA-CRS to ViT’s attention mechanism and evaluates trade-offs between memory usage and classification accuracy.

---

## Methodology

### 1. Base ViT Implementation  

- We start from a standard ViT-Base (12 layers, hidden size 768) implementation.  
- Multi-head attention blocks use GEMM for all Q, K, V projections and for scoring/softmax/context matmuls.

### 2. Integrating WTA-CRS  

- **Scope:** only in the **backward pass**, so that gradient estimates remain unbiased.  
- **Targets:**  
  - Q/K/V projection weight gradients (`dWq`, `dWk`, `dWv`)  
  - Attention score gradient matmuls (the `d(softmax(QKᵀ))` step)  
  - Context gradient matmuls (`d(attn · V)`).

#### 2.1. Column-Row Sampling  

- For a 3D tensor multiplication \(A \in \mathbb{R}^{B\times D\times M}\) × \(W \in \mathbb{R}^{D\times N}\), we sample a subset of rows/columns:  
  1. Compute forward pass normally to get outputs.  
  2. In backward, pick top-**k** rows/columns by magnitude of activations (the “winner” indices).  
  3. Only accumulate weight gradients on those slices, scaling by \(\frac{1}{\rho}\) where \(\rho\) is the sampling ratio.

#### 2.2. Implementation Details  

- We patch the PyTorch `Functional.conv1d`/`matmul` gradient kernels via a custom autograd hook:  
  ```python
  def wta_crs_grad_hook(grad):
      # grad: upstream gradient of shape [..., M, N]
      topk_idx = torch.topk(torch.abs(grad), k=int(ratio*M), dim=-2).indices
      mask = torch.zeros_like(grad)
      mask[..., topk_idx, :] = 1.0
      return grad * mask / ratio

- **Model & Datasets**  
  - **Backbone:** Vision Transformer (ViT-Base)  
  - **Datasets:** CIFAR-10, CIFAR-100 (fine & coarse labels)  
  - **Hardware:** Single NVIDIA A40 GPU  

- **WTA-CRS Integration**  
  1. In the **backward pass** only, replace standard matrix‐multiplication in multi-head attention (Q, K, V projections and attention score/context matmuls) with WTA-CRS.  
  2. Test three sampling ratios:  
     - **1.0 (GEMM baseline)**  
     - **0.5**  
     - **0.3**  

- **Training Setup**  
  - Batch size: 64  
  - Epochs: 8  
  - Optimizer: Adam (lr=3e-4)  
  - Data augmentation: standard CIFAR flips/crops  

- **Metrics**  
  - **Classification Accuracy**  
  - **Peak GPU Memory Usage** (measured via `torch.cuda.max_memory_allocated()`)

---

## Results & Conclusion

The WTA-CRS method at a 0.5 sampling ratio yields under 0.4 % drop in accuracy on both CIFAR-10 and CIFAR-100, while reducing peak memory by ≈ 1.2×. 
A more aggressive 0.3 ratio further lowers memory (up to 1.5×) but incurs a larger accuracy penalty, particularly on CIFAR-100. Overall, WTA-CRS offers
a meaningful memory-accuracy trade-off for ViT, though less dramatic than in language models. Further hyperparameter tuning or hybrid sampling 
schedules could narrow the gap.

---

## Accuracy Drop Across Sampling Ratios

| Sampling Ratio | CIFAR-10 Accuracy | CIFAR-100 Fine Label | CIFAR-100 Coarse Label |
|---------------:|------------------:|---------------------:|-----------------------:|
| **0.3**        | 97.60 %           | 72.42 %              | 82.49 %                |
| **0.5**        | 98.91 % (+0.1 %)  | 91.22 % (+0.4 %)     | 96.13 % (+0.3 %)       |
| **1.0 (GEMM)** | 99.01 %           | 91.62 %              | 96.41 %                |

---

## Peak Memory Usage Across Sampling Ratios

| Sampling Ratio | CIFAR-10 (GB)     | CIFAR-100 Fine (GB)  | CIFAR-100 Coarse (GB)  |
|---------------:|------------------:|---------------------:|-----------------------:|
| **0.3**        | 13.16 (1.3× ↓)    | 10.97 (1.5× ↓)       | 11.02 (1.5× ↓)         |
| **0.5**        | 13.90 (1.2× ↓)    | 14.11 (1.2× ↓)       | 14.11 (1.2× ↓)         |
| **1.0 (GEMM)** | 16.82              | 16.82                | 16.82                  |

---

## Future Work

- Explore **adaptive** or **layer-wise** sampling schedules.  
- Combine WTA-CRS with **quantization** or **pruning** for further memory reduction.  
- Evaluate on larger vision benchmarks (ImageNet, video Transformers).  

---

## License

This work is released under the **Apache 2.0** license.

---

## Contact

**Roger Dai**  ([qd8@rice.edu](mailto:qd8@rice.edu))
