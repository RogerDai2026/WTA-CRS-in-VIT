# Exploring the Efficacy of the WTA-CRS Method on Vision Transformers

**Author:** Roger Dai  
**Date:** September 5, 2024

## Abstract

With the rapid growth in model size, fine-tuning large pre-trained models has become increasingly difficult due to their extensive memory usage. Following the paper *"Winner-Take-All Column Row Sampling for Memory Efficient Adaptation of Language Models"*, this project further investigates the application of the Winner-Take-All Column Row Sampling (WTA-CRS) method on the Vision Transformer (ViT) model for image classification tasks. The project focuses on its impact on memory efficiency and classification accuracy using popular datasets like CIFAR-10 and CIFAR-100.

## Introduction

In this project, we applied the Winner-Take-All Column Row Sampling (WTA-CRS) method to the Vision Transformer (ViT) model for image classification tasks. This work builds on prior research where the WTA-CRS method was successfully tested on transformer-based models like T5 and BERT. The main goal was to extend the application of WTA-CRS to Vision Transformers and analyze its impact on memory efficiency and classification accuracy using CIFAR-10 and CIFAR-100 datasets.

## Methodology

The WTA-CRS method was implemented on Vision Transformers, utilizing a single A40 GPU with the CIFAR-10 and CIFAR-100 datasets. We tested two different sampling ratios, 0.5 and 0.3, to explore the trade-offs between memory usage and model accuracy, comparing the results to traditional GEMM (general matrix multiplication).

We modified the calculation inside the multi-head attention mechanism by replacing GEMM with the WTA-CRS method for three-dimensional tensors (query, key, and value vectors). We also replaced GEMM with CRS in the matrix multiplication used to calculate attention scores and context layers for four-dimensional tensors. As in the original WTA-CRS paper, we applied the method only to the backward pass to ensure unbiased gradients. The experiments were conducted with sampling ratios of 1 (GEMM), 0.5, and 0.3, using a batch size of 64 over 8 epochs on both datasets.

## Results and Conclusion

The experimental results showed that using the WTA-CRS method with a sampling ratio of 0.5 resulted in an accuracy drop of less than 0.4% on both datasets, while achieving at least a 1.2x memory saving. However, a sampling ratio of 0.3 led to a more noticeable accuracy drop, especially on the CIFAR-100 dataset, while still reducing peak memory usage by 1.5x.

Compared to traditional transformers, the WTA-CRS method was less effective on Vision Transformers, as maintaining an acceptable accuracy drop required higher sampling ratios, which diminished memory savings. However, given the limited datasets and parameters tested, better results might be achievable with further fine-tuning.

### Accuracy Drop Across Sampling Ratios

| Sampling Ratio | CIFAR-10 | CIFAR-100 Fine Label | CIFAR-100 Coarse Label |
|----------------|----------|----------------------|------------------------|
| 0.3            | 0.9760   | 0.7242               | 0.8249                 |
| 0.5            | 0.9891 (0.1%) | 0.9122 (0.4%)   | 0.9613 (0.3%)          |
| 1 (GEMM)       | 0.9901   | 0.9162               | 0.9641                 |

### Peak Memory Usage Across Sampling Ratios

| Sampling Ratio | CIFAR-10           | CIFAR-100 Fine Label | CIFAR-100 Coarse Label |
|----------------|--------------------|----------------------|------------------------|
| 0.3            | 13.16 GB (1.3x)    | 10.97 GB (1.5x)      | 11.02 GB (1.5x)        |
| 0.5            | 13.90 GB (1.2x)    | 14.11 GB (1.2x)      | 14.11 GB (1.2x)        |
| 1 (GEMM)       | 16.82 GB           | 16.82 GB             | 16.82 GB               |
