Exploring the Efficacy of the CRS Method on Vision Transformers
Roger Dai August 15, 2024
Abstract
With the rapid growth in model size, fine-tuning large pre-trained”Winner-Take-All Column Row Sampling for Memory Efficient Adaptation of Language Model”, models has become increasingly difficult due to their extensive memory usage. Following the paper ”Winner-Take-All Column Row Sampling for Memory Efficient Adaptation of Language Model”, this project further investigates the application of the Column Row Sampling (CRS) method on the Vision Transformer (ViT) model for image classifi- cation tasks, focusing on its impact on memory efficiency and classification accuracy using popular datasets like CIFAR-10 and CIFAR-100.
1 Introduction
For this project, we focused on applying the Column Row Sampling (CRS) method to the Vision Transformer (ViT) model for image classification tasks. This work builds on prior research where the CRS method was successfully developed and tested on transformer-based models such as T5 and BERT. The main goal of this project was to extend the application of CRS to Vision Transformers, specifically testing its impact on memory efficiency and classification accuracy using popular datasets like CIFAR-10 and CIFAR-100.
2 Methodology
We implemented the CRS method on the Vision Transformers using the CIFAR-10 and CIFAR-100 datasets on a single A40 GPU. We selected two different sampling ratios, 0.5 and 0.3, to test the trade-offs between memory usage and model accuracy, and then compared the results to normal GEMM (general matrix multiplication).
Specifically, we modified the calculation inside the multi-head attention mechanism by re- placing GEMM in the calculation of the query, key, and value (Q, K, V) vectors with the CRS method for three-dimensional tensors. The matrix multiplication used in calculating atten- tion scores and context layers was also replaced with the CRS method for four-dimensional tensors. Similar to the original CRS paper, we only applied this method to the backward pass to ensure the gradients remain unbiased. We conducted experiments with sampling ratios of 1 (GEMM), 0.5, and 0.3, with a batch size of 64, over 8 epochs on both datasets.
1
 Figure 1: Vision Transformer architecture
3 Results and Conclusion
The experimental results revealed that applying the CRS method with a sampling ratio of 0.5 resulted in an accuracy drop of less than 0.4% on both datasets. Furthermore, this setup achieved at least a 1.2x memory saving. On the other hand, while lowering peak memory usage by approximately 1.5x, a sampling ratio of 0.3 led to a noticeable accuracy drop, par- ticularly in the CIFAR-100 dataset. Overall, we observe that compared to its performance on traditional transformers, the CRS method wasn’t that effective on the Vision Transform- ers, as we need higher sampling ratios to keep the accuracy drop within a reasonable range, which results in memory savings being less effective. However, as the tested datasets and pa- rameters were limited, we might encounter better results during future fine-tuning on Vision Transformers.
Sampling Ratio
0.3
0.5
1 (GEMM)
CIFAR-10
0.9760 0.9891 (0.1%) 0.9901
CIFAR-100 Fine Label
0.7242 0.9122 (0.4%) 0.9162
CIFAR-100 Coarse Label
Table 1: Accuracy Drop Across Sampling Ratios
  0.8249 0.9613 (0.3%) 0.9641
   Table 2: Peak Memory Usage Across Sampling Ratios
 Sampling Ratio
0.3
0.5
1 (GEMM)
CIFAR-10
13.16 GB (1.3x) 13.90 GB (1.2x) 16.82 GB
CIFAR-100 Fine Label
10.97 GB (1.5x) 14.11 GB (1.2x) 16.82 GB
CIFAR-100 Coarse Label
11.02 GB (1.5x) 14.11 GB (1.2x) 16.82 GB
    2
