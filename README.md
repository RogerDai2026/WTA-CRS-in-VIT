\documentclass[12pt]{article}

\usepackage{amsmath}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{hyperref}
\usepackage{geometry}
\geometry{margin=1in}

\title{Exploring the Efficacy of the CRS Method on Vision Transformers}
\author{Roger Dai}
\date{\today}

\begin{document}

\maketitle

\begin{abstract}
With the rapid growth in model size, fine-tuning large pre-trained"Winner-Take-All Column Row Sampling for Memory Efficient Adaptation of Language Model",  models has become increasingly difficult due to their extensive memory usage. Following the paper "Winner-Take-All Column Row Sampling for Memory Efficient Adaptation of Language Model", this project further investigates the application of the Column Row Sampling (CRS) method on the Vision Transformer (ViT) model for image classification tasks, focusing on its impact on memory efficiency and classification accuracy using popular datasets like CIFAR-10 and CIFAR-100.
\end{abstract}

\section{Introduction}
For this project, we focused on applying the Column Row Sampling (CRS) method to the Vision Transformer (ViT) model for image classification tasks. This work builds on prior research where the CRS method was successfully developed and tested on transformer-based models such as T5 and BERT. The main goal of this project was to extend the application of CRS to Vision Transformers, specifically testing its impact on memory efficiency and classification accuracy using popular datasets like CIFAR-10 and CIFAR-100.

\section{Methodology}
We implemented the CRS method on the Vision Transformers using the CIFAR-10 and CIFAR-100 datasets on a single A40 GPU. We selected two different sampling ratios, 0.5 and 0.3, to test the trade-offs between memory usage and model accuracy, and then compared the results to normal GEMM (general matrix multiplication).

Specifically, we modified the calculation inside the multi-head attention mechanism by replacing GEMM in the calculation of the query, key, and value (Q, K, V) vectors with the CRS method for three-dimensional tensors. The matrix multiplication used in calculating attention scores and context layers was also replaced with the CRS method for four-dimensional tensors. Similar to the original CRS paper, we only applied this method to the backward pass to ensure the gradients remain unbiased. We conducted experiments with sampling ratios of 1 (GEMM), 0.5, and 0.3, with a batch size of 64, over 8 epochs on both datasets.

\begin{figure*}
\begin{center}
\includegraphics[width=0.85\linewidth]{Screenshot 2024-08-15 at 6.05.12 PM.png}
\end{center}
\caption{Vision Transformer architecture}
\end{figure*}

\section{Results and Conclusion}
The experimental results revealed that applying the CRS method with a sampling ratio of 0.5 resulted in an accuracy drop of less than 0.4\% on both datasets. Furthermore, this setup achieved at least a 1.2x memory saving. On the other hand, while lowering peak memory usage by approximately 1.5x, a sampling ratio of 0.3 led to a noticeable accuracy drop, particularly in the CIFAR-100 dataset. Overall, we observe that compared to its performance on traditional transformers, the CRS method wasn’t that effective on the Vision Transformers, as we need higher sampling ratios to keep the accuracy drop within a reasonable range, which results in memory savings being less effective. However, as the tested datasets and parameters were limited, we might encounter better results during future fine-tuning on Vision Transformers.

\begin{table}[h!]
\centering
\caption{Accuracy Drop Across Sampling Ratios}
\begin{tabular}{|c|c|c|c|}
\hline
\textbf{Sampling Ratio} & \textbf{CIFAR-10} & \textbf{CIFAR-100 Fine Label} & \textbf{CIFAR-100 Coarse Label} \\ \hline
0.3 & 0.9760 & 0.7242 & 0.8249 \\ \hline
0.5 & 0.9891 (0.1\%) & 0.9122 (0.4\%) & 0.9613 (0.3\%) \\ \hline
1 (GEMM) & 0.9901 & 0.9162 & 0.9641 \\ \hline
\end{tabular}
\end{table}
\begin{table}[h!]
\centering
\caption{Peak Memory Usage Across Sampling Ratios}
\begin{tabular}{|c|c|c|c|}
\hline
\textbf{Sampling Ratio} & \textbf{CIFAR-10} & \textbf{CIFAR-100 Fine Label} & \textbf{CIFAR-100 Coarse Label} \\ \hline
0.3 & 13.16 GB (1.3x) & 10.97 GB (1.5x) & 11.02 GB (1.5x) \\ \hline
0.5 & 13.90 GB (1.2x) & 14.11 GB (1.2x) & 14.11 GB (1.2x) \\ \hline
1 (GEMM) & 16.82 GB & 16.82 GB & 16.82 GB \\ \hline
\end{tabular}
\end{table}


\end{document}


\documentclass[12pt]{article}

\usepackage{amsmath}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{hyperref}
\usepackage{geometry}
\geometry{margin=1in}

\title{Exploring the Efficacy of the Column-row Sampling on Vision Transformers}
\author{Roger Dai}
\date{\today}

\begin{document}

\maketitle

\begin{abstract}
With the rapid growth in model size, fine-tuning large pre-trained models has become increasingly difficult due to their extensive memory usage. This project investigates the application of the Winner-Take-All (WTA) method on the Vision Transformer (ViT) model for image classification tasks, focusing on its impact on memory efficiency and classification accuracy using popular datasets like CIFAR-10 and CIFAR-100.
\end{abstract}

\section{Introduction}
For this project, we focused on applying the Winner-Take-All (WTA) method to the Vision Transformer (ViT) model for image classification tasks. This work builds on prior research where the WTA method was successfully developed and tested on transformer-based models such as T5 and BERT. The main goal of this project was to extend the application of WTA to ViT models, specifically testing its impact on memory efficiency and classification accuracy using popular datasets like CIFAR-10 and CIFAR-100.

\section{Methodology}
We implemented the WTA method on the ViTForImageClassification model using the CIFAR-10 and CIFAR-100 datasets on a single A40 GPU. We selected two different sampling ratios, 0.5 and 0.3, to test the trade-offs between memory usage and model accuracy, and then compared the results to normal GEMM (general matrix multiplication).

Specifically, we modified the calculation inside the multi-head attention mechanism by replacing GEMM in the calculation of the query, key, and value (Q, K, V) vectors with the WTA method for three-dimensional tensors. The matrix multiplication used in calculating attention scores and context layers was also replaced with the WTA method for four-dimensional tensors. Similar to the original WTA paper, we only applied this method to the backward pass to ensure the gradients remain unbiased. We conducted experiments with sampling ratios of 1 (GEMM), 0.5, and 0.3, with a batch size of 64, over 8 epochs on both datasets.

\begin{figure*}
\begin{center}
\includegraphics[width=0.85\linewidth]{Screenshot 2024-08-15 at 6.05.12 PM.png}
\end{center}
\caption{ViT Model detail}
\end{figure*}
\section{Results and Conclusion}
The experimental results revealed that applying the WTA method with a sampling ratio of 0.5 resulted in an accuracy drop of less than 0.4\% on both datasets. Furthermore, this setup achieved at least a 1.2x memory saving. On the other hand, while lowering peak memory usage by approximately 1.5x, a sampling ratio of 0.3 led to a noticeable accuracy drop, particularly in the CIFAR-100 dataset. Overall, we observe that compared to its performance on traditional transformers, the WTA method wasn’t as effective on the ViT model. Higher sampling ratios were required to keep the accuracy drop within a reasonable range, and the memory savings did not meet expectations as they did with traditional transformers. However, as the tested datasets and parameters were limited, we might encounter better results during later fine-tuning on ViT models.

\begin{table}[h!]
\centering
\caption{Accuracy Drop Across Sampling Ratios}
\begin{tabular}{|c|c|c|c|}
\hline
\textbf{Sampling Ratio} & \textbf{CIFAR-10} & \textbf{CIFAR-100 Fine Label} & \textbf{CIFAR-100 Coarse Label} \\ \hline
0.3 & 0.9760 & 0.7242 & 0.8249 \\ \hline
0.5 & 0.9891 (0.1\%) & 0.9122 (0.4\%) & 0.9613 (0.3\%) \\ \hline
1 (GEMM) & 0.9901 & 0.9162 & 0.9641 \\ \hline
\end{tabular}
\end{table}
\begin{table}[h!]
\centering
\caption{Peak Memory Usage Across Sampling Ratios}
\begin{tabular}{|c|c|c|c|}
\hline
\textbf{Sampling Ratio} & \textbf{CIFAR-10} & \textbf{CIFAR-100 Fine Label} & \textbf{CIFAR-100 Coarse Label} \\ \hline
0.3 & 13.16 GB (1.3x) & 10.97 GB (1.5x) & 11.02 GB (1.5x) \\ \hline
0.5 & 13.90 GB (1.2x) & 14.11 GB (1.2x) & 14.11 GB (1.2x) \\ \hline
1 (GEMM) & 16.82 GB & 16.82 GB & 16.82 GB \\ \hline
\end{tabular}
\end{table}
\end{document}
