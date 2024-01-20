# Statistically Optimal $K$-means Clustering via Nonnegative Low-rank Semidefinite Programming.
This repository contains Matlab codes and clustering benchmark dataset. 

The codes are sourced from the following paper:

- Yubo Zhuang, Xiaohui Chen, Yun Yang, Richard Y. Zhang. *Statistically Optimal K-means Clustering via Nonnegative Low-rank Semidefinite Programming.*
   
  
The benchmark clustering dataset is sourced from the following resource:

- Levine et al. (2015). *Data-Driven Phenotypic Dissection of AML Reveals Progenitor-like Cells that Correlate with Prognosis.* Cell, 162, pp. 184-197.\
  https://github.com/lmweber/benchmark-data-Levine-32-dim


# Background
Yubo Zhuang, Xiaohui Chen, Yun Yang, Richard Y. Zhang. *Statistically Optimal K-means Clustering via Nonnegative Low-rank Semidefinite Programming.*
## The abstract of the paper
$K$-means clustering is a widely used machine learning method for identifying patterns in large datasets. Semidefinite programming (SDP) relaxations have recently been proposed for solving  the $K$-means optimization problem that enjoy strong statistical optimality guarantees, but the prohibitive cost of implementing an SDP solver renders these guarantees inaccessible to practical datasets. By contrast, nonnegative matrix factorization (NMF) is a simple clustering algorithm that is widely used by machine learning practitioners, but without a solid statistical underpinning nor rigorous guarantees. In this paper, we describe an NMF-like algorithm that works by solving a *nonnegative* low-rank restriction of the SDP relaxed $K$-means formulation using a nonconvex Burer-Monteiro factorization approach. The resulting algorithm is just as simple and scalable as state-of-the-art NMF algorithms, while also enjoying the same strong statistical optimality guarantees as the SDP. In our experiments, we observe that our algorithm achieves substantially smaller mis-clustering errors compared to the existing state-of-the-art.

# This repository
## Purpose
The codes show the comparision of mis-clustering errors on CyTOF dataset across five different clustering methods including BM (we proposed), SDP, KM (K-means++), NNMF (non-negative matrix factorization) and SC (spectral clustering from Statistics and Machine Learning Toolbox).
## Steps
 - First you will need to install the optimization toolbox SDPNAL+ from the following website:\
    https://blog.nus.edu.sg/mattohkc/softwares/sdpnalplus/ \
   Please download and install SDPNAL+ correctly following the instructions in the website. Then put all our files in the path of SDPNAL+ folder. (Or you may just download the file BM_kmeans.zip, which contains the toolbox.)
 - Next run SDPNALplus_Demo.m in toolbox SDPNAL+ to properly install it.
 - Finally we could run main.m to get the mis-clustering error for different methods.
## Contents
The files in this repository are:

- main.m: The main code for comparisons of different clustering methods for benchmark dataset.
- BM_cluster.m: The code for clustering method based on BM (our proposed method).
- kmeans_sdp.m: The code for clustering method based on SDP.
- NNMF_cluster.m: The code for clustering method based on non-negative matrix factorization.
- kmeansplus.m: The code for kmeans++ method.
- err_rate.m: The function to get the mis-clustering error for two labels.
- Levine_32dim.fcs: The dataset from Levine et al. (2015).
- fca_readfcs.m: The function to read Levine_32dim.fcs data.
- BM_kmeans.zip: The file contains all the files above and the toolbox SDPNAL+.
## Remarks
One could choose download BM_kmeans.zip if one would not prefer to download and install the optimization toolbox SDPNAL+ through above steps.
