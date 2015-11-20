## Classification and Regression 分类与回归

* Generalized Linear Models 泛化线性模型
    + Linear Regression 线性回归
    + Logistic Regression 逻辑回归
    + SVM 支持向量机
        - Linear Kernel 线性核
        - Gaussian Kernel 高斯核

* Neural Network (NN) 神经网络

* Bayesian
    + Naive Bayes (NB) 朴素贝叶斯

* Decision Trees 决策树
    + ID3
    + C4.5
    + Classification and Regression Tree (CART)

* Ensembles of Trees 
    + Random Forests 随机森林
    + Gradient-Boosted Trees

* Isotonic regression

## Clustering 聚类

* K-means K-均值

* Gaussian Mixture Model 混合高斯模型 (GMM)

* power iteration clustering (PIC)

## Dimensionality Reduction 降维

* Principal Component Analysis 主成分分析 (PCA)

## Frequent Pattern Mining 频繁模式挖掘

* Association Rules

* FP-growth

* PrefixSpan

## Collaborative Filtering 协同过滤

* Alternating Least Squares (ALS)

## Topic Models [主题模型](http://blog.csdn.net/hxxiaopei/article/details/7617838)
* Latent Semantic Indexing (LSI)

* Probability Latent Semantic Indexing ([pLSI](http://www.52nlp.cn/%E6%A6%82%E7%8E%87%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B%E5%8F%8A%E5%85%B6%E5%8F%98%E5%BD%A2%E7%B3%BB%E5%88%971-plsa%E5%8F%8Aem%E7%AE%97%E6%B3%95)) [[SIGIR 1999](http://dl.acm.org/citation.cfm?id=312649)]

* Unigram model

* Mixture of unigrams model

* Latent Dirichlet Allocation ([LDA](http://blog.csdn.net/v_july_v/article/details/41209515)) [[JMLR 2003](http://dl.acm.org/citation.cfm?id=944919.944937)]

## Sequence Labeling [序列标注](http://blog.csdn.net/caohao2008/article/details/4242308) 

* Hidden Markov Model 隐马尔科夫模型 ([HMM](http://www.52nlp.cn/hmm-learn-best-practices-one-introduction))
    + Evaluation 评估: Forward algorithm [前向算法](http://www.cnblogs.com/tornadomeet/archive/2012/03/24/2415583.html)
    + Decoding 解码: Viterbi algorithm [维特比算法](http://www.cnblogs.com/tornadomeet/archive/2012/03/24/2415889.htm)
    + Learning 学习: Forward-backward algorithm 前向-后向算法
    
* Maximum Entropy Markov Model 最大熵马尔科夫模型 (MEMM)
    + Label Bias Problem [标注偏置问题](http://blog.csdn.net/zhoubl668/article/details/7787690)

* Markov Random Field 马尔科夫随机场 (MRF)

* Conditional Random Field 条件随机场 (CRF)

## Linear Algebra 线性代数

* Eigenvalue Decomposition 特征值分解

* Singular Value Decomposition 奇异值分解 (SVD)

* Low Rank Matrix Decomposition 低秩矩阵分解

## Probability and Mathematical Statistics 概率论与数理统计

* Probability Distributions 概率分布
    + Beta distribution vs. Binomial distribution
    + Dirichlet distribution vs. Multinomial distribution

* Parameter Estimation 参数估计方法
    + Maximum Likelihood Estimation 最大似然估计 (MLE)
    + Maximum A Posteriori probability 最大后验概率 (MAP)
    + Expectation Maximization 期望最大化 ([EM](http://blog.csdn.net/zouxy09/article/details/8537620))
    
## Numerical Optimization 数值优化

* (Batch) Gradient Descent 梯度下降法/最速下降法 (GD)

* Stochastic Gradient Descent 随机梯度下降法 (SGD)

* Conjugate Gradient Descent [共扼梯度下降法](http://www.cnblogs.com/daniel-D/p/3377840.html)

* Levenberg-marquardt 

* Newton Method [牛顿法](http://blog.csdn.net/dsbatigol/article/details/12448627)

* Qusi-newton Method [拟牛顿法](http://blog.csdn.net/itplus/article/details/21896453)
    + DFP
    + BFGS
    + L-BFGS
