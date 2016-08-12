## 1. Primary Topics 基础

### Classification and Regression 分类与回归

* Generalized Linear Model 广义线性模型 ([GLM](http://www.cnblogs.com/dreamvibe/p/4259460.html))
    + Linear Regression 线性回归
        - Locally Weighted Linear Regression [局部加权线性回归](http://www.cnblogs.com/hust-ghtao/p/3587971.html) [LWLR](http://wenda.chinahadoop.cn/question/3524)
    + Logistic Regression 逻辑回归

* Suport Vector Machine 支持向量机 [SVM](http://blog.csdn.net/v_july_v/article/details/7624837)
    + Linear Kernel 线性核
    + Polynomial Kernel 多项式核
    + Radial Basis Function/Gaussian Kernel 高斯核 RBF


* Neural Network 神经网络 (NN)

* K-Nearest Neighbor [K近邻](http://blog.csdn.net/yunduanmuxue/article/details/21777907) ([KNN](http://dataunion.org/4237.html))

* Bayesian Models 贝叶斯模型
    + Naive Bayes 朴素贝叶斯 (NB)
    + Bayesian Network/Belief Network/Directed Acyclic Graphical model [贝叶斯网络](http://blog.csdn.net/v_july_v/article/details/40984699#t6)/信念网络/有向无环图模型

* Decision Trees [决策树](http://blog.csdn.net/cyningsun/article/details/8735169)
    + [ID3](http://blog.csdn.net/mmc2015/article/details/42525655)
    + [C4.5](http://blog.csdn.net/delltdk/article/details/38681949)
    + [Classification](http://blog.csdn.net/u011067360/article/details/24871801) and [Regression](http://blog.csdn.net/google19890102/article/details/32329823) Tree 分类回归树 (CART)

* Ensemble [模型组合](http://baogege.info/2015/04/27/model-aggregation/)
    + 线性组合
    + Bootstrap aggregating (Bagging) -> Random Forests [随机森林](http://blog.csdn.net/holybin/article/details/25653597) ([RF](http://blog.csdn.net/dianacody/article/details/40706483))
    + Boosting [提升](http://www.cnblogs.com/liuwu265/p/4690486.html)
        * Adaptive Boosting [自适应提升](http://blog.csdn.net/v_july_v/article/details/40718799) ([AdaBoost](http://blog.csdn.net/dark_scope/article/details/14103983)) -> Boosting Tree [提升树](http://www.tqcto.com/article/framework/2770.html)
        * [Gradient Boosting](http://www.cnblogs.com/LeftNotEasy/archive/2011/01/02/machine-learning-boosting-and-gradient-boosting.html) -> Gradient-Boosted [Regression](http://blog.csdn.net/w28971023/article/details/8240756) Trees [梯度提升回归树](http://blog.csdn.net/dark_scope/article/details/24863289) ([GBRT/GBDT](http://blog.csdn.net/dianacody/article/details/40688783))
        * L2 Boosting
        * Logit Boosting
    + Cascade

### Clustering 聚类

* K-means K-均值

* DB-SCAN

* Gaussian Mixture Model 混合高斯模型 (GMM)

* Power Iteration Clustering (PIC)

### Frequent Pattern Mining 频繁模式挖掘

* Association Rules

* FP-growth

* PrefixSpan

## 2. Practice 实践

### Feature Engineering [特征工程](http://blog.csdn.net/dream_angel_z/article/details/49388733)

* Feature Construction 特征构建

* Feature Extraction 特征提取
    + Principal Component Analysis 主成分分析 ([PCA](http://www.cnblogs.com/jerrylead/archive/2011/04/18/2020209.html))
    + Linear Discriminant Analysis 线性判别分析 ([LDA](http://www.cnblogs.com/LeftNotEasy/archive/2011/01/08/lda-and-pca-machine-learning.html))
    + Independent Component Analysis 独立成分分析 ([ICA](http://www.cnblogs.com/jerrylead/archive/2011/04/19/2021071.html))

* Feature Selection [特征选择](http://blog.csdn.net/google19890102/article/details/40019271)
    + Filter 过滤式方法 
        * Coefficient Score 相关系数 
        * Chi-squared Test 卡方检验 
        * Mutual Information/Information Gain 互信息/信息增益 
    + Wrapper [封装式方法](http://www.cnblogs.com/heaad/archive/2011/01/02/1924088.html) 
        * Complete 完全搜索
        * Heuristic 启发式搜索
        * Random 随机搜索
    + Embedded 嵌入式方法
        * 正则化
        * 决策树
        * 深度学习

### Model Evaluation [模型评价](http://blog.csdn.net/heyongluoyao8/article/details/49408319)

* Model Validation 模型验证
    * Hold-out Validation
    * K-fold cross-validation K折交叉验证
    * Leave one out/Jackknife 留一交叉验证/刀切法
    * Bootstrapping 自助法

* Model Testing 模型测试
    + A/B Testing


### Model Selection [模型选择](http://pages.cs.wisc.edu/~arun/vision/)

* Feature Engineering

* Algorithm Selection

* Hyperparameter Tuning 超参数调优
    + Grid Search 格搜索
    + Random Search 随机搜索
    + Smart Search 智能搜索
        * Derivative-free optimization
        * Bayesian optimization
        * random forest smart tuning

## 3. Special Topics 专题

### Recommender System 推荐系统
* Content Filtering 

* Collaborative Filtering 协同过滤
    + Neighborhood Methods
        - Item-oriented
        - User-oriented
    + Latent Factor Models
        - [Matrix Factorization](http://dl.acm.org/citation.cfm?id=1608614) 

### Topic Models [主题模型](http://blog.csdn.net/hxxiaopei/article/details/7617838)
* Latent Semantic Indexing 潜语义索引 (LSI)

* Probability Latent Semantic Indexing 概率潜语义索引 ([pLSI](http://www.52nlp.cn/%E6%A6%82%E7%8E%87%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B%E5%8F%8A%E5%85%B6%E5%8F%98%E5%BD%A2%E7%B3%BB%E5%88%971-plsa%E5%8F%8Aem%E7%AE%97%E6%B3%95)) [[SIGIR 1999](http://dl.acm.org/citation.cfm?id=312649)]

* Latent Dirichlet Allocation ([LDA](http://blog.csdn.net/v_july_v/article/details/41209515)) [[JMLR 2003](http://dl.acm.org/citation.cfm?id=944919.944937)]

### Sequence Labeling [序列标注](http://blog.csdn.net/caohao2008/article/details/4242308) 

* Hidden Markov Model 隐马尔科夫模型 ([HMM](http://www.52nlp.cn/hmm-learn-best-practices-one-introduction))
    + Evaluation 评估: Forward algorithm [前向算法](http://www.cnblogs.com/tornadomeet/archive/2012/03/24/2415583.html)
    + Decoding 解码: Viterbi algorithm [维特比算法](http://www.cnblogs.com/tornadomeet/archive/2012/03/24/2415889.htm)
    + Learning 学习: Forward-backward algorithm 前向-后向算法
    
* Maximum Entropy Markov Model 最大熵马尔科夫模型 (MEMM)
    + Label Bias Problem [标注偏置问题](http://blog.csdn.net/zhoubl668/article/details/7787690)

* Markov Random Field 马尔科夫随机场 (MRF)

* Conditional Random Field 条件随机场 (CRF)


### [Deep Learning](http://deeplearning.net/) [深度学习](http://blog.csdn.net/zouxy09/article/details/8775360) 

* AutoEncoder [自动编码器](http://blog.csdn.net/zouxy09/article/details/8775524)
    + Sparse AutoEncoder 稀疏自动编码器
    + Denoising AutoEncoders 降噪自动编码器

* Sparse Coding [稀疏编码](http://blog.csdn.net/zouxy09/article/details/8777094)

* Restrict Boltzmann Machine 限制波尔兹曼机 (RBM)

* Deep Belief Networks 深信度网络

* Convolutional Neural Networks [卷积神经网络](http://blog.csdn.net/zouxy09/article/details/8781543)

## 4. Important Concepts 重要概念

* Bias vs. Variance [偏差与方差](http://nanshu.wang/post/2015-05-17/)

* Underfitting vs. Overfitting 欠拟合与过拟合

* Regularization [正则化](https://site.douban.com/182577/widget/notes/10567212/note/288551448/)
    + Ridge Regression 岭回归
    + Least Absolute Shrinkage and Selection Operator 最小绝对值收敛和选择算子算法 LASSO

* Normalization 归一化

* Learning Curve [学习曲线](http://52opencourse.com/217/coursera%E5%85%AC%E5%BC%80%E8%AF%BE%E7%AC%94%E8%AE%B0-%E6%96%AF%E5%9D%A6%E7%A6%8F%E5%A4%A7%E5%AD%A6%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E7%AC%AC%E5%8D%81%E8%AF%BE-%E5%BA%94%E7%94%A8%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E7%9A%84%E5%BB%BA%E8%AE%AE-advice-for-applying-machine-learning)

* Discriminative Model vs. Generative Model 判别式模型与生成式模型

* Parametric Model vs. Nonparametric Model [参数模型和非参数模型](http://wenda.chinahadoop.cn/question/3669) 

## 5. Mathematical Fundament 数学基础

### Linear Algebra 线性代数

* Eigenvalue Decomposition 特征值分解

* Singular Value Decomposition 奇异值分解 ([SVD](http://www.cnblogs.com/LeftNotEasy/archive/2011/01/19/svd-and-applications.html))

* Low Rank Matrix Decomposition 低秩矩阵分解
    - Stochastic Gradient Descent
    - Alternating Least Squares (ALS)

### Probability and Mathematical Statistics 概率论与数理统计

* Probability Distributions 概率分布
    + Conjugate Prior [共轭先验](http://blog.csdn.net/xianlingmao/article/details/7340099)
        - Beta distribution and Binomial distribution
        - Dirichlet distribution and Multinomial distribution
    + [Exponential Family](https://en.wikipedia.org/wiki/Exponential_family) [指数族](http://blog.csdn.net/stdcoutzyx/article/details/9207047)
        - Gaussian Distribution
        - Binomial Distribution
        - Poisson Distribution
        - Gamma Distribution
        - Exponential Distribution
        - Beta Distribution
        - Dirichlet Distribution


* Parameter Estimation 参数估计方法
    + Maximum Likelihood Estimation 最大似然估计 (MLE)
    + Maximum A Posteriori probability 最大后验概率 (MAP)
    + Expectation Maximization 期望最大化 ([EM](http://blog.csdn.net/zouxy09/article/details/8537620))
    + Monte Carlo Simulation [蒙特卡罗模拟](http://www.52nlp.cn/lda-math-mcmc-%E5%92%8C-gibbs-sampling1)
        - Metropolis–Hastings algorithm
        - Gibbs sampling

### Numerical Optimization 数值优化

+ First Order Derivative一阶导数法
    * (Batch) Gradient Descent (批量)梯度下降法/最速下降法 (GD)

    * Stochastic Gradient Descent 随机梯度下降法 (SGD)

    * Mini-Batch Gradient Descent 微型批量梯度下降

    * Conjugate Gradient Descent [共轭梯度下降法](http://www.cnblogs.com/daniel-D/p/3377840.html)
    
    * Levenberg-Marquardt 

+ Second Order Derivative 二阶导数法
    * Newton Method [牛顿法](http://blog.csdn.net/dsbatigol/article/details/12448627)

    * Qusi-newton Method [拟牛顿法](http://blog.csdn.net/itplus/article/details/21896453)
        + DFP
        + BFGS
        + L-BFGS



