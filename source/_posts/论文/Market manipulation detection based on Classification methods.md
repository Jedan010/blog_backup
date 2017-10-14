---
title: Market manipulation detection based on Classification methods
tags:
	- Market manipulation
	- Classification methods
	- Anomaly detection
categories:
	- 论文
---

**Abstract**
In this paper, we use supervised machine learning methods to detect the market manipulation in China based on the information released by China Securities Regulation Commission (CSRC). For the supervised machine learning, we mainly use classified method to detect the anomaly from the daily and tick trading data of manipulated stocks. As a result, we find that the supervised machine learning method are good at detecting market manipulation from daily trading data and have poor performance of tick data. The best used supervised machine learning models are K-nearest neighbor (KNN) and Decision Tree classifier (DTC) which have over 99% of accuracy, sensitivity, specificity and area under the curve (AUC).

# 1. Introduction

arket manipulation is a deliberate attempt to intervene in  the free and fair operation of the market and create artificial, false or misleading appearances with respect to price of security [^1]. Market manipulation, broadly defined, has existed since the infancy of financial markets [^2], which has become an import issue of emerging and developed market. Manipulated security market not only distorts the prices and transactions in the security market, but also undermines the function of security market. What's more, many investors would lose significantly because of most manipulators making illegal profit.

There has been increasing research on data mining techniques in detecting market manipulation. Allen and Gale[^3]classify manipulation as action-based manipulation, information based manipulation, and trade-based manipulation. We study the trade-based manipulation that a trader attempting to manipulate a stock price simply by buying and then selling (or vice versa), without releasing any false information or taking any other publicly observable actions designed to alter the value of security , by using supervised and unsupervised machine learning methods. Based on information about the manipulated stocks from China Securities Regulation Commission (CSRC). We obtain daily stock data and tick stock data of 64 manipulated stocks published in CSRC website. We use those data to train the supervised machine learning models, K-Nearest Neighbors (KNN), Support Vector Machines (SVM), Decision Tree Learning (DTL), Linear Discriminant Analysis (LDA), Quadratic Discriminant Analysis (QDA), Logistic Regression (LR), Artificial Neural Networks (ANN), so when the sample data set is given, they can classify the manipulated period and the normal period.

Currently, most scholars' study of detecting market manipulation is theoretical and pattern description which is still hard to accurately and fast detect market manipulation. To increase the possibility and efficiency of detecting market manipulation, we apply supervised machine learning models to detect market manipulation in real time, which might undermine the extent to those individual investors losing money in trading the manipulated stocks. By collecting historical stock data, it is also very helpful for regulator to train the supervised machine learning models and detect the market manipulation.

This paper is organized as follows: Section 2 summarizes the main result of prior study. Section 3 and Section 4 give information about supervised machine learning models. Empirical analysis and data are presented in Section 4. Section 5 has our conclusion and some future works.

# 2. Literature Review

Allen and Gale (1992)[^3] and Jarrow (1992)[^4] were the first researchers to study manipulation. After studying the history of stock-price manipulation, Allen and Gale classified the manipulation as action-based manipulation, information-based manipulation and trade-based manipulation[^3]. They define trade-based manipulation as manipulating stocks through actual trading or trading orders by distorting stock market prices, rather than changing company values or issuing false information. In a dynamic model of asset markets, by investigating large traders' manipulation of trading strategies in the securities market, Jarrow found that large investors have a greater impact on stock prices[^4]. Carhart and Reed (2002)[^5] found that a large trader can manipulate the market where without derivative securities by cornering the market or lure the market to "manipulate" their own trades. And he develop a theory for option pricing in an economy with a larger trader showing that the standard option pricing models holds, but with a random “volatility” price process. Hanson and Oprea (2004)[^6] developed an experimental model to study whether the manipulators could distort the prices in a prediction market.

Some researchers have attempted to detect manipulation in different methods. Aggarwal and Wu (2003)[^7] studied The US stock market from 1990 to 2001 to detect market manipulation and found that the manipulated stock had abnormal stock prices, liquidity, volatility and return. H.Ogut and Aktas (2009)[^8] compared the performance of Artificial Neural Networks (ANN) and Support Vector Machine (SVM) with discriminant analysis and logistics regression on detecting the market manipulation and found that data mining techniques were better than multivariate techniques. According to Mongkolnavin and Tirapat (2009)[^9], association rules were applied to detect mark-the-close in intraday trades from the Thai Bond Market Association. Price variation in the market and behavior of investors were integrated to analyze warning signals in real time. The method can produce a list of investors, who perhaps are manipulators. Fallh and Kordlouie (2011)[^10] used logit model, artificial neural network, and multiple discriminant analysis to create stock price manipulation models in Tehran stock exchange. The performances of three aforesaid models were effective. The selected data were thoroughly studied by runs test, skewness test, and duration correlative test. The events of price manipulation were indicated. The selected data can be divided into two sets: manipulated and non-manipulated companies. The factors that were related to stock price manipulation were defined such as: size of company, P/E ratio, liquidity of stock, status of information clarity, and structure of shareholders. In Yang (2014)[^11], logistic regression model was chosen to detect stock price manipulation activities in Shanghai and Shenzhen market that were published as manipulated stocks. They analyzed independent variables based on primary component analysis, which increased performance for forecasting the model. The model was better than the linear regression model. Cao and McGinnity (2015)[^12] proposed the Adaptive Hidden Markov Model with Anomaly States (AHMMAS) to detect intraday stock price manipulation activities. The stock tick data were level 2 data from NASDAQ and London stock exchange. The model was tested with simulated data and real market data. The performance evaluation of AHMMAS outperforms other benchmark algorithms such as: Gaussian Mixture Models (GMM), K-Nearest Neighbors Algorithm (kNN), and One Class Support Vector Machines (OCSVM). Leangarun and Thajchayapong (2016)[^8] investigated two popular scenarios of stock price manipulations: pump-and-dump and spoof trading. They defined and used level 2 data to train the neural network models which achieve high accuracy for detecting pump-and-dump, and two dimensional Gaussian model which shows it can detect spoof trading.

# 3. Methodology
Supervised machine learning is the search for algorithms that reason from externally supplied instances to produce general hypotheses, which then make predictions about future instances[^13]. In other words, the goal of supervised learning is to build a concise model of the distribution of class labels in terms of predictor features. The resulting classifier is then used to assign class labels to the testing instances where the values of the predictor features are known, but the value of the class label is unknown. Supervised machine learning is mainly consist of classification and regression. Classification is learning a function that maps (classifies) a data item into one of several predefined classes[^14]. And similarly regression is learning a function that maps a data item to a real-valued prediction variable.

## 3.1. K-Nearest Neighbour

K-Nearest Neighbour (KNN) is an instance based classifier method. The parameter units consist of samples that are used in the method and this algorithm then assumes that all instances relate to the points in the n-dimensional space $R^{N}$[^15]. The algorithm is very expedient as the information in the training data is never lost. However, this algorithm would be suitable if the training data set is large as this algorithm is very time consuming when each of the sample in training set is processed while classifying a new data and this process requires a longer classification time.

## 3.2. Support Vector
Support Vector Method (SVM) is a classier that performs classification tasks by constructing hyperplanes in a multidimensional space that separates cases of different class labels. SVM method provides an optimally separating hyperplane in the sense that the margin between two groups is maximized. The SVM is proven to be advantageous in handling classification tasks with excellent generalization performance. The method seeks to minimize the upper bound of the generalization error based on the structural risk minimization principle. SVM training is equivalent to solve a linear constrained quadratic programming problem[^16].

## 3.3. Decision Tree Classifier
Decision Tree Classifier (DTC) is one of the most popular technique for prediction. An empirical tree represents a segmentation of the data that is created by applying a series of simple rules. Each rule assigns an observation to a segment based on the value of one input. Most of researchers have used this technique because of its simplicity and comprehensibility to uncover small or large data structure and predict the value. Romero et al. (2008) said that the decision tree models are easily understood because of their reasoning process and can be directly converted into set of IF-THEN rules[^17].

## 3.4. Linear Discriminant Analysis
Linear Discriminant Analysis (LDA) is widely used in discriminant analysis to predict the class based on a given set of measurements on new unlabeled observations[^14]. The algorithm’s ability to capture statistical dependencies among the predictor variables indicates that this algorithm would be suitable to explore the linear constraint of this study to discovery the synergy between motor and non-motor symptoms.

## 3.5. Quadratic Discriminant Analysis
Quadratic Discriminant Analysis (QDA) is closely related to linear discriminant analysis, where it is assumed that the measurements from each class are normally distributed. Unlike LDA however, in QDA there is no assumption that the covariance of each of the classes is identical. When the normality assumption is true, the best possible test for the hypothesis that a given measurement is from a given class is the likelihood ratio test.

## 3.6. Logistic Regression
Logistic Regression (LR) is a method that would use the given set of features either continuous, discrete, or a mixture of both types and the binary target, LR then computes a linear combination of the inputs and passes through the logistic function[^18]. This method is commonly used because it is easy to implementation and it provides competitive results.

## 3.7. Artificial Neural Networks
Artificial Neural Network (ANN) is another popular technique used in educational data mining. The advantage of neural network is that it has an ability to detect all possible interactions between predictors' variables[^19]. Artificial neural network could also do a complete detection without having any doubt even in complex nonlinear relationship between dependent and independent variables[^20]. Therefore, artificial neural network technique is selected as one of the best prediction method.

# 4. Empirical Analysis
We used supervised machine learning models to detect the daily trading data and tick trading data of the manipulated stock to find out the manipulated time and evaluate the supervised machine learning model. The supervised machine learning models are suitable to daily trading data and have high accuracy to detect the manipulated data, while have poor performance of detecting the tick trading data.

## 4.1. Data Description and Data Preprocessing
In this paper, we use the market manipulation cases released by China Security Regulatory Commission(CSRC). Once CSRC officials discovered someone have manipulated the stock market, they would punish the manipulators and release this case on the website regularly. Based on that, we can download the daily trading data and tick trading data of these manipulated stocks. We select 64 manipulated stocks of daily trading data, which include daily open price, daily highest price, daily lowest price, daily close price and daily trading volume and tick trading data, which include tick price, tick price change volume, tick trading volume, tick trading amount and type released by CSRC from 2013 to 2016. For each stock's analysis period is from the time that the first manipulation day -2\*the manipulation period to that the end manipulation day + 2\* the manipulation period (Table1), which make sure the rate of manipulated data about 20%.

We combine all of the daily trading data and tick trading data, and standardize them by zero-mean normalization. We get 4, 593 daily trading data and 8, 986, 466 tick trading data. We simple random sampling 1% of the tick trading data as the aggregate of tick trading is too big to compute (Table2). And we transform the nonnumerical data into numerical data. Then, we label the data as 1 if it was manipulated at the time according to CSRC, and others as 0. So the input data for the supervised machine learning is the preprocessed daily trading data and tick trading data, while the output is binary, marked 0 and 1.

<center>Table 1: Some manipulated stock cases</center>

| stock   name                | Stock code | starting day of manipulation  | ending dayof manipulation  | starting day of analysis | ending dayof analysis |
|-----------------------------|------------|-------------------------------|----------------------------|--------------------------|-----------------------|
| Jingyi Co., Ltd             | 002295     | 1/19/2015                     | 1/19/2015                  | 1/15/2015                | 1/21/2015             |
| Fuda Co., Ltd               | 603166     | 7/5/2016                      | 7/18/2016                  | 5/26/2016                | 8/25/2016             |
| Zhongxing Commerce Co., Ltd | 000715     | 1/4/2013                      | 5/26/2014                  | 2/12/2009                | 4/17/2018             |
| Shibeigaoxin Co., Ltd       | 600604     | 9/8/2015                      | 9/9/2015                   | 9/2/2015                 | 9/15/2015             |
| Shuangxin Co., Ltd          | 300100     | 11/26/2014                    | 11/28/2014                 | 11/18/2014               | 12/8/2014             |

<center>Table 2: Analysis stock data</center>

|             | daily  | tick    | simplified tick |
|-------------|--------|---------|-----------------|
| total point | 4593   | 8986466 | 89864           |
| out point   | 919    | 1985096 | 19851           |
| out rate    | 20.27% | 22.09%  | 22.09%          |

## 4.2. Performance Evaluation
Measures of the quality of classification are built from a confusion matrix which records correctly and incorrectly recognized examples for each class. Table 3 presents a confusion matrix for binary classification, where TP are true positive, TF are false positive, FN are false negative, and TN are true negative counts. In this paper, for example, TN means this data are predicted as being manipulated and they are truly being manipulated according to CSRC.

<center>Table 3: Confusion Matrix</center>

|              |         | predicted class |         |
|:------------:|---------|-----------------|---------|
|              |         | success         | failure |
| Actual class | success | TP              | FN      |
|              | failure | FP              | TN      |

Classifying performance without focusing on a class is the most general way of comparing algorithms.
It does not favor any particular application. The most used empirical measure, accuracy, does not distinguish between the number of correct labels of different classes. Accuracy approximates how effective the algorithm is by showing the probability of the true value of the class label. In other words it assesses the overall effectiveness of the algorithm.


$$ Accuracy = \frac{TP+TN}{TP+FP+FN+TN} $$

Corresponding to it, two measures that separately estimate a classifier’s performance on different classes are sensitivity and specificity which approximates the probability of the positive (negative) label being true; in other words, it assesses the effectiveness of the algorithm on a single class.


$$ Sensitivity =   \frac{TP}{TP + FN} $$


$$
Specificity =   \frac{TN}{FP + TN}
$$

A comprehensive evaluation of classifier performance can be obtained by the ROC: shows a relation between the sensitivity and the specificity of the algorithm ROC curves, which plot sensitivity as a function of specificity for all possible thresholds[^21], and illustrate a classifier’s trade-off between true positives and false negatives. A higher value of sensitivity for a given value of specificity indicates better performance. The area under the ROC curve (AUC) is a commonly used metric for evaluating a classifier’s performance.

## 4.3. Experimental Results
We use 5-fold cross-validation to evaluate our predictive models by partitioning the daily training data and tick training data into a training set to train our models, and a test set to evaluate them. For the supervised machine learning models, we use K-nearest neighbors (KNN), Decision tree classifier (DTC),Linear discriminant analysis (LDA), Quadratic discriminant analysis (QDA), Logistic regression (LR), Artificial neural networks (ANN), Support vector machines(SVM).
Table 4 and Table 5 show the confusing matrix and the proportion of all mentioned models of daily training data and tick training data. As can be seen, most of the TP and TN have a rather higher rate than FN and FP of the daily training data, while the rate of TN is lower than FP and even some are near to zero of the tick training data.

<center>Table 4: Confusing Matrix of daily training data</center>

|  KNN  |   　  |  DTC  |   　  |  LDA  |   　  |  QDA  |   　  |  ANN  |   　  |  LGR  |   　  |  SVM  |   　  |
|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|
|  3658 |   4   |  3658 |   4   |  3658 |   4   |  3622 |   40  |  3658 |   4   |  3596 |   66  |  3658 |   4   |
|   0   |  931  |   0   |  931  |  434  |  497  |  236  |  695  |   88  |  843  |  434  |  497  |  372  |  559  |
| 79.6% |  0.1% | 79.6% |  0.1% | 79.6% |  0.1% | 78.9% |  0.9% | 79.6% |  0.1% | 78.3% |  1.4% | 79.6% |  0.1% |
|  0.0% | 20.3% |  0.0% | 20.3% |  9.4% | 10.8% |  5.1% | 15.1% |  1.9% | 18.4% |  9.4% | 10.8% |  8.1% | 12.2% |

<center>Table 5: Confusion Matrix of tick training data</center>


|  KNN  |   　  |  DTC  |   　  |  LDA  |  　  |  QDA  |  　  |  ANN  |  　  |  LGR  |  　  |  SVM  |  　  |
|:-----:|:-----:|:-----:|:-----:|:-----:|:----:|:-----:|:----:|:-----:|:----:|:-----:|:----:|:-----:|:----:|
| 58468 | 11545 | 52871 | 17142 | 68854 | 1159 | 66168 | 3845 | 69772 |  241 | 69250 |  763 | 69348 |  665 |
| 15507 |  4344 | 14106 |  5745 | 18490 | 1361 | 17262 | 2589 | 19091 |  760 | 18788 | 1063 | 18940 |  911 |
| 65.1% | 12.8% | 58.8% | 19.1% | 76.6% | 1.3% | 73.6% | 4.3% | 77.6% | 0.3% | 77.1% | 0.8% | 77.2% | 0.7% |
| 17.3% |  4.8% | 15.7% |  6.4% | 20.6% | 1.5% | 19.2% | 2.9% | 21.2% | 0.8% | 20.9% | 1.2% | 21.1% | 1.0% |

Fig 1 presents the result of different models of daily and tick training data based on above 4 evaluation index, accuracy, sensitivity, specificity and AUC. From the fig 1(a), most of models of daily training have a good accuracy, sensitivity and AUC which are over 90%, and some of them even reach 100%, while the result of specificity is not good as other index for some models of daily training data. It is worth noting that KNN and DTC perform excellent in the four indexes that all exceed 99%.

<center>Fig. 1:(a) daily training data</center>

![daily training data](http://oeclakg19.bkt.clouddn.com/daily.png)

<center>Fig. 1:(b) tick training data</center>

![tick training data](http://oeclakg19.bkt.clouddn.com/tick.png)


The result is not ideal for the supervised machine learning models to train tick training data according to the fig 1(b) showing all of the models have high score of sensitivity and rather low specificity. Our mainly aim is to accurately detect the manipulated data which pursue a high specificity.

Fig 2 shows Receiver Operating Characteristic (ROC) and Area Under the Curve (AUC) for different models of daily and tick training data. The AUC of all the models of daily training data is greater than 90% which means excellent estimate result, while the AUC of all the models of tick training data is nearly 50% which is slightly better than random estimate result.

<center>Fig. 2: (a) daily training data</center>

![ROC_daily_colorless](http://oeclakg19.bkt.clouddn.com/ROC_daily_colorless.png)


<center>Fig. 2: (a) daily training data</center>

![ROC_tick_colorless](http://oeclakg19.bkt.clouddn.com/ROC_tick_colorless.png)


# 5. Conclusion and Future Work
This paper presents a comparative machine learning method for market manipulation detecting, especially for stock-price market manipulation. Based on manipulated information released by China Securities Regulation Commission (CSRC), we use supervised and unsupervised machine learning models to detect the anomaly from daily and tick trading data from the manipulated stocks. The supervised machine learning models, which are mainly classify machine learning models, have excellent performance for detecting the anomaly from the daily data, while having poor performance of tick data. Among the used classify machine learning models, KNN and DTC are the best, which exceed 99% of all the indexes, including accuracy, sensitivity, specificity and AUC.

For the supervised machine learning models to detecting the anomaly from the tick trading data, one of the reason for its poor performance is hard to exactly label the tick trading data as normal or abnormal, because it is almost impossible to know exact which specific time or specific tick trading data was manipulated. As the same, the evaluation of clustering tick trading data are inaccurate as the difficulty of deciding accurate anomalies. So a more suitable way to process the tick trading data is worth to be further studied.

# 6. Acknowledgement
This paper is partly supported by the National Natural Science Foundation (71401188), Beijing Social Science Foundation (15SHB017) and Supported by Program for Innovation Research in Central University of Finance and Economics.


[Paper Download](http://oeclakg19.bkt.clouddn.com/Market%20manipulation%20detection%20based%20on%20Classification%20methods.pdf "Market manipulation detection based on Classification methods")

[^1]:T. C. Lin, The new market manipulation
[^2]:J. W. Markham, Law enforcement and the history of financial market manipulation, ME Sharpe, 2013.
[^3]:F. Allen, D. Gale, Stock-price manipulation, The Review of Financial Studies (1992) 503–529
[^4]:R. A. Jarrow, Market manipulation, bubbles, corners, and short squeezes, Journal of financial and Quantitative Analysis 27 (3) (1992)
311–336.
[^5]:M. M. Carhart, R. Kaniel, D. K. Musto, A. V. Reed, Leaning for the tape: Evidence of gaming behavior in equity mutual funds, The
Journal of Finance 57 (2) (2002) 661–693.
[^6]:R. Hanson, R. Oprea, Manipulators increase information market accuracy, George Mason University.
[^7]:R. K. Aggarwal, G. Wu, Stock market manipulation-theory and evidence.
[^8]:H. O gut, M. M. Do  ganay, R. Aktas, Detecting stock-price manipulation in an emerging market: The case of Turkey, Expert Systems ˇ
with Applications 36 (9) (2009) 11944–11949.
[^9]:J. Mongkolnavin, S. Tirapat, Marking the Close analysis in Thai Bond Market Surveillance using association rules, Expert Systems with
Applications 36 (4) (2009) 8523–8527.
[^10]:S. M. Fallah, H. Kordlouie, Forecasting Stock Price Manipulation in Capital Market.
[^11]:F. Yang, H. Yang, M. Yang, Discrimination of China’s stock price manipulation based on primary component analysis, in: Behavior,
Economic and Social Computing (BESC), 2014 International Conference on, IEEE, 2014, pp. 1–5.
[^12]:Y. Cao, Y. Li, S. Coleman, A. Belatreche, T. M. McGinnity, Adaptive hidden Markov model with anomaly states for price manipulation
detection, IEEE transactions on neural networks and learning systems 26 (2) (2015) 318–330.
[^13]:S. B. Kotsiantis, I. Zaharakis, P. Pintelas, Supervised Machine Learning: A Review of Classification Techniques, 2007.
[^14]:U. Fayyad, G. Piatetsky-Shapiro, P. Smyth, From data mining to knowledge discovery in databases, AI magazine 17 (3) (1996) 37
[^15]:R. S. Michalski, J. G. Carbonell, T. M. Mitchell, Machine Learning: An Artificial Intelligence Approach, Springer Science & Business
Media, 2013.
[^16]:B. Zheng, S. W. Yoon, S. S. Lam, Breast cancer diagnosis based on feature extraction using a hybrid of K-means and support vector
machine algorithms, Expert Systems with Applications 41 (4) (2014) 1476–1482.
[^17]:C. Romero, S. Ventura, P. G. Espejo, C. Hervas, Data mining algorithms to classify students, in: Educational Data Mining 2008, 2008. ´
[^18]:P. J. Garc´ıa-Laencina, P. H. Abreu, M. H. Abreu, N. Afonoso, Missing data imputation on the 5-year survival prediction of breast cancer
patients with unknown discrete values, Computers in biology and medicine 59 (2015) 125–133.
[^19]:G. Gray, C. McGuinness, P. Owende, An application of classification models to predict learner progression in tertiary education, in:
Advance Computing Conference (IACC), 2014 IEEE International, IEEE, 2014, pp. 549–554.
[^20]:P. M. Arsad, N. Buniyamin, others, A neural network students’ performance prediction model (NNSPPM), in: Smart Instrumentation,
Measurement and Applications (ICSIMA), 2013 IEEE International Conference On, IEEE, 2013, pp. 1–5.
[^21]:P. Cortez, A. Cerdeira, F. Almeida, T. Matos, J. Reis, Modeling wine preferences by data mining from physicochemical properties,
Decision Support Systems 47 (2009) 547–553.
