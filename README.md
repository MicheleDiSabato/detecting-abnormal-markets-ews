# Authors:
* [Raffaella D'Anna](https://www.linkedin.com/in/raffaella-d-anna-b779501a6/)
* [Alessandro Del Vitto](https://www.linkedin.com/in/alessandro-del-vitto-46a9b9241/)
* [Michele Di Sabato](https://www.linkedin.com/in/michele-di-sabato/)
* [Rita Numeroli]()
* [Andrea Puricelli](https://www.linkedin.com/in/andrea-puricelli-/)

<p align="center">
  <img src="plots/readme_images/pic.PNG">
</p>

# Detecting Abnormal Markets - Early Warning Systems

## :clipboard: Framework:
To improve financial performance and prevent risks, it is important to detect anomalous behavior in the Financial Markets,
which have a tendency to crash.
* During normal periods or **risk-on** periods, investors have a high-risk appetite and bid up the prices of risky assets in the
market
* During a crisis, or **risk-off** periods, risk premia and financial assets exhibit anomalous behavior; investors become more
risk-averse and sell risky assets, sending their prices lower, and tend to gravitate toward lower-risk investments.

## :dart: Goal and statistical tools:
In this project we use [copulas](https://arxiv.org/abs/2009.09463) and k-NN coupled with over-sampling techniques[^1] (Naive random oversampling, [SMOTE](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html), [BorderlineSMOTE](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.BorderlineSMOTE.html), [ADASYN](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.ADASYN.html) and [Naive random over-sampling](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.RandomOverSampler.html)) applied to some financial indices, with the goal of classifying a week as a risk-on week (nromal week) or risk-off week (anomaly)[^1].[^1]

Feel free to visit [this link](report_slides.pdf) for more details.

## :chart_with_upwards_trend: The dataset:
The dataset is composed of 42 time series, corresponsing to the time evolution of 42 indices from [Bloomberg](https://www.bloomberg.com/europe):
![VIX_XAUBGNL_EONIA_USGG2YR](plots/VIX_XAUBGNL_EONIA_USGG2YR.png)
The vertical blue lines correspond to risk-off (anomalous) weeks, while the white regions are risk-on (normal) periods. 

## :pushpin: Workflow:
To improve the classification performance and to apply "static" statistical analysis (such as statistical tests), we transformed the data to achieve stationarity, passing to the logarithm (for non negative indices) or to finite differences. For example, the previous four indices become:
![VIX_XAUBGNL_EONIA_USGG2YR_stationary](plots/VIX_XAUBGNL_EONIA_USGG2YR_stationary.png)

We used augmented Dickey Fuller test to check whether stationarity has been reached.

Many features are correlated, some have low variance with repect to the others and some indices will not help to predict anomalies according to a financial perspective. For these reasons, we used Kolmogorov Smirnov test to remove useless indices (i.e. indices which have the same distribution under risk-on and risk-off perdiods)and the remaining indices were further reduced according to correlation and financial meaning.

After an extensive feature selection, we end up with 10 indices:
![final_features](plots/final_features.png)

<!-- ![final_features_stationary](plots/final_features_stationary.png) -->

We then moved to choosing a suitable model for the classification task. We decided to rely on the recall as optimizing metric, since correctly pointing out an anomaly is more imprortant than falsly classifying a week as a risk-off week.

t-Copulas seemed to be the best offer for our task, since the dataset is highly imbalanced toward the normal data points (i.e. there are much more white regions with respect to blu ones). We found out that t-copulas had a tendency of classifying too many points as anomalies, so we decided to attach a "confidence level" to the anomalies predicted by the copula. The higher the confidence level, the higher the probability of the copula's prediction to be correct. To get this confidence level, we over-sampled the dataset[^1] and applied k-NN on each instance in our augmented dataset to produce a probability measure of that point belonging to one of the two classes. The probability is computed as the frequency of k nearest points which belong to one of the two classes. We then attached this "confidence measure" to the copula's predictions. Parameter k has been tuned using Cross Validation.

[^1]: We are aware of the [limitations](https://academic.oup.com/jamia/advance-article/doi/10.1093/jamia/ocac093/6605096?searchresult=1&login=false) of oversampling. Class imbalance has been the main limitation of the data we worked on, as the ratio of normal and anomalous weeks was rougly 85% and 15% of the entire dataset, respectively. Oversampling has been useed to restore these ratio to something easier to work with (such as 70% - 30% or 60% - 40%).

**Results** are shown in [here](python/AnomalyDetection.ipynb), [here](matlab/AnomalyDetection.m) and [here](report_slides.pdf).
