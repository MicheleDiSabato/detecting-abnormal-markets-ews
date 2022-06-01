# Detecting Abnormal Markets - Early Warning Systems

Current objectves and ideas:

1) augmented dickey fuller test for stationarity of functions
2) feature selection might be more important than stationarity
3) look at cointegration (johansen test) and fractional differentiation 
4) Kolmogorov smirnov (nonparametric test) to check if the distribution during the two periods is the same
5) rolling logistic regression and COPOD
6) combine models 
7) use [conformal prediction](https://github.com/valeman/awesome-conformal-prediction) to predict the anomalies

Moreover,
* look at correlation among the selected features: to detect an anomaly it could be useful to look for "hints", i.e. events after which the correlation among the selected features increaes in absolute value.
* use FFNN to elaborate some statistics about the time series (average, variance, skewness, kurtosis, ...). With this approach, the sime dependance of each feature is "hidden" in the statistics.
* choose the optimizing metric based on a business approach: it's better to capture a crisis whenever it happens, rahter than missing it (i.e. avoid false negatives). Still, the other metrics should be greater than a threshold.
* PCA might not be appropriate, since it is based on second moments, but anomaly detection is often related to higher order moments (e.g. fat tails).
