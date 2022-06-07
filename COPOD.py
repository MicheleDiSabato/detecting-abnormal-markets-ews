import scipy.io
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from scipy.stats import kstest, ks_2samp, pearsonr
import seaborn as sns
import os
from sklearn.ensemble import IsolationForest
from sklearn.metrics import confusion_matrix
from pyod.models.copod import COPOD

data = pd.read_pickle("reduced_dataset.pkl")
clf = COPOD()
clf.fit(data.values)
test_scores = clf.decision_function(data.values)
trainig_label = clf.labels_
y_test_pred, y_test_pred_confidence = clf.predict(X_test, return_confidence=True)  # outlier labels (0 or 1) and confidence in the range of [0,1]

plt.hist(label)
plt.show()