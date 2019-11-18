'''Plot the performance of the model.'''

import pdb
import pickle
import gc
import copy
import random
import zlib
import base64from scipy import interp

import numpy as np
import pandas as pd
import sklearn

from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.grid_search import GridSearchCV
from sklearn import cross_validation
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score, precision_score, recall_score, roc_auc_score, roc_curve, auc
from sklearn.cross_validation import StratifiedKFold

def azureml_main(dataframe1 = None, dataframe2 = None):

	X_pickled = dataframe2["X_pickled"][0]

	model_pickled = dataframe1["model"][0]

	clf = pickle.loads(model_pickled)

	#X = pickle.loads(X_pickled)

	X = pickle.loads(zlib.decompress(base64.b64decode(X_pickled)))

	y_preds=clf.predict_proba(X)

	# df['Col2'].map(lambda x: 42 if x > 1 else 55)

	probabilities = y_preds[:,1]

	for x in range(0, len(probabilities)):

	   	if probabilities[x] > 0.5:

			y_pred_val.append(1)

		else:

			y_pred_val.append(0)

	accuracy, precision, recall, auc = \

	accuracy_score(y_test, y_pred_val),\

	precision_score(y_test, y_pred_val),\

	recall_score(y_test, y_pred_val),\

   	roc_auc_score(y_test, probabilities)

	import matplotlib.pyplot as plt

   	metrics = pd.DataFrame();

  	metrics["Metric"] = ["Accuracy", "Precision", "Recall", "AUC"];

   	metrics["Value"] = [accuracy, precision, recall, auc]    # Plot ROC Curve

   	fpr, tpr, thresholds = roc_curve(y_test, probabilities)

	fig = plt.figure()

	axis = fig.gca()

	axis.plot(fpr, tpr, linewidth=4)

	axis.grid("on")

	axis.set_xlabel("False positive rate")

	axis.set_ylabel("True positive rate")

	axis.set_title("ROC Curve")

	fig.savefig("roc.png")


   output = pd.DataFrame({"Scores":pd.Series(probabilities)})

   return output,
