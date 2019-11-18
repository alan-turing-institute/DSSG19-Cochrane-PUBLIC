'''Generate one model (for one Review Group). Input is a file with positive and negative examples'''

import pdb
import pickle
import gc
import copy
import random
import zlib
import base64
from scipy import interp
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


# input: two pandas dataframes
# df1: articles [id, label, title, abstract]
# df2: df with single column "vocab_pickled", with as value the pickled vocabulary
def azureml_main(dataframe1 = None, dataframe2 = None):

	# save articles as np object
	myData = np.array(dataframe1)

	# get the vocabulary as pickle
	vocab_pickled = dataframe2["Vocabulary"][0]

   	#vocab = pickle.loads(vocab_pickled)

	# load vocabulary from pickle
   	vocab = pickle.loads(zlib.decompress(base64.b64decode(vocab_pickled)))

	# the ids in the dataset
   	ids = myData[:,0]

	# the labels in the dataset
   	labels = [1]*myData[:,1].astype(int)

	# save all features in list
	all_features = []

	# loop over all articles
	for index in range(len(myData)):

		# title of article
	   title_tokens = myData[:,2][index]

	   # abstract of article
	   abstract_tokens = myData[:,3][index]

	   # title into tokens...
	   features = ["TI_%s" % t for t in title_tokens.split(" ")]

	   # abstract into tokens...
	   features2 = ["AB_%s" % t for t in abstract_tokens.split(" ")]

	   # features combined...
	   features.extend(features2)

	   # converted to string and added to list of all features, 1 entry per document
	   all_features.append(" ".join(features))

	# TF-IDF vectorizer object with
	# vocabulary = predefined vocabulary. This means that the TF-IDF score
	# will only be calculated for the words in this vocabulary.
	# ngram_range = which ngrams does it take into account (now length 1 to 3)
	# max_features = only consider the 75000 most frequent words in the corpus
	# min_df = is ignored when vocabulary is not none.. so doesn't do anything
	vec = TfidfVectorizer(vocabulary=vocab, ngram_range=(1,3), max_features=75000, min_df=3)

	# X is a matrix of shape [len(all_features), len(vocab)] - i.e. [num_of_documents, all_words_in_vocab] ie the term-document matrix
	# each value is the TF-IDF score of each word in the document
   	X = vec.fit_transform(all_features)

	# creates a logistic classifier!
	# class_weight auto?? does not seem to be a parameter in the documentation
	sgd = SGDClassifier(class_weight="auto", loss="log")

	# dictionary of parameters. Alpha receives 6 values, ranging from 1.0 to 1.0^-7
   	parameters = {'alpha':10.0**-np.arange(1,7)}

	# do a parameter search on the logistic classifier with the specified parameters,
	# selecting the one with the highest roc_auc score
   	clf = GridSearchCV(sgd, parameters, scoring="roc_auc")

	# fit the best classifier
   	clf.fit(X, labels)

	# save the model
	model = pickle.dumps(clf)

	# save into dataframe
	output = pd.DataFrame({"model" :[model]})

   	return output,

df1 = pd.read_csv("data/dummy_articles_oneclass.csv")
df2 = pd.read_csv("data/ModelAndVocab.txt", sep="\t")
azureml_main(df1, df2)
