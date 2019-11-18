'''Creating the vocabulary'''


import pdb
import pickle
import gc
import copy
import random
from scipy import interp
import numpy as np
import base64
import zlib
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


# input: one pandas dataframe and one list(?)
# df1: data to create vocabulary from [ids, labels (int), titles, abstracts]
# df2: stopwords (in list?)
def azureml_main(dataframe1 = None, dataframe2 = None):


	# load in data - articles to create vocabulary from
	myData = np.array(dataframe1)

	# stopwords that we want to exclude
	stopwords = np.array(dataframe2)

	# the ids of the papers that we build the vocabulary from
	ids = myData[:,0]

	# the labels (i.e. Review Groups) that we build the vocabulary from
	labels = myData[:,1].astype(int)

	# list to store all vocabularies
	all_features = []

	# loop over all papers
	for index in range(len(myData)):

		# extract title of paper
		title_tokens = myData[:,2][index]

		# extract abstract of paper
	   	abstract_tokens = myData[:,3][index]

		# save title as ["TI_This", "TI_is", "TI_a", "TI_title"] with stopwords excluded
		features = ["TI_%s" % t for t in title_tokens.split(" ") if not t in stopwords]

		# save abstract as ["AB_This", "AB_is", "AB_an", "AB_abstract"]
	   	features2 = ["AB_%s" % t for t in abstract_tokens.split(" ") if not t in stopwords]

		# combine into one list (and then join as one string)
	   	features.extend(features2)

		all_features.append(" ".join(features))

	# vectorizer object
	# ngram range: consider all 1-to-3-grams
	# only consider 75000 most frequently used words in the vocabulary
	# min_df: a word has to occur in minimum 3 documents
	# strip_accents: remove accents and 'other character normalization'
	vec = TfidfVectorizer(ngram_range=(1,3), max_features=75000, min_df=3, strip_accents='unicode')


	# X is a matrix of shape [len(all_features), len(vocab)] - i.e. [num_of_documents, all_words_in_vocab] ie the term-document matrix
	# each value is the TF-IDF score of each word in the document
   	X = vec.fit_transform(all_features)

	# gets the vocabulary
   	vocab = vec.get_feature_names()

	print(vocab)
   	#vocab_pickled = pickle.dumps(vocab)

	# pickle the vocabulary
   	vocab_pickled = base64.b64encode(zlib.compress(pickle.dumps(vocab)))

	# save the pickled vocabulary in a dataframe
   	output = pd.DataFrame({"vocab_pickled":[vocab_pickled]})

   	return output,


df1 = pd.read_csv('data/dummy_articles_with_classes.csv')
df2 = ['a', 'the']
azureml_main(df1, df2)
