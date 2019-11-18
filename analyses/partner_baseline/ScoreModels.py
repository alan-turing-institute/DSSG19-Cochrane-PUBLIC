'''Loading a predefined model and applying the score to articles'''

import pdb
import pickle
import gc
import copy
import random
import base64
import zlib
import joblib

from scipy import interp

import numpy as np

import pandas as pd
import sklearn
print(sklearn.__version__)

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
# dataframe1: [id, Title, Abstract, BatchID]
# dataframe2: [ModelDataID, ModelID, Model, Vocabulary]
def azureml_main(dataframe1 = None, dataframe2 = None):

    print("start!")

    # saving features: titles and abstracts tokenized
    all_features = []

    # saving data with titles and abstract as numpy object
    myData = np.array(dataframe1)
    #myModels = np.array(dataframe2)

    # ids are in first column
    ids = myData[:,0]

    # batchIDs are in third column
    BatchId = myData[:,3]

    #utcDateCreated = myData[:,4]

    # empty dataframe to which final outputs are appended
    finalOutput = pd.DataFrame(columns=["ReferenceId", "Score", "ModelId", "BatchId"])

    # loop over documents to classify
    for index in range(len(myData)):

        # look at title of a particular paper
        title_tokens = myData[:,1][index]

        # look at abstract of a particular paper
        abstract_tokens = myData[:,2][index]

        # title words are saved as ["TI_This", "TI_is", "TI_a", "TI_title"]
        features = ["TI_%s" % t for t in title_tokens.split(" ")]

        # abstract words are saved as ["AB_This", "AB_is", "AB_an", "AB_abstract"]
        features2 = ["AB_%s" % t for t in abstract_tokens.split(" ")]

        # features are combined into one list and then joined into one string
        features.extend(features2)
        all_features.append(" ".join(features))

        # features looks like this (string): "TI_This TI_is TI_a TI_title AB_This AB_is AB_an AB_abstract"
        # and all_features is a list of these

    # looping over the models
    for modelIndex in range(len(dataframe2.index)):

        #print(modelIndex)

        # select the model ID
        ModelId = dataframe2["ModelId"][modelIndex]

        # select the model vocabulary
        vocab_pickled = dataframe2["Vocabulary"][modelIndex]

        # select the model itself
        model_pickled = dataframe2["Model"][modelIndex]

        # 10000 is an identifier for the RCT model? so this is for all Review Group classifiers
        if (str(ModelId).strip() != '10000'):

            if not type(vocab_pickled) is float:# and not np.isnan(vocab_pickled):

                vocab = pickle.loads(zlib.decompress(base64.b64decode(vocab_pickled)))
                step1 = base64.b64decode(model_pickled)
                step2 = zlib.decompress(step1)
                #print(step2)
                clf = pickle.loads(step2)

                #if not np.isnan(vocab_pickled):
                #vocab = pickle.loads(vocab_pickled)
                #clf = pickle.loads(model_pickled)

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

                # clf is the model that is applied (but unclear what kind of model it is)
                y_preds=clf.predict_proba(X)
                probabilities = y_preds[:,1]

                #output = pd.DataFrame({"ReferenceID":[ids], "ScoredValue":[probabilities]})
                #output = pd.DataFrame([ids, probabilities], axis=1)

                # save output of the model to intermediate DF
                TempOutput = pd.DataFrame({"ReferenceId":pd.Series(ids), "Score":pd.Series(probabilities), "ModelId": ModelId, "BatchId":pd.Series(BatchId)})

                # append to final DF
                finalOutput = finalOutput.append(TempOutput)

    print("Done")
    print(finalOutput)
    return finalOutput,

df2 = pd.read_csv('data/ModelAndVocab.txt', sep='\t')
df1 = pd.read_csv('data/dummy_articles.csv')
# print(df1.head())
#print(df2["Vocabulary"])
azureml_main(df1, df2)
