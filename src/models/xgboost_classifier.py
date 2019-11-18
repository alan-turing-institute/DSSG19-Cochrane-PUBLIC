import sys
from collections import defaultdict

import pandas as pd
import xgboost as xgb
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

sys.path.append('../store/')
from persist import create_hash_id, check_persisted, load, save


class XGBoost:

    """
    XGBoost Classifer

    If TFIDF flag is specified then:

    From a predefined vocabulary that is compiled for each review group, each
    model considers only the n-grams in the paper that are contained in that
    vocabulary. It then calculates tf-idf vectors for these n-grams and classifies
    papers based on these vectors.

    Parameters
    ==========
    target : string
        Defines what we aim to classify papers into, either review_group or
        review. Defaults to review_group.

    TFIDF Parameters
    ================

    tfidf : boolean

    ngram_range : set
        Defines the upper and lower bound of the ngrams that should be considered.
        Defaults to (1, 3).

    max_vocab_f : int
        The maximum size of the vocabulary, i.e. the maximum length of the tf-idf
        vector that is created for an paper. Defaults to 75000.

    tokens_col : str
        Column which contains the tokens of interest

    XGBoost
    =======

    learning_rate: step size shrinkage used to prevent overfitting.
        Range is [0,1]

    max_depth: determines how deeply each tree is allowed to grow during
        any boosting round.

    subsample: percentage of samples used per tree. Low value can lead to
        underfitting.

    colsample_bytree: percentage of features used per tree. High value can
        lead to overfitting.

    n_estimators: number of trees you want to build.

    objective: binary:logistic

    gamma: controls whether a given node will split based on the expected
        reduction in loss after the split. A higher value leads to fewer splits.
        Supported only for tree-based learners.

    l1: L1 regularization on leaf weights. A large value leads to
        more regularization.

    l2: L2 regularization on leaf weights and is smoother than L1
        regularization.

    env : dict
        Dict containing filepaths.
        
    load_fresh : bool
        Whether persisted files should be used (False) or not (True).  
    """

    def __init__(self, fold_hash, target=None, nthread=1,
        tfidf=False, ngram_range=(1,3),
        max_vocab_f=75000, tokens_col="tokens_baseline",
        learning_rate=0.1, max_depth=5, subsample=1,
        colsample_bytree=1, n_estimators=10, objective="binary:logistic",
        gamma=None, l1=0, l2=1, env=None, load_fresh=False):

        # identify which fold model was trained on
        self.fold_hash = fold_hash

        # model contents
        self.models = defaultdict(dict)
        self.vectorizer = None

        # parameters for tf-idf vectorizer
        self.tfidf = tfidf
        self.ngram_range = ngram_range
        self.max_vocab_f = max_vocab_f
        self.tokens_col = tokens_col
        self.vectorizer_hash = create_hash_id(f'{self.fold_hash}{self.ngram_range}{self.max_vocab_f}{self.tokens_col}')

        # parameter for XGBoost
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.n_estimators = n_estimators
        self.objective = objective
        self.gamma = gamma
        self.l1 = l1
        self.l2 = l2

        self.env = env
        self.load_fresh = load_fresh

    def train(self, x_train, y_train):

        """
        Trains one elastic logistic classifier per review group. Saves the trained
        classifiers within self.models.

        Parameters
        ==========

        x_train : pandas DataFrame
            DataFrame containing the papers we aim to
            classify, with a column for the token to use.

        y_train : pandas DataFrame
            DataFrame containing the labels for each paper. Each
            column represents one review group with binary labels.
        """

        if self.tfidf:

            # check if vectorizer has been created before, if so load from file
            if check_persisted(f"{self.env['store_misc']}/tfidf", f'{self.vectorizer_hash}_X', self.load_fresh):

                vec = load(f"{self.env['store_misc']}/tfidf", f'{self.vectorizer_hash}_vec')
                X = load(f"{self.env['store_misc']}/tfidf", f'{self.vectorizer_hash}_X')
                X = hstack([csr_matrix(x_train.drop(self.tokens_col, axis=1)),X])

            else:

                # get the tokenized papers
                tokenized_papers = list(x_train[self.tokens_col])

                vec = TfidfVectorizer(ngram_range=self.ngram_range,
                                        max_features=self.max_vocab_f,
                                        strip_accents='unicode')

                # generate term document matrix (model inputs)
                X = vec.fit_transform(tokenized_papers)
                save(vec, f"{self.env['store_misc']}/tfidf", f'{self.vectorizer_hash}_vec', persist=True)
                save(X, f"{self.env['store_misc']}/tfidf", f'{self.vectorizer_hash}_X', persist=True)
                X = hstack([csr_matrix(x_train.drop(self.tokens_col, axis=1)),X])



            self.vectorizer = vec

        else:
            X = x_train

        # discard fold ID column from labels
        review_groups = [col for col in y_train.columns if not col=='k']

        for review_group in tqdm(review_groups, desc='Train Review Groups'):

            # pull label column
            labels = y_train[review_group]

            # Create data structure for XGBoost
            data_dmatrix = xgb.DMatrix(data=X,label=labels)

            # creating parameters for xgboost
            params = {
            'objective' :self.objective,
            'learning_rate' : self.learning_rate, 'max_depth' : self.max_depth,
            'subsample' : self.subsample, 'colsample_bytree' : self.colsample_bytree,
            'n_estimators' : self.n_estimators, 'objective' : self.objective,
            'gamma' : self.gamma, 'alpha' : self.l1, 'lambda' : self.l2
            }
            # xgboost
            self.models[review_group] = xgb.train(params, data_dmatrix)

    def predict(self, papers):

        """
        Generates predictions from the trained classifiers. Each binary
        classifier is applied once.

        Parameters
        ==========

        papers : pd.DataFrame
            papers that we want to classify. Required column:
                tokens_baseline - previously tokenized title-abstract

        Returns
        =======
        
        scores : pd.DataFrame
            Dataframe containing the predictions generated by each model.
            Each column corresponds to a review group and the values in
            that column are the probabilities that each paper belong to
            that review group.
        """

        scores = {}

        if self.tfidf:
            tokenized_papers = list(papers[self.tokens_col])

            # get vectorizer and determine tfidf for papers
            vec = self.vectorizer
            X = vec.transform(tokenized_papers)
            X = hstack([csr_matrix(papers.drop(self.tokens_col, axis=1)),X])
            X = xgb.DMatrix(X)

        else:
            X = xgb.DMatrix(papers)

        for model_group in tqdm(self.models, desc='Test Review Groups'):

            # get the classifier
            classifier = self.models[model_group]

            # predictions as probabilities
            y_preds = classifier.predict(X)

            probabilities = y_preds

            # store scores of model
            scores[model_group] = probabilities

        scores = pd.DataFrame.from_dict(scores)

        return scores
