import sys
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from tqdm import tqdm

sys.path.append('../store/')
from persist import create_hash_id, check_persisted, load, save


class ElasticClassifier:

    """
    Elastic net classifer

    One-vs-all logistic regression. The class contains one model for each review
    group, and returns the probability that an paper belongs to each of the
    review groups.

    Parameters
    ==========

    fold_hash : str
        Unique identifier for the fold. 

    ngram_range : int
        Defines the upper bound of the ngrams that should be considered.
        Defaults to 3.

    max_vocab_f : int
        The maximum size of the vocabulary, i.e. the maximum length of the tf-idf
        vector that is created for a paper. Defaults to 75000.

    max_vocab_f2 : int
        The maximum size of the vocabulary, i.e. the maximum length of the tf-idf
        vector that is created for journal titles. Defaults to 100.

    alpha : float
        Constant that is used in the regularization (l2). Defaults to 0.0001.

    l1_ratio : float
        Constant that is used int he regularization (l1). Defaults to 0.0001.

    min_df : int
        The minimum number of documents that a feature should be in to be included
        in a vectorizer's vocabulary.

    tokens_col : str
        Column from which tokens should be pulled (for titles and abstracts).

    tokens_col2 : str
        Columns from which tokens should be pulled for journals.

    token_pattern : str
        Token pattern to use

    citations_cols : str
        Columns from which citations data should be pulled (if desired).

    env : dict
        Dict containing filepaths.
        
    load_fresh : bool
        Whether persisted files should be used (False) or not (True).    

    """

    def __init__(self, fold_hash, ngram_range=3, max_vocab_f=75000, max_vocab_f2=100,
        alpha=0.0001, l1_ratio=0.15, min_df=3, tokens_col=None, tokens_col2=None,
        token_pattern='default', citations_cols=None, env=None, load_fresh=False):

        # identify which fold model was trained on
        self.fold_hash = fold_hash

        # model contents
        self.models = defaultdict(dict)
        self.vectorizer = None
        self.vectorizer2 = None

        # parameters for tf-idf vectorizer
        self.ngram_range = (1, ngram_range)
        self.max_vocab_f = max_vocab_f
        self.max_vocab_f2 = max_vocab_f2
        self.min_df = min_df
        self.tokens_col = tokens_col
        self.tokens_col2 = tokens_col2
        if token_pattern == 'alpha':
            self.token_pattern = r'(?u)\b[A-Za-z]+\b'
        else:
            self.token_pattern = r'(?u)\b\w\w+\b'
        self.vectorizer_hash = create_hash_id(f'{self.fold_hash}{self.ngram_range}{self.max_vocab_f}{self.min_df}{self.tokens_col}')

        # parameter for logistic regression with elastic net
        self.alpha = alpha
        self.l1_ratio = l1_ratio

        # parameters for citations
        self.citations_cols = citations_cols

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
            classify, with as columns (at least):
                tokens_baseline - previously tokenized title-abstract

        y_train : pandas DataFrame
            DataFrame containing the labels for each paper. Each
            column represents one review group with binary labels.
        """

        # check if data has been created before, if so load from file
        if check_persisted(f"{self.env['store_misc']}/tfidf", f'{self.vectorizer_hash}_X', self.load_fresh):

            X = load(f"{self.env['store_misc']}/tfidf", f'{self.vectorizer_hash}_X')

            # check for vectorizers
            if self.tokens_col is not None:
                vec = load(f"{self.env['store_misc']}/tfidf", f'{self.vectorizer_hash}_vec')
                self.vectorizer = vec

            if self.tokens_col2 is not None:
                vec2 = load(f"{self.env['store_misc']}/tfidf", f'{self.vectorizer_hash}_vec2')
                self.vectorizer2 = vec2

        else:

            if self.tokens_col is not None:

                # get the tokenized papers
                tokenized_papers = list(x_train[self.tokens_col])
                vec = TfidfVectorizer(ngram_range=self.ngram_range,
                                        max_features=self.max_vocab_f,
                                        strip_accents='unicode',
                                        token_pattern=self.token_pattern,
                                        min_df=self.min_df)

                # generate term document matrix (model inputs)
                X = vec.fit_transform(tokenized_papers)

                save(vec, f"{self.env['store_misc']}/tfidf", f'{self.vectorizer_hash}_vec', persist=True)
                self.vectorizer = vec

            if self.tokens_col2 is not None:

                tokenized_papers2 = x_train[self.tokens_col2].apply(lambda x: np.str_(x))
                vec2 = TfidfVectorizer(ngram_range=self.ngram_range,
                                    max_features=self.max_vocab_f2,
                                    strip_accents='unicode',
                                    token_pattern=self.token_pattern,
                                    min_df=self.min_df,
                                    decode_error='ignore')

                X2 = vec2.fit_transform(tokenized_papers2)

                try:
                    X = hstack([X, X2])
                except:
                    X = X2

                save(vec2, f"{self.env['store_misc']}/tfidf", f'{self.vectorizer_hash}_vec2', persist=True)
                self.vectorizer2 = vec2

            if self.citations_cols is not None:

                X3 = csr_matrix(x_train[self.citations_cols].values)

                try:
                    X = hstack([X, X3])
                except:
                    X = X3

            save(X, f"{self.env['store_misc']}/tfidf", f'{self.vectorizer_hash}_X', persist=True)

        # discard fold ID column from labels
        review_groups = [col for col in y_train.columns if not col=='k']

        for review_group in tqdm(review_groups, desc='Train Review Groups'):

            # pull label column
            labels = y_train[review_group]

            # logistic classifier
            classifier = SGDClassifier(loss="log", alpha=self.alpha,
                        l1_ratio = self.l1_ratio, penalty="elasticnet").fit(X, labels)

            # save the model in dictionary of models
            self.models[review_group] = classifier

    def predict(self, papers):

        """
        Generates predictions from the trained classifiers. Each binary
        classifier is applied once.

        Parameters
        ==========

        papers : pd.DataFrame
            papers that we want to classify.

        Returns
        =======

        scores : pd.DataFrame
            Dataframe containing the predictions generated by each model.
            Each column corresponds to a review group and the values in
            that column are the probabilities that each paper belong to
            that review group.
        """

        scores = {}

        if self.tokens_col is not None:

            tokenized_papers = list(papers[self.tokens_col])

            # get vectorizer and determine tfidf for papers
            vec = self.vectorizer
            X = vec.transform(tokenized_papers)

        if self.tokens_col2 is not None:
            tokenized_papers2 = papers[self.tokens_col2].apply(lambda x: np.str_(x))

            # get vectorizer and determine tfidf for papers
            vec2 = self.vectorizer2
            X2 = vec2.transform(tokenized_papers2)

            try:
                X = hstack([X, X2])
            except:
                X = X2

        if self.citations_cols is not None:

            X3 = papers[self.citations_cols]

            try:
                X = hstack([X, X3]).todense()
            except:
                X = X3

        for model_group in tqdm(self.models, desc='Test Review Groups'):

            # get the classifier
            classifier = self.models[model_group]

            # predictions as probabilities
            y_preds = classifier.predict_proba(X)

            probabilities = y_preds[:,1]

            # store scores of model
            scores[model_group] = probabilities

        scores = pd.DataFrame.from_dict(scores)

        return scores
