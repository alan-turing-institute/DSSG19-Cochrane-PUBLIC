import sys

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

sys.path.append('../store/')
from persist import create_hash_id, check_persisted, load, save


class RandomForestTFIDF:

    """
    Random Forest classifer using TFIDF

    This ensemble RF classifer uses the defined features to initialize, train
    and then perform prediction.

    Parameters
    ==========
    fold_hash : str
        Unique identifier for the fold.
    seed : int
        Seed for random forest.

    ngram_range : int
        Defines the upper bound of the ngrams that should be considered.
        Defaults to 3.

    max_vocab_f : int
        The maximum size of the vocabulary, i.e. the maximum length of the tf-idf
        vector that is created for a paper. Defaults to 75000. (title + abstract)
    max_vocab_f2 : int
        The maximum size of the vocabulary, i.e. the maximum length of the tf-idf
        vector that is created for a paper. Defaults to 100. (journal)
    min_df : int
        The minimum number of documents that a feature should exist in to be included
        in a vectorizer.


    n_estimators: int
        The number of trees to be trained
    max_depth: int
        The maximum number of branches to employ for each tree
    max_features: int or str
        The maximum number of features to consider at each split
    n_jobs : int
        The number of cores the process should be run on.     
    min_samples_split: int
        The minimum number of data points there must be per split
    min_samples_leaf: int
        The minimum number of samples there must be per leaf

    tokens_col : str
        The column from which the token features should be extracted (title + abstract)
    tokens_col2 : str
        The column from which the token features should be extracted (journal)
    token_pattern : str
        The tokenization pattern that should be used, either default or alpha

    env : dict
        Dict containing filepaths. 
    load_fresh : bool
        Whether persisted files should be used (False) or not (True).  
    """

    def __init__(self, fold_hash, seed, ngram_range=3, max_vocab_f=75000, max_vocab_f2=100,
                 min_df=3, n_estimators=10, max_depth=None, max_features='auto', n_jobs=3,
                 min_samples_split=2, min_samples_leaf=1, tokens_col="tokens_no_stopwords",
                 tokens_col2=None, token_pattern="default", env=None, load_fresh=False):

        # identify which fold model was trained on
        self.fold_hash = fold_hash

        # model contents
        self.models = {}
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

        # parameter for model
        self.seed = seed
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.n_jobs = n_jobs

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

        # check if vectorizer has been created before, if so load from file
        if check_persisted(f"{self.env['store_misc']}/tfidf", f'{self.vectorizer_hash}_X', self.load_fresh):

            vec = load(f"{self.env['store_misc']}/tfidf", f'{self.vectorizer_hash}_vec')
            X = load(f"{self.env['store_misc']}/tfidf", f'{self.vectorizer_hash}_X')

            if self.tokens_col2 is not None:
                vec2 = load(f"{self.env['store_misc']}/tfidf", f'{self.vectorizer_hash}_vec2')
                self.vectorizer2 = vec2


        else:

            # get the tokenized papers
            tokenized_papers = list(x_train[self.tokens_col])
            vec = TfidfVectorizer(ngram_range=self.ngram_range,
                                    max_features=self.max_vocab_f,
                                    strip_accents='unicode',
                                    token_pattern=self.token_pattern,
                                    min_df=self.min_df)

            # generate term document matrix (model inputs)
            X = vec.fit_transform(tokenized_papers)

            if self.tokens_col2 is not None:

                tokenized_papers2 = x_train[self.tokens_col2].apply(lambda x: np.str_(x))
                vec2 = TfidfVectorizer(ngram_range=self.ngram_range,
                                    max_features=self.max_vocab_f2,
                                    strip_accents='unicode',
                                    token_pattern=self.token_pattern,
                                    min_df=self.min_df,
                                    decode_error='ignore')

                X2 = vec2.fit_transform(tokenized_papers2)
                X = hstack([X, X2])


                save(vec2, f"{self.env['store_misc']}/tfidf", f'{self.vectorizer_hash}_vec2', persist=True)
                self.vectorizer2 = vec2


            save(vec, f"{self.env['store_misc']}/tfidf", f'{self.vectorizer_hash}_vec', persist=True)
            save(X, f"{self.env['store_misc']}/tfidf", f'{self.vectorizer_hash}_X', persist=True)

        self.vectorizer = vec

        # discard fold ID column from labels
        review_groups = [col for col in y_train.columns if not col=='k']

        for review_group in tqdm(review_groups, desc='Train Review Groups'):

            # pull label column
            labels = y_train[review_group]

            #initiate random forest model
            classifier = RandomForestClassifier(n_estimators=self.n_estimators,
            max_depth=self.max_depth, max_features=self.max_features,
            min_samples_split=self.min_samples_split, min_samples_leaf=self.min_samples_leaf,
            random_state=self.seed, n_jobs=self.n_jobs).fit(X, labels)

            # save the model in dictionary of models
            self.models[review_group] = classifier

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

        tokenized_papers = list(papers[self.tokens_col])

        # get vectorizer and determine tfidf for papers
        vec = self.vectorizer
        X = vec.transform(tokenized_papers)

        if self.tokens_col2 is not None:
            tokenized_papers2 = papers[self.tokens_col2].apply(lambda x: np.str_(x))

            # get vectorizer and determine tfidf for papers
            vec2 = self.vectorizer2
            X2 = vec2.transform(tokenized_papers2)

            X = hstack([X, X2])

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
