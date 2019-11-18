import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

import sys
sys.path.append('../store/')

from persist import create_hash_id, check_persisted, load, save


class AdaBoost:

    """
    AdaBoost Classifier

    If TF-IDF flag is specified: computes vocabulary and TF-IDF vectors for the training data
    based on the specified parameters. These are used as the features to train the classifier.

    Persist parameters
    ==================
    fold_hash : str
        ID to be used to persist TF-IDF vectorizer or other model parts.

    AdaBoost parameters
    ===================
    base_estimator : sklearn classifier
    n_estimators : int
    learning_rate : float
    algorithm : str
    seed : int

    Feature parameters
    ==================
    tfidf : boolean
    ngram_range : tuple
    max_vocab_f : int
    tokens_col : str
    token_pattern : str

    Other parameters
    ================
    env : dict
    load_fresh : bool

    """

    def __init__(self, fold_hash, base_estimator=DecisionTreeClassifier(max_depth=1),
                 n_estimators=50, learning_rate=1.0, algorithm='SAMME.R', seed=None,
                 tfidf=False, ngram_range=3, max_vocab_f=75000, min_df=3,
                 tokens_col="tokens_baseline", token_pattern="default", env=None, load_fresh=False):

        # Persist parameters
        self.fold_hash=fold_hash

        # AdaBoost parameters
        self.base_estimator=base_estimator
        self.n_estimators=n_estimators
        self.learning_rate=learning_rate
        self.algorithm=algorithm
        self.seed=seed

        # Features parameters
        self.tfidf=tfidf
        self.ngram_range=(1, ngram_range)
        self.max_vocab_f=max_vocab_f
        self.min_df=min_df
        self.tokens_col=tokens_col
        self.token_pattern=token_pattern
        self.vectorizer_hash=create_hash_id(f'{self.fold_hash}{self.ngram_range}{self.max_vocab_f}{self.tokens_col}')

        # Models dictionary
        self.models = {}

        self.env = env
        self.load_fresh = load_fresh

    def train(self, x_train, y_train):
        """
        Trains classifier for each review group and stores
        it in a dictionary that is a class attribute.

        Parameters
        ==========
        x_train : pd.DataFrame
            Dataframe with columns corresponding to features to include in the model.
        y_train : pd.DataFrame
            DataFrame containing the labels for each paper. Each column represents one
            review group with binary labels.

        Returns
        =======
        None
        """
        ### preprocess ###
        if self.tfidf:

            # check if vectorizer has been created before, if so load from file
            if check_persisted(f"{self.env['store_misc']}/tfidf", f'{self.vectorizer_hash}_X', self.load_fresh):

                vec = load(f"{self.env['store_misc']}/tfidf", f'{self.vectorizer_hash}_vec')
                X = load(f"{self.env['store_misc']}/tfidf", f'{self.vectorizer_hash}_X')

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

                save(vec, f"{self.env['store_misc']}/tfidf", f'{self.vectorizer_hash}_vec', persist=True)
                save(X, f"{self.env['store_misc']}/tfidf", f'{self.vectorizer_hash}_X', persist=True)

            self.vectorizer = vec

        else:

            X = x_train

        # discard fold ID column from labels
        review_groups = [col for col in y_train.columns if not col=='k']

        # train
        for review_group in tqdm(review_groups, desc='Train Review Groups'):

            # pull label column
            labels = y_train[review_group]

            # creating parameters for xgboost
            classifier = AdaBoostClassifier(base_estimator=self.base_estimator,
                                            n_estimators=self.n_estimators,
                                            learning_rate=self.learning_rate,
                                            algorithm=self.algorithm,
                                            random_state=self.seed).fit(X, labels)

            # save classifier to class attribute
            self.models[review_group] = classifier

    def predict(self, papers):
        """
        Generates predictions from the trained classifiers.
        Each class's binary classifier is applied once.

        Parameters
        ==========
        papers : pd.DataFrame
            Dataframe with the papers we want to classify. Requires the same
            columns as were used in the train() method.

        Returns
        =======
        scores : pd.DataFrame
            Dataframe containing the predictions generated by each model.
            Each column corresponds to a review group and the values in
            that column are the probabilities that each paper belongs to
            that review group.
        """

        scores = {}

        if self.tfidf:

            tokenized_papers = list(papers[self.tokens_col])

            # get vectorizer and determine tfidf for papers
            vec = self.vectorizer
            X = vec.transform(tokenized_papers)

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
