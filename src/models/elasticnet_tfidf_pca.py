import sys
from collections import defaultdict

import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from tqdm import tqdm

sys.path.append('../store/')
from persist import create_hash_id, check_persisted, load, save


class ElasticClassifierTfidfPCA:

    """
    Elastic net classifer

    One-vs-all logistic regression. The class contains one model for each review
    group, and returns the probability that an paper belongs to each of the
    review groups.

    First calculates TF-IDF and then does a dimensionality reduction, after which
    classification happens using a logistic regression with elasticnet regularization.

    Parameters
    ----------
    
    fold_hash : str
        Unique identifier for the string.

    seed : int 
        Seed used for PCA.

    ngram_range : set
        Defines the upper and lower bound of the ngrams that should be considered.
        Defaults to (1, 2).

    max_vocab_f : int
        The maximum size of the vocabulary, i.e. the maximum length of the tf-idf
        vector that is created for an paper. Defaults to 75000.

    alpha : float
        Constant that is used in the regularization (l2). Defaults to 0.0001.

    l1_ratio : float
        Constant that is used int he regularization (l1). Defaults to 0.0001.

    min_df : int
        Minimum number of documents that a feature should be in to be included in 
        a vectorizer's vocabulary.

    tokens_col : str
        Column from which tokens should be pulled.

    n_components : int
        Number of components that should be kept.

    env : dict
        Dict containing filepaths.
        
    load_fresh : bool
        Whether persisted files should be used (False) or not (True).    
    
    """

    def __init__(self, fold_hash, seed, ngram_range=(1,2), max_vocab_f=10000,
        alpha=0.0001, l1_ratio=0.15, min_df=3, tokens_col="tokens_baseline", n_components=1000,
        env=None, load_fresh=False):

        # identify which fold model was trained on
        self.fold_hash = fold_hash

        # model contents
        self.models = defaultdict(dict)
        self.pca = None

        # parameters for tf-idf vectorizer
        self.ngram_range = ngram_range
        self.max_vocab_f = max_vocab_f
        self.min_df = min_df
        self.tokens_col = tokens_col
        self.vectorizer = None
        self.vectorizer_hash = create_hash_id(f'{self.fold_hash}{self.ngram_range}{self.max_vocab_f}{self.min_df}{self.tokens_col}')

        # parameter for logistic regression with elastic net
        self.alpha = alpha
        self.l1_ratio = l1_ratio

        # parameter for pca
        self.n_components = n_components
        self.seed = seed

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
        if check_persisted(f"{self.env['store_misc']}/pca", f'{self.vectorizer_hash}_X', self.load_fresh):

            pca = load(f"{self.env['store_misc']}/pca", f'{self.vectorizer_hash}_pca')
            vec = load(f"{self.env['store_misc']}/pca", f'{self.vectorizer_hash}_vec')
            X = load(f"{self.env['store_misc']}/pca", f'{self.vectorizer_hash}_X')

        else:

            # get the tokenized papers
            tokenized_papers = list(x_train[self.tokens_col])

            vec = TfidfVectorizer(ngram_range=self.ngram_range,
                                    max_features=self.max_vocab_f,
                                    strip_accents='unicode')

            # generate term document matrix (model inputs)
            X_tfidf = vec.fit_transform(tokenized_papers)

            # reduce dimensionality of tf-idf features through PCA
            pca = TruncatedSVD(n_components=self.n_components, random_state=self.seed)
            X = pca.fit_transform(X_tfidf)

            save(pca, f"{self.env['store_misc']}/pca", f'{self.vectorizer_hash}_pca', persist=True)
            save(vec, f"{self.env['store_misc']}/pca", f'{self.vectorizer_hash}_vec', persist=True)
            save(X, f"{self.env['store_misc']}/pca", f'{self.vectorizer_hash}_X', persist=True)

        self.pca = pca
        self.vectorizer = vec

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

        tokenized_papers = list(papers[self.tokens_col])

        # get pca and transform papers
        pca = self.pca
        vec = self.vectorizer
        X_tfidf = vec.transform(tokenized_papers)
        X = pca.transform(X_tfidf)

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
