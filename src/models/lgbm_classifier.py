import sys
from collections import defaultdict

import lightgbm as lgb
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

sys.path.append('../store/')
from persist import create_hash_id, check_persisted, load, save


class LightGBM:

    """
    LightGBM Classifer

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

    LightGBM
    ========

    task : default value = train ; options = train , prediction ;
        Specifies the task we wish to perform which is either train or prediction.

    application: default=regression, type=enum, options= options :
        regression : perform regression task
        binary : Binary classification
        multiclass: Multiclass Classification
        lambdarank : lambdarank application

    num_iterations: number of boosting iterations to be performed ; default=100; type=int

    num_leaves : number of leaves in one tree ; default = 31 ; type =int

    device : default= cpu ; options = gpu,cpu. Device on which we want to train
    our model. Choose GPU for faster training.

    min_data_in_leaf: Min number of data in one leaf ; default=20 ; type=int

    feature_fraction: default=1 ; specifies the fraction of features to be
    taken for each iteration

    bagging_fraction: default=1 ; specifies the fraction of data to be used
    for each iteration and is generally used to speed up the training and
    avoid overfitting.

    min_gain_to_split: default=.1 ; min gain to perform splitting

    num_threads: default=OpenMP_default, type=int ;Number of threads for Light GBM.
    """

    def __init__(self, fold_hash, target=None, nthread=1,
        tfidf=False, ngram_range=3,
        max_vocab_f=75000, tokens_col="tokens_baseline",
        task="train", application="binary", num_iterations= 100,
        num_leaves=31, device="cpu", min_data_in_leaf=20, feature_fraction=1,
        bagging_fraction=1, min_gain_to_split=0.1, num_threads=0, max_depth=100, token_pattern='alpha',
        env=None, load_fresh=False):

        # identify which fold model was trained on
        self.fold_hash = fold_hash

        # model contents
        self.models = defaultdict(dict)
        self.vectorizer = None

        # parameters for tf-idf vectorizer
        self.tfidf = tfidf
        self.ngram_range = (1,ngram_range)
        self.max_vocab_f = max_vocab_f
        self.tokens_col = tokens_col
        self.token_pattern = token_pattern
        self.vectorizer_hash = create_hash_id(f'{self.fold_hash}{self.ngram_range}{self.max_vocab_f}{self.tokens_col}')

        # parameter for LightGBM
        self.task=task
        self.application=application
        self.num_iterations=num_iterations
        self.num_leaves=num_leaves
        self.device=device
        self.min_data_in_leaf=min_data_in_leaf
        self.feature_fraction=feature_fraction
        self.bagging_fraction=bagging_fraction
        self.min_gain_to_split=min_gain_to_split
        self.num_threads=num_threads
        self.max_depth=max_depth

        self.env = env
        self.load_fresh = load_fresh

    def train(self, x_train, y_train):

        """
        Trains one classifier per review group. Saves the trained
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
                #X = hstack([csr_matrix(x_train.drop(self.tokens_col, axis=1)),X])

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
                #X = hstack([csr_matrix(x_train.drop(self.tokens_col, axis=1)),X])



            self.vectorizer = vec

        else:
            X = x_train

        # discard fold ID column from labels
        review_groups = [col for col in y_train.columns if not col=='k']

        for review_group in tqdm(review_groups, desc='Train Review Groups'):

            # pull label column
            labels = y_train[review_group]

            # Create data structure for light gbm
            data_dmatrix = lgb.Dataset(data=X,label=labels)

            # creating parameters for light gbm
            params = {
            'task': self.task, 'application':self.application,
            'num_iterations':self.num_iterations, 'num_leaves':self.num_leaves,
            'device':self.device, 'min_data_in_leaf':self.min_data_in_leaf,
            'feature_fraction':self.feature_fraction, 'bagging_fraction':self.bagging_fraction,
            'min_gain_to_split':self.min_gain_to_split, 'num_threads':self.num_threads,
            'max_depth': self.max_depth, 'verbosity':-1
            }
            # light gbm
            self.models[review_group] = lgb.train(params, data_dmatrix)

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
            #X = hstack([csr_matrix(papers.drop(self.tokens_col, axis=1)),X])

        else:
            X = papers

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
