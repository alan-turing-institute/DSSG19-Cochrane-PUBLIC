import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

sys.path.append('../store/')
sys.path.append('../features')
from persist import create_hash_id, check_persisted, load, save
from embeddings import load_word2vec


class RandomForestWeightedEmbeddings:

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
    env : dict
        Dict containing filepaths. 

    ngram_range : set
        Defines the upper and lower bound of the ngrams that should be considered.
        Defaults to (1, 1).

    max_vocab_f : int
        The maximum size of the vocabulary, i.e. the maximum length of the tf-idf
        vector that is created for an paper. Defaults to 75000.
    min_df : int
        The minimum number of documents that a feature should exist in to be included
        in a vectorizer.

    n_estimators: int
        The number of trees to be trained
    max_depth: int
        The maximum number of branches to employ for each tree
    max_features: int
        The maximum number of features to consider at each split
    min_samples_split: int
        The minimum number of data points there must be per split
    min_samples_leaf: int
        The minimum number of samples there must be per leaf

    tokens_col : str
        The column from which the token features should be extracted (title + abstract)
    load_fresh : bool
        Whether persisted files should be used (False) or not (True). 
    """

    def __init__(self, fold_hash, seed, env, ngram_range=(1,1), max_vocab_f=75000,
        min_df=3, n_estimators=10, max_depth=None, max_features='auto', n_jobs=8,
                    min_samples_split=2, min_samples_leaf=1, tokens_col = "tokens_no_stopwords",
                    load_fresh=False):

        # identify which fold model was trained on
        self.fold_hash = fold_hash

        # model contents
        self.models = {}
        self.vectorizer = None

        # parameters for tf-idf vectorizer
        self.ngram_range = ngram_range
        self.max_vocab_f = max_vocab_f
        self.min_df = min_df
        self.tokens_col = tokens_col

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
            classify, with a column corresponding to the tokens to use.

        y_train : pandas DataFrame
            DataFrame containing the labels for each paper. Each
            column represents one review group with binary labels.
        """

        # check if vectorizer has been created before, if so load from file
        if check_persisted(f"{self.env['store_misc']}/tfidf", f'{self.vectorizer_hash}_X', self.load_fresh):

            vec = load(f"{self.env['store_misc']}/tfidf", f'{self.vectorizer_hash}_vec')
            X = load(f"{self.env['store_misc']}/tfidf", f'{self.vectorizer_hash}_X')

        else:

            # get the tokenized papers
            tokenized_papers = list(x_train[self.tokens_col])

            vec = TfidfVectorizer(max_features=self.max_vocab_f,
                                    strip_accents='unicode')

            # generate term document matrix (model inputs)
            X = vec.fit_transform(tokenized_papers)

            save(vec, f"{self.env['store_misc']}/tfidf", f'{self.vectorizer_hash}_vec', persist=True)
            save(X, f"{self.env['store_misc']}/tfidf", f'{self.vectorizer_hash}_X', persist=True)

        self.vectorizer = vec

        if check_persisted(f"{self.env['store_misc']}/embeddings", f'{self.vectorizer_hash}_X', self.load_fresh):
            weighted_embeddings = load(f"{self.env['store_misc']}/embeddings", f'{self.vectorizer_hash}_X')

        else:
            self.embeddings_model = load_word2vec(self.env['word2vec_model'])
            weighted_embeddings = np.array(self.create_embeddings(X, vec))
            save(weighted_embeddings, f"{self.env['store_misc']}/embeddings", f'{self.vectorizer_hash}_X', persist=True)

        
        # discard fold ID column from labels
        review_groups = [col for col in y_train.columns if not col=='k']

        for review_group in tqdm(review_groups, desc='Train Review Groups'):

            # pull label column
            labels = y_train[review_group]

            #initiate random forest model
            classifier = RandomForestClassifier(n_estimators=self.n_estimators,
            max_depth=self.max_depth, max_features=self.max_features,
            min_samples_split=self.min_samples_split, min_samples_leaf=self.min_samples_leaf,
            random_state=self.seed, n_jobs=self.n_jobs).fit(weighted_embeddings, labels)
            
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

        # get embeddings for papers
        if check_persisted(f"{self.env['store_misc']}/embeddings", f'{self.vectorizer_hash}_y', self.load_fresh):
            weighted_embeddings = load(f"{self.env['store_misc']}/embeddings", f'{self.vectorizer_hash}_y')

        else:
            self.embeddings_model = load_word2vec(self.env['word2vec_model'])
            vec = self.vectorizer
            X = vec.transform(tokenized_papers)
            weighted_embeddings = np.array(self.create_embeddings(X, vec))
            save(weighted_embeddings, f"{self.env['store_misc']}/embeddings", f'{self.vectorizer_hash}_y', persist=True)
        

        for model_group in tqdm(self.models, desc='Test Review Groups'):

            # get the classifier
            classifier = self.models[model_group]

            # predictions as probabilities
            y_preds = classifier.predict_proba(weighted_embeddings)

            probabilities = y_preds[:,1]

            # store scores of model
            scores[model_group] = probabilities

        scores = pd.DataFrame.from_dict(scores)

        return scores

    def create_embeddings(self, X, vec):

        weighted_embeddings = []

        # get the names of the features
        feature_names = vec.get_feature_names()

        # loop over tfidf matrix
        for doc in tqdm(range(X.shape[0]), desc='emb'):
            
            # zip nonzero features and their scores
            feature_index = X[doc,:].nonzero()[1]
            tfidf_scores = zip(feature_index, [X[doc, x] for x in feature_index])

            one_paper_text_embeddings = []
            
            for (i, s) in tfidf_scores:
                
                # weight embedding by tfidf score
                if feature_names[i] in self.embeddings_model.vocab:
                    one_paper_text_embeddings.append(self.embeddings_model[feature_names[i]] * s)
                    
            # average the weighted embeddings
            one_paper_text_embeddings = np.average(np.array(one_paper_text_embeddings), axis=0)
                
            # store the weighted embedding
            weighted_embeddings.append(one_paper_text_embeddings)

        return weighted_embeddings

