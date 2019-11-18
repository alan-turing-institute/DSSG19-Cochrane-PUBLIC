import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import *
from nltk.stem import WordNetLemmatizer
import re
nltk.download('wordnet')


def tokenize_title_abstract_baseline(papers, title='ti', abstract='ab'):

    """
    Tokenizes the title and abstract of a paper consistent with the
    approach of the model currently in place at Cochrane.

    Title words are prepended with "TI_"; abstract words with "AB_". Stopwords
    are removed from the text. Follows the approach of the model currently in
    place at Cochrane.

    Parameters
    ----------
    papers : DataFrame
        DataFrame where each row represents a paper. Required columns are
        title and abstract.
    title : str
        Column name of title.
    abstract : str
        Column name of abstract.

    Returns
    -------
    all_tokens : list
    """
    # stopwords from nltk
    stopw = set(stopwords.words('english'))

    # transform titles and abstracts to list (move outside of pandas for speed)
    titles_list = papers[title].tolist()
    abs_list = papers[abstract].tolist()

    # list to store final tokens
    all_tokens = []

    regex = re.compile('[^a-zA-Z]')

    # transform title and abstracts to tokens (one by one)
    for title, ab in zip(titles_list, abs_list):

        # split into strings
        try:
            title = title.split()
        except:
            title = ['']

        try:
            ab = ab.split()
        except:
            ab = ['']

        # prepending and discarding stopwords
        title_tokens = ["TI_" + regex.sub('', t) for t in title if t not in stopw]
        ab_tokens = ["AB_" + regex.sub('', a) for a in ab if a not in stopw]

        # combine into lists, then string
        combined_token_lists = title_tokens + ab_tokens
        combined_tokens = " ".join(combined_token_lists)

        # store in list of all tokens
        all_tokens.append(combined_tokens)


    return all_tokens


def tokenize_title_abstract(papers, title='ti', abstract='ab', to_lower=False, remove_stopwords=True):

    """
    Tokenizes the title and abstract of a paper in a more flexible way.

    Parameters
    ----------
    papers : DataFrame
        DataFrame where each row represents a paper.
        Required columns are title and abstract.
    title : str
        Column name of title.
    abstract : str
        Column name of abstract.
    to_lower: boolean
        Whether to convert words to lower case. Defaults to False.
    remove_stopwords : boolean
        Whether to remove stopwords. Defaults to True.

    Returns
    -------
    all_tokens : list

    """

    if remove_stopwords:
        stopw = set(stopwords.words('english'))

    # transform titles and abstracts to list (move outside of pandas for speed)
    titles_list = papers[title].tolist()
    abs_list = papers[abstract].tolist()

    # list to store final tokens
    all_tokens = []

    regex = re.compile('[^a-zA-Z0-9.,-]')


    # transform title and abstracts to tokens (one by one)
    for title, ab in zip(titles_list, abs_list):

        # split into strings
        try:
            title = title.split()
        except:
            title = ['']

        try:
            ab = ab.split()
        except:
            ab = ['']

        # discarding stopwords, if specified, and removing non-alphanumeric tokens
        if remove_stopwords:
            title_tokens = [regex.sub('', t) for t in title if t not in stopw]
            ab_tokens = [regex.sub('', a) for a in ab if a not in stopw]

        else:
            title_tokens = title
            ab_tokens = ab

        # combine into lists, then string
        combined_token_lists = title_tokens + ab_tokens
        combined_tokens = " ".join(combined_token_lists)

        # store in list of all tokens
        all_tokens.append(combined_tokens)

    return all_tokens


def create_vocabularies(X_train, y_train, ngram_range=(1,3), max_features=75000,
                    min_df=3):

    """
    Creates a vocabulary from each of the paper collections that is provided.

    From the tokenized (through tokenize_title_abstract()) title/abstracts,
    calculates the tf-idf scores of each word, and takes the features with the
    highest score. This is performed for each element of the dictionary provided,
    i.e. each element should represent a collection of papers from which a
    vocabulary should be created.

    Parameters
    ----------

    X_train : pandas DataFrame

    y_train : pandas DataFrame

    ngram_range : tuple
        Defines which n-grams should be considered. Defaults to (1, 3), meaning
        that 1-grams, 2-grams and 3-grams are considered.

    max_features : int
        Defines the maximum number of features that will be returned for each
        vocabulary. The features with the highest tf-idf scores are considered.
        Defaults to 75000.

    min_df : int
        Defines the minimum number of documents a feature should exist in to be
        considered for the vocabulary.

    Returns
    -------

    vocabularies : dict
        Dictionary contains the created vocabularies as lists. Key: id (of a Review
        Group), Value: list.

    """

    vocabularies = {}

    groups = [col for col in y_train.columns if not col=='k']
    for review_group in groups:

        print(X_train.head())
        print(y_train.head())
        print(len(y_train[review_group] == 1.0))
        print(review_group)
        print(X_train.shape)
        print(y_train.shape)

        vocab_data = X_train[y_train[review_group] == 1.0]

        # tokenize the text
        full_vocabulary = list(vocab_data["tokens"])

        # get the ngrams with the top-75000 tfidf score - those are vocabulary
        vec = TfidfVectorizer(ngram_range=ngram_range,
                                max_features=max_features,
                                min_df=min_df, strip_accents='unicode')

        term_doc_matrix = vec.fit_transform(full_vocabulary)
        vocabulary = vec.get_feature_names()

        vocabularies[review_group] = vocabulary

    return vocabularies


def lemmatize(papers, cols):

    """
    Function to take in the papers and the columns of interest and lemmatize
    all of the tokens 
    
    Parameters:
    ==========
    papers : pd.Dataframe 
        A dataframe of all of the papers along with their data of interest
        
    cols : list
        list of the columns to be considered for lematization 
    Returns: 
    
    papers : a pandas dataframe with lematized columns of interest
    ========
    """

    lemmatizer = WordNetLemmatizer()

    # lemmatize all specified columns
    for col in cols:

        # turn column into list (for speed, rather than pandas apply)
        column_list = list(papers[col])

        # new list to add lemmatized phrases to
        column_list_lemmatized = []

        for paper in column_list:

            # lemmatize phrase and add to list
            lemmatized_paper = lemmatize_phrase(paper, lemmatizer)
            column_list_lemmatized.append(lemmatized_paper)

        # store lemmatized column in dataframe
        papers[col + '_lemmatized'] = column_list_lemmatized

    return papers


def lemmatize_phrase(phrase, lemmatizer):
    """
    Function to turn a phrase into a lemma 
    
    Parameters:
    ===========
    phrase : string
        a string to be turned into a lemma
        
    lemmatizer : lematizing function 
        a lematizer function 
    
    """

    split_phrase = phrase.split(" ")

    # lemmatizes word and assumes it is a verb
    lemmad_split_phrase = [lemmatizer.lemmatize(word, "v") for word in split_phrase]

    lemmad_phrase = " ".join(lemmad_split_phrase)

    return lemmad_phrase


def stemming(papers, col='tokens_no_stopwords'):

    """
    Reducing words to their root form (stem).

    Parameters
    ----------

    papers : DataFrame
        DataFrame containing the columns that stemming should be applied to.

    cols : list
        List of the columns (in papers) that stemming should be applied to.

    Returns
    -------

    papers : DataFrame
        DataFrame identical to the one provided as input, with in addition the
        stemmed columns (original column name + "_stemmed").
    """

    stemmer = PorterStemmer()

    # turn column into list (for speed, rather than pandas apply)
    column_list = list(papers[col])

    # new list to add stemmed phrases to
    column_list_stemmed = []

    for paper in column_list:

        # stem phrase and add to list
        stemmed_paper = stem_phrase(paper, stemmer)
        column_list_stemmed.append(stemmed_paper)

    return column_list_stemmed


def stem_phrase(phrase, stemmer):

    """
    Helper function to create stemmed version of a phrase.

    Parameters
    ----------

    phrase : string
        Phrase to be stemmed.

    stemmer : nltk stemmer object

    Returns
    -------

    stemmed_phrase : string

    """

    split_phrase = phrase.split(" ")

    stemmed_split_phrase = [stemmer.stem(word) for word in split_phrase]

    stemmed_phrase = " ".join(stemmed_split_phrase)

    return stemmed_phrase
