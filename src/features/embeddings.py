import numpy as np
from gensim.models import KeyedVectors
from tqdm import tqdm


def load_word2vec(path):
    """
    Loads a Word2Vec model using gensim.
    Parameters
    ==========
    path : str
        Local path to Word2Vec model binary file.
    Returns
    =======
    word2vec_model : gensim model object
        Word2Vec model.
    """
    word2vec_model = KeyedVectors.load_word2vec_format(path, binary=True)

    return word2vec_model


def compute_embeddings(papers, col='tokens_no_stopwords', word2vec_model_path='/data/w2v/PubMed-and-PMC-w2v.bin'):
    """
    Computes Word2Vec embeddings for a provided column of a dataframe with pre-trained model.
    If word does not exist in the model's vcoabulary,
    Parameters
    ==========
    papers : pd.DataFrame
        DataFrame where each row represents a paper.
    col : str
        Column of papers on which to compute embeddings.
        Typically the concatenated title and abstract.
    word2vec_model_path : str
        Local path to word2vec model binary file.
    
    Returns
    =======
    all_papers_text_embeddings : list
    """

    ### Load model ###
    word2vec_model = load_word2vec(word2vec_model_path)

    print("w2v model loaded")

    ### Compute embeddings for each row in papers ###
    papers_text_list = papers[col].tolist()
    all_papers_text_embeddings = []

    ### Loop through all papers ###
    for paper_text in tqdm(papers_text_list, desc="papers"):

        one_paper_text_embeddings = []

        ### Split up paper text into tokens ###
        for token in paper_text.split():

            ### If token exists in the model's vocabulary, get the embedding ###
            if token in word2vec_model.vocab:
                one_paper_text_embeddings.append(word2vec_model[token])

        one_paper_text_embeddings = np.average(np.array(one_paper_text_embeddings), axis=0)
        
        all_papers_text_embeddings.append(one_paper_text_embeddings)

    return all_papers_text_embeddings


def average_embeddings(papers):

    """
    Take the average embedding over title and abstract column
    """

    av_embeddings = []

    for embeddings in list(papers["all_embeddings"]):

        av_emb = np.average(np.array(embeddings), axis=0)

        av_embeddings.append(av_emb)

    return av_embeddings
