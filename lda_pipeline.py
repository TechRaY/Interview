'''
This file contains the code to prepare our corpus and extract topics.
We define our own tokenizer - Tokenizer

'''
import nltk
import re
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd
class Tokenizer():
    '''
    A utility tokenizer which we will pass to the vectorizer. Implements lematization from nltk and strips numbers.
    '''
    def __init__(self):
        self.lemma = nltk.stem.WordNetLemmatizer()
        self.tokenizer = nltk.RegexpTokenizer(r'\w+')
        self.reg = re.compile('\d+') # To strip numbers

    def proc_word(self, word):
        word = self.reg.sub('', word)
        word = self.lemma.lemmatize(word)

        return word

    def __call__(self, doc):
        tokenized= [self.proc_word(word) for word in self.tokenizer.tokenize(doc.lower())]
        no_short_words =list(filter(lambda x: len(x)>3,tokenized))
        return no_short_words

def build_pipeline(num_docs):
    '''

    :param args: Args passed to the command line via argparser
    :return: A pipeline that implements CountVectorizer and LDA with the args passed to argparser
    '''
    tf_vectorizer = CountVectorizer(
        max_df=0.95, #  Keep words that apear in only up to 95% of the documents (eg corpus specific stop words)
        min_df=5, # Only use words that apear in at least 5 documents
        stop_words='english',
        tokenizer=Tokenizer(), # Use our custom tokenizer
        ngram_range=(1, 3) #Use key words of length 1,2 or 3
    )

    lda = LatentDirichletAllocation(
        n_topics=num_docs * 2, # learn twice as many document as we have documents
        max_iter=6,
        learning_method='online',
        learning_offset=50.,
         random_state=0)
    pipeline = Pipeline([('count_vectorizer', tf_vectorizer), ('lda', lda)])

    return pipeline


class Ranker():
    '''
    Class to rank words in relation to our corpus based on our transcripts.
    We want to answer questions, which topics do our customers talk about that we just barely mention.
    To do this we will calculate an inverse log tfidf score. We'll get the document frequencies from the transcripts.
    The meaning of this score is "what words do our customers talk about a lot that we just barely mention"

    '''
    def __init__(self,documents,):
        '''

        :param documents: A list of strings, the documents to train on.
        '''
        corpus =[]
        for doc in documents:
            corpus += doc.splitlines()
        tfidf = TfidfVectorizer(tokenizer=Tokenizer(),stop_words='english',ngram_range=(1,3),smooth_idf=True)
        self.model = tfidf.fit(corpus)
        self.feature_names = tfidf.get_feature_names()

    def rank_words_in_doc(self,doc,words):
        '''
        :param doc: The document whose words we want to rank
        :param words: The particular words we are interested in
        :return: Two series, one of words ranked and one of their scores
        '''
        doc_tfidf_sparse = self.model.transform([doc]) #returns a sparse matrix
        doc_vector = doc_tfidf_sparse.toarray()[0] #turn it to an array
        ranking_dict = self._get_ranking(doc_vector)
        ranking_series = pd.Series(ranking_dict,)
        relevent_subset = ranking_series[words]
        ranked = relevent_subset.rank(ascending=False,method='first').sort_values()
        return ranked,relevent_subset

    def _get_ranking(self,doc_vector):
        ranking_dict = {self.feature_names[i]: -np.log(doc_vector[i]) #dictionary comprehension, key is word val is score
                   for i in doc_vector.argsort()  # iterate over indices sorted by the value
                   if doc_vector[i] >0 # and only take values that are non zero e.g. appear in the document
                   }
        return ranking_dict












