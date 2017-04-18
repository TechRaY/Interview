import pandas as pd
import json
import os
from lda_pipeline import Ranker
from top_words import top_words_by_key


def task_1(lda_model, topic_df,args):
    '''
    Get a list of the n most important words
    :param lda_model: Scikit pipeline that vectorizes and does lda
    :param topic_df: A dataframe of documents/topics
    :param args:
    :return: the top num_words that are most important to our script
    '''
    top_words = top_words_by_key(df=topic_df, key="script", model=lda_model, no_top_words=args.num_words)
    save_path = os.path.join(args.save_path,'top_words_list.json')
    with open(save_path,'w') as f:
        json.dump(top_words,f)
    return top_words


def task_2(document_topic_dict,args):
    '''
    Compare the words in our list to all words in the corpus. Here we just create output, the topic breakdowns of
    each document
    :param document_topic_dict: A dictionary mapping documents to topic vectors
    :param args:
    :return: a dataframe whose columns are documents and rows are vectors
    '''
    topic_df = pd.DataFrame(document_topic_dict)
    ax = topic_df.T.plot.bar(figsize=(20,10),colormap='jet',title="Most important topics per document")
    fig = ax.get_figure()
    save_path = os.path.join(args.save_path,'task_2_top_topics.png')
    fig.savefig(save_path)

    return topic_df


def task_3(documents, top_words,args):
    '''
    Generate a score/rank for each word in our top n_words vs the transcriptions
    :param documents: The dicitonary of all documents for the entire task
    :param top_words: A list of the top words we found in task 1
    :param args:
    :return: a pandas series. The most important words to our script ranked by how much more we should be using them
    '''
    transcript_corpus = [val for key, val in documents.items() if key.startswith('transcript')]
    ranker = Ranker(documents=transcript_corpus)
    ranked_words_series, score_word_series = ranker.rank_words_in_doc(documents["script"], top_words)

    save_path = os.path.join(args.save_path,'ranked_words_list.json')
    ranked_words_series.to_json(save_path)
    ax =score_word_series.fillna(-1).sort_values(ascending=False).plot.bar(figsize=(20,10),title="Scoring of the words we should use more (Higher means more important")
    fig = ax.get_figure()
    save_path = os.path.join(args.save_path,'task_3_ranked_words.png')
    fig.savefig(save_path)

    return ranked_words_series