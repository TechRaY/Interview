def get_top_words(model,topic_id, no_top_words):
    '''

    :param model: The scikit pipeline with an lda step and vectorizer step
    :param topic_id: The id of the topic we want to get words for
    :param no_top_words: Number of words
    :return: A list of the most important words for the topic in descending order
    '''
    topic = model.named_steps["lda"].components_[topic_id]
    feature_names = model.named_steps["count_vectorizer"].get_feature_names()
    top_words = [feature_names[i]
                 for i in topic.argsort()[:-no_top_words - 1:-1] #Sort the words by their importantce to topic
                 ]
    return top_words

def top_words_by_key(df,key,model,no_top_words):
    '''

    :param df: A dataframe whose columns are the input doc names and rows are the topics
    :param key: The name of the document we want to get top words for
    :param model: The scikit pipeline with an lda step and vectorizer step
    :param no_top_words:
    :return: A pandas Series of scores indexed by words.
    '''
    most_important_topic_id = df[key].argmax()
    top_words_list = get_top_words(model,most_important_topic_id,no_top_words)

    return top_words_list


