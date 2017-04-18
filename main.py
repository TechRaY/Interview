import os
from data_loader import load_data
from lda_pipeline import build_pipeline
from tasks import task_1, task_2, task_3


def main(args):
    document_topic_dict, documents, lda_model = preapre_data(args)
    topic_df = task_2(document_topic_dict, args)
    top_words = task_1(lda_model, topic_df, args)
    ranked_words_series = task_3(documents, top_words, args)
    return ranked_words_series ==1

def preapre_data(args):
    documents, lda_corpus = load_data(args)
    pipeline = build_pipeline(len(documents))
    lda_model = pipeline.fit(lda_corpus)
    document_topic_dict = {name: pipeline.transform([doc])[0] for name, doc in documents.items()}
    return document_topic_dict, documents, lda_model

def validate_args(args):
    if not os.path.exists(args.data_path):
        raise Exception("data path invalid {}".format(args.data_path))
    if (args.num_words) <= 0:
        raise Exception("Most specify a positive number of words. Got  {}".format(args.num_words))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description = 'Shows off data science skills')
    parser.add_argument("--num_words", help="The number of top words to output for task 1", type=int, default=20)
    parser.add_argument("--data_path", help="Where the data is located", type=str, default='./data')
    parser.add_argument("--save_path", help="Where to save results", type=str, default='./results')
    args = parser.parse_args()
    main(args)
