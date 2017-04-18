import os
def load_data(args):
    '''

    :param args: The args we caught with argparser
    :return: A tuple (documents,corpus) where
        documents is a dictionary whose keys are the filenames and values are their text
        corpus is a list of each "section" in each document, where section is each time a newline apears
    '''
    documents = {} # A dictionary whose keys are the filenames and values are their text
    corpus = [] # All of the documents, split into lines.

    for name in os.listdir(args.data_path):
        if name.endswith('.txt'):
            key = name[:-4] #truncate the .txt
            f_path = os.path.join(args.data_path, name)
            with open(f_path) as f:
                documents[key] = f.read()
                corpus += documents[key].splitlines()
    return documents,corpus
