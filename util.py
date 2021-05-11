import pandas as pd


def parse_file_A(data_file):
    """
    Reads a tab-separated sts benchmark file and returns
    texts: tweets (array of string)
    labels: labels of Task A (numpy array)
    """

    train = pd.read_csv(data_file, delimiter="\t")
    corpus = train['tweet'].to_numpy() #array
    labels = train['subtask_a'].to_numpy()
    labels[labels == 'OFF'] = 1
    labels[labels == 'NOT'] = 0

    labels = labels.astype(float) #np array

    return corpus,labels
def parse_file_B(data_file):
    """
        Reads a tab-separated sts benchmark file and returns
        texts: tweets (array of string)
        labels: labels of Task B (numpy array)
        """
    train = pd.read_csv(data_file, delimiter="\t")
    train_b = train.loc[train['subtask_b'].isin(['TIN', 'UNT'])]
    corpus = train_b['tweet'].to_numpy()
    labels = train_b['subtask_b'].to_numpy()
    labels[labels == 'TIN'] = 1
    labels[labels == 'UNT'] = 0
    labels = labels.astype(float)
    return corpus, labels




