from os import listdir
import numpy as np
import json
from nltk.stem.porter import PorterStemmer

porter_stemmer = PorterStemmer()
bad_words = {'aed','oed','eed'} # Remove words that will fail in nltk stemmer algorithm

# Helper to load training data and test data
def loadFile(name,stemming,sarcasm,training):
    count = 0
    text = []
    with open(name, 'rb') as f:
        for line in f:
            temp = []
            # Training data file
            if training:
                curr = json.loads(line)
                # Sarcasm
                if sarcasm and curr['label'] == 'SARCASM':
                    temp += (curr['response'].split())
                    for c in curr['context']:
                        temp = temp + c.split()
                # Not sarcasm
                elif not sarcasm and curr['label'] == 'NOT_SARCASM':
                    temp += (curr['response'].split())
                    for c in curr['context']:
                        temp = temp + c.split()
            # Test data file
            else:
                temp += (json.loads(line)['response'].split())
                for c in json.loads(line)['context']:
                    temp = temp + c.split()

            text.append(temp)

    # Stemming option, default false
    if stemming:
        for j in range(len(text)):
            for i in text[j]:
                if i in bad_words:
                    continue
                i = porter_stemmer.stem(i)

    count = count + 1

    return text

# Load data and form structures that can be used by naive bayes algorithm
def load_dataset(train_dir,dev_dir,stemming):
    X0 = loadFile(train_dir + 'train.jsonl',stemming, True, True)
    X1 = loadFile(train_dir + 'train.jsonl',stemming, False, True)
    X = X0 + X1
    Y = len(X0) * [0] + len(X1) * [1]
    Y = np.array(Y)

    X_test = loadFile(dev_dir + 'test.jsonl',stemming, True, False)

    return X,Y,X_test,[]