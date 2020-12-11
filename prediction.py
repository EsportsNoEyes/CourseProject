import sys
import argparse
import configparser
import copy
import numpy as np
import json

import reader
import naive_bayes as nb

# Extract data, run naive bayes algorithm and output prediction results
def main(args):
    train_set, train_labels, dev_set,dev_labels = reader.load_dataset(args.training_dir,args.development_dir,args.stemming)
    predicted_labels = nb.naiveBayes(train_set,train_labels, dev_set, args.laplace)
    answer = open('answer.txt', 'w')
    test = open('./data/test.jsonl', 'r').readlines()
    for i in range(len(test)):
        answer.write(json.loads(test[i])['id'])
        answer.write(",")
        if predicted_labels[i] == 0:
            answer.write("SARCASM")
        else:
            answer.write("NOT_SARCASM")
        answer.write("\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CS410 Classification Competition')

    parser.add_argument('--training', dest='training_dir', type=str, default = './data/',
                        help='the directory of the training data')
    parser.add_argument('--development', dest='development_dir', type=str, default = './data/',
                        help='the directory of the development data')
    parser.add_argument('--stemming',default=False,action="store_true",
                        help='Use porter stemmer')
    parser.add_argument('--laplace',dest="laplace", type=float, default = 1.0,
                        help='Laplace smoothing parameter - default 1.0')
    args = parser.parse_args()
    main(args)