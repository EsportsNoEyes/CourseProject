import math
import nltk
from nltk.corpus import stopwords

# We apply the unigram model for naive bayes
def naiveBayes(train_set, train_labels, dev_set, smoothing_parameter):
    return do_unigram(train_set, train_labels, dev_set, smoothing_parameter)

# Apply unigram model and predict the labels of tweets
def do_unigram(train_set, train_labels, dev_set, smoothing_parameter):
    sarcasm_dict, not_sarcasm_dict, sarcasm_word_count, not_sarcasm_word_count = get_dicts(train_set, train_labels)
    sarcasm_p_dict = get_probability_dict(sarcasm_dict, sarcasm_word_count, smoothing_parameter)
    not_sarcasm_p_dict = get_probability_dict(not_sarcasm_dict, not_sarcasm_word_count, smoothing_parameter)
    result = []

    for tweet in dev_set:
        if (get_probability(tweet, sarcasm_p_dict) > get_probability(tweet, not_sarcasm_p_dict)):
            result.append(0)
        else:
            result.append(1)

    return result

# Calculate the sum of the probability of a word in a dictionary
def get_probability(tweet,p_dict):
    sum_p = 0
    for word in tweet:
        if (word in p_dict.keys()):
            sum_p=sum_p+p_dict[word]
        else:
            sum_p=sum_p+p_dict["does_not_appear"]

    return sum_p


# Calculate the probability of every single key in the SARCASM and NOT_SARCASM dictionary, including the words that we
# have not seen in the training data. We used the equations below to calculate the probability.
#
# P(UNK|D) = alpha / (n + alpha * (V + 1)), P(W|D) = (count(W) + alpha) / (n + alpha * (V + 1))
# UNK: the words that we have not seen in the training data
# D: the dictionary we used when we calculate the probability of its words. It can be either SARCASM dictionary or NOT_SARCASM dictionary
# alpha: the laplace smoothing parameter we set before training, default to be 1.0
def get_probability_dict(some_dict,word_count,smoothing_parameter):
    p_dict={}
    domain = len(some_dict.keys()) + 1
    for key in some_dict.keys():
        p = (some_dict[key]+smoothing_parameter)/(word_count+domain*smoothing_parameter)
        p_dict[key]=math.log(p)
    p_dict["does_not_appear"]=math.log(((0+smoothing_parameter)/(word_count+domain*smoothing_parameter)))

    return p_dict

# Create SARCASM dictionary and NOT_SARCASM dictionary respectively and store the count of the occurrences of all the 
# SARCARSM words and all the NOT_SARCASM words
def get_dicts(train_set, train_labels):
    sarcasm_dict={}
    not_sarcasm_dict={}
    sarcasm_word_count=0
    not_sarcasm_word_count=0
    stop_words = set(stopwords.words('english'))

    for i in range(0,len(train_set)):
        for word in train_set[i]:
            if (word.endswith("\n")):
                word = word[:-1]

            if (word.endswith("\r")):
                word = word[:-1]

            if (not word.isalpha()):
                continue

            if (word in stop_words):
                continue

            if (train_labels[i]==0):
                #sarcasm
                sarcasm_word_count=sarcasm_word_count+1
                if (word not in sarcasm_dict.keys()):
                    sarcasm_dict[word]=1

                else:
                    sarcasm_dict[word]=sarcasm_dict[word]+1

            else:
                #not_sarcasm
                not_sarcasm_word_count=not_sarcasm_word_count+1
                if (word not in not_sarcasm_dict.keys()):
                    not_sarcasm_dict[word]=1

                else:
                    not_sarcasm_dict[word]=not_sarcasm_dict[word]+1

    return sarcasm_dict,not_sarcasm_dict,sarcasm_word_count,not_sarcasm_word_count


