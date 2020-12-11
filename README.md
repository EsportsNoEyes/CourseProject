# CS410 Classification Competition
 
## Usage
First clone the source code to local, and use this command to install all the dependencies `pip3 install -r requirements.txt`, assuming Python3 is used. 

Use command `python3 prediction.py` to generate the prediction results which is stored in `answer.txt`

If there is any errors with nltk package when running the code, please try to install suggested additional dependencies to solve the issue. We also welcome our reviewers to schedule a live demo.

## Description of Algorithm
As we planned in the project proposal, the first method we tried is classifier that based on Naive Bayes. The equation to compute the probability is ![eq1](https://github.com/EsportsNoEyes/CourseProject/blob/main/eq1.svg)

We used Laplace smoothing when we calculate the probability of every single key in the "SARCASM" and "NOT\_SARCASM" dictionary. The equation for calculating the probability is ![eq2](https://github.com/EsportsNoEyes/CourseProject/blob/main/eq2.svg) and ![eq3](https://github.com/EsportsNoEyes/CourseProject/blob/main/eq3.svg). UNK stands for the words that we have not seen in the training data. D stands for the dictionary we used when we calculate the probability of its words. It can be either "SARCASM" dictionary or "NOT\_SARCASM" dictionary. Alpha stnads for the laplace smoothing parameter we set before training, default to be 1.0. V stands for the size of the corresponding dictionary.

## Overview of Functions & Implementation Details
### reader.py, provide helpers to load the datasets into proper data structures to be used by the algorithm
**loadFile(name,stemming,sarcasm,training):** The helper function to load training data and test data. The parameter "name" indicates the directory path to the file of data. The parameter "sarcasm", a boolean variable, indicates whether the training data is labelled as "SARCASM" or "NOT_SARCASM". The parameter "training", a boolean variable, indicates whether the input data file is training or test. "stemming" is provided as an optional parameter to enable stemming. It returns a list containing the tweets.

**load\_dataset(train\_dir,dev\_dir,stemming):** It loads data and form structures that can be used by naive bayes algorithm using `loadFile()`. It returns lists indicating the label of each data entry.


### prediction.py, the wrapper file that is called by the user to run the our Naive Bayes Classifier
**main(args):** It is a wrapper function that extracts data, run naive bayes algorithm and output prediction results using the functions provided by `reader.py` and `naive_bayes.py`

### naive\_bayes.py, implementation of our Naive Bayes Classifier
**naiveBayes(train\_set, train\_labels, dev_set, smoothing\_parameter):** The wrapper function of our implemented Naive Bayes algorithm.

**do\_unigram(train\_set, trai\_labels, dev\_set, smoothing\_parameter):** It applies unigram model and predict the labels of tweets.

**get\_probability(tweet,p\_dict):** It calculates the sum of the probability of a word in a dictionary.

**get\_probability\_dict(some\_dict,word\_count,smoothing\_parameter):** It calculates the probability of every single key in the "SARCASM" and "NOT_SARCASM" dictionary, including the words that we have not seen in the training data. We used the equations mentioned in the above section to calculate the probability.

**get\_dicts(train\_set, train\_labels):** It creates "SARCASM" dictionary and "NOT\_SARCASM" dictionary respectively and store the count of the occurrences of all the "SARCARSM" words and all the "NOT_SARCASM" words

Based on the training data given, we first tried to predict the labels using the response tweets without context tweets as our dictionary. The accuracy was around 0.69. Then we included the context tweets into our dictionary. This time the accuracy beat the baseline. We tried to tune the laplace smoothing parameter and it turned out that the default one gave the highest. 

We also tried with stemming using the PorterStemmer of nltk package but it does not improve the accuracy considerably.


## Contribution of team members
We went through design of algorithms, coding, and documentation together.

 
  
 
