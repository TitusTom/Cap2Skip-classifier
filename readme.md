# Sentiment, Tone, and Nature Classifiction of Tweets

This script is used to classify labeled tweets using an SVM classifier. 

The following are the recommended classes for each label.
Sentiment : Postive, Neutral, Negative
Tone      : Passive, Active, Agressive
Nature    : Casual, Serious

Twitter dataset not provided due to it's sensitive nature. This script should work for any custom labeled twitter dataset.
## Prerequisites

1. CSV file of tweets in the following format:
   Input CSV file in the following format: Tweets, Label1, label 2, ...label N
2. The following python 2.7+ libraries:    
    1. Nltk
    2. Numpy
    3. Pandas
    4. ScikitLearn

## Getting Started

1. Change line #13 to match your input file.
2. Change line #15 to select which label to match.
3. change line #21 to =['tweets', 'label1', 'label2',... 'labeln'], skiprows=1)

## Running script
```
python PrepNclassify.py
```
## To Do

1. Add twitter data collection scripts.
