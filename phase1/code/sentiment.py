"""
Sentiment analysis script.
Has four polarity scorers
    - homerolled using sentiwordnet
    - vader
    - distilbert
    - roBERTa
quick test on ~20 sentences shows we should use vader and roBERTa.

Results are VADER and roBERTa looks great. other two not trustworty.
Plan is to use VADER and roBERTa. If they agree use average.
If they disagree, look manually.

"""

import os, sys
import pandas as pd
import numpy as np
sys.path.append("./code")

from .utils import *
import pyforest

import nltk
## corpora from nltk:
from nltk.corpus import stopwords
# from nltk.corpus import opinion_lexicon
from nltk.corpus import wordnet as wn
from nltk.corpus import sentiwordnet as swn
## classes from nltk
from nltk import sent_tokenize, word_tokenize, pos_tag
from nltk.sentiment import SentimentAnalyzer
from nltk.stem import WordNetLemmatizer
from nltk.sentiment import vader
from nltk.corpus import sentiwordnet as swn

from transformers import pipeline
from transformers import AutoModelForSequenceClassification
from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer
from scipy.special import softmax

os.environ["TOKENIZERS_PARALLELISM"] = "false" ## this turns off annoying warnings that get printed 1000 times...

# from string import punctuation


## worthwhile ideas
# https://nlpforhackers.io/sentiment-analysis-intro/

## SentiWordNet Scorer: cooking up sentiment scores from sentiwordnet:
stop_words = set(stopwords.words('english'))
wordnet_lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.replace("<br />", " ")
    text = text.decode("utf-8")
    return text


def penn_to_wn(tag):
    """
    Convert between the PennTreebank tags to simple Wordnet tags
    """
    if tag.startswith('J'):
        return wn.ADJ
    elif tag.startswith('N'):
        return wn.NOUN
    elif tag.startswith('R'):
        return wn.ADV
    elif tag.startswith('V'):
        return wn.VERB
    return None


def nltk_preprocess(s):
    """
    Input: s = string that is a sentence with words separated by space
    Output:
        stemmed = list of tuples of strings of form (w,p). w is the set of lemmatized words as above that are not stop words or punctuation, and have been lowercased. p is part of speech in wordnet format. if p == None, they are tossed out.
    """
    words = word_tokenize(s) ## now a single list of word tokens
    words
    pos = pos_tag(words) ## part of speech tags added, not list of tuples.
    words = list(map(lambda t: (t[0],penn_to_wn(t[1])) , pos ))
    words = [(w.lower(), p) for (w,p) in words if w.isalnum() and  w not in stop_words and p] ## remove punctuation, remove stopwords and make lower case (noncommutative). remove stopwords, then what's left make lower, else I'll lose "CAN":
    stemmed = [ (wordnet_lemmatizer.lemmatize(w, pos = p), p)  for (w,p) in words ]    ### lemmatize: ## Note stemming is poor mans verion (just cuts off endings) Lemmatizing is "done right"--uses a well defined mapping for most words to their stem word.
    return stemmed


def synset_polarity(s):
    """
    Returns a score in [-1, 1]
    tokenizes, pos-tags, throws out stopwords/words that don't have POS tag in wordnet. Lemmatizes
    For each word get's every synset and averages their polarity scores to get a (latent) word score
    returns averages the words scores
    """
    stemmed = nltk_preprocess(s)
    scores = {} ## for every word, get every synset and average their polarity scores in sentiwordnet
    for w,p in stemmed:
        synsets = wn.synsets(w, pos = p)
        scores[w]=np.average([swn.senti_synset(x.name()).pos_score() - swn.senti_synset(x.name()).neg_score() for x in synsets])
    ## return average of the polarity scores of all the words in the sentence
    return  np.average([x for x in scores.values() if not np.isnan(x)])



## VADER Scorer:
SIA =  vader.SentimentIntensityAnalyzer()

def vader_polarity(text):
    """ Returns a score in [-1,1] using the "compound" output of VADER"""
    return SIA.polarity_scores(text)["compound"]

## The Compound score is a metric that calculates the sum of all the lexicon ratings which have been normalized between -1(most extreme negative) and +1 (most extreme positive)
## If you use the VADER sentiment analysis tools, please cite:
## Hutto, C.J. & Gilbert, E.E. (2014). VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text. Eighth International Conference on Weblogs and Social Media (ICWSM-14). Ann Arbor, MI, June 2014.


# distillbert
## the  “distilbert-base-uncased-finetuned-sst-2-english” pretrained model.
## Huggingface transformers default sentiment analyzer.
## it uses the https://huggingface.co/transformers/model_doc/distilbert.html

DBUFclassifier = pipeline('sentiment-analysis')
# DBUFclassifier("Text goes here.")

def hf_preprocess(text):
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)


def dBERT_polarity(s):
    """
    “distilbert-base-uncased-finetuned-sst-2-english” pretrained model, the default sentiment analyzer in huggingface transformers
    Linearly maps the output (0,1) to [-1,1] and returns it
    """
    hf_preprocess(s)
    d = DBUFclassifier(s)[0]
    if d['label'] == "POSITIVE":
        return 2*(d["score"]-.5)
    return 2*((1-d["score"]) - .5 )



# Twitter-roBERTa-base for Sentiment Analysis https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment?text=We+hope+you+don%27t+hate+it.
## another model but using the huggingfact transformers classes.

task='sentiment'
MODEL = f"cardiffnlp/twitter-roberta-base-{task}"

# roBERTa = pipeline('sentiment-analysis', model='cardiffnlp/twitter-roberta-base-sentiment')
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

def roBERTa_polarity(text):
    """
    Returns overall score in [-1,1]
    model returns three scores that sum to 1 corresponding with labels ['negative', 'neutral', 'positive']
    we take the weighted sum of these to get a [-1,1] score.
    """
    text = hf_preprocess(text)
    encoded_input = tokenizer(text, return_tensors='pt')
    output = model(**encoded_input)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    scores
    return np.dot(scores, np.array([-1,0,1]))


if __name__ == '__main__':

    ## some test examples
    sentences = [
        "This really long sentence is interestingly boring, which seems oxymoronical, but its not!",
        "This sentence is terrible and horrible and negative!",
        "This sentence is not terrible or horrible or negative.",
        "This is a neutral sentence.",
        "I am testing this not great, in fact, terrible sentiment analyzer.",
        "I am testing this not terrible, in fact, great sentiment analyzer.",
        'I am testing this seemingly awesome, super cool package!',
        "i am testing this seemingly awesome super cool package",
        "I went to the store and they were out of potatoes.",
        "I went to the store.",
        "i think this could be useful",
        "I think this could be useful!",
        "We are very happy to show you the transformers library.",
        "We hope you don't hate it.",
        "we hope you don't hate it",
        "We hope you don't hate it too much.",
        "The tool looks useful.",
        "I think the threat intelligence piece would be very nice.",
        "The UI seems hard to use.",
        "The UI seems really hard to learn to use effectively.",
        "I liked the rankings it provides, but the automation looks shoddy."]

    for i,s in enumerate(sentences):
        print(f"{i, s}\n\tSynsetScore:{synset_polarity(s)}\n\tVADER:{vader_polarity(s)}\n\tdistillBERT:{dBERT_polarity(s)}\n\troBERTa:{roBERTa_polarity(s)}")
