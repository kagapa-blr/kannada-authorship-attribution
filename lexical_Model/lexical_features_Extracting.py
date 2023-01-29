import collections as coll
import math
import pickle
import string
import numpy as np
from nltk.corpus import cmudict
import pandas as pd
from nltk.tokenize import word_tokenize, sent_tokenize
cmuDictionary = None
import os 
from os import path
import logging
import sys
import re
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
log = logging.getLogger('Lexical features extracting ....')
log.setLevel(logging.DEBUG)
from sklearn.model_selection import train_test_split
log.info("importing libraries finished...")


path_of_Stopword="..\Dataset\stopwords.txt"
file_path = path.relpath(path_of_Stopword)
file = open(file_path, encoding="utf8")
stop_words=file.read()
from nltk.tokenize import sent_tokenize
import nltk
#nltk.download('averaged_perceptron_tagger')
from sklearn.preprocessing import StandardScaler
# removing stop words plus punctuation.



def Avg_wordLength(str):
    norm=[]
    str.translate(string.punctuation)
    tokens = word_tokenize(str)
    st = [",", ".", "'", "!", '"', "#", "$", "%", "&", "(", ")", "*", "+", "-", ".", "/", ":", ";", "<", "=", '>', "?",
          "@", "[", "\\", "]", "^", "_", '`', "{", "|", "}", '~', '\t', '\n']
    stop = list(stop_words)
    words = [word for word in tokens if word not in stop and word not in st]
#     print(" words",words)
#     print([len(word) for word in words])
#     print(np.average([len(word) for word in words]))
    return np.average([len(word) for word in words])


# ----------------------------------------------------------------------------


# returns avg number of characters in a sentence
def Avg_SentLenghtByCh(text):
    tokens = sent_tokenize(text)
#     print(tokens)
#     print(np.average([len(token) for token in tokens]))
    return np.average([len(token) for token in tokens])


# ----------------------------------------------------------------------------

# returns avg number of words in a sentence
def Avg_SentLenghtByWord(text):
    tokens = sent_tokenize(text)
#     print(tokens)
#     print("Avg_SentLenghtByWord ",np.average([len(token.split()) for token in tokens]))
    return np.average([len(token.split()) for token in tokens])


# -----------------------------------------------------------------------------

# COUNTS SPECIAL CHARACTERS NORMALIZED OVER LENGTH OF CHUNK
def CountSpecialCharacter(text):
    st = ["#", "$", "%", "&", "(", ")", "*", "+", "-", "/", "<", "=", '>',
          "@", "[", "\\", "]", "^", "_", '`', "{", "|", "}", '~', '\t', '\n']
    count = 0
    for i in text:
        if (i in st):
            count = count + 1
    return count / len(text)


# ----------------------------------------------------------------------------

def CountPuncuation(text):
    st = [",", ".", "'", "!", '"', ";", "?", ":", ";"]
    count = 0
    for i in text:
        if (i in st):
            count = count + 1
    return float(count) / float(len(text))


# ---------------------------------------------------------------------------

def hapaxDisLegemena(text):
    words = RemoveSpecialCHs(text)
    count = 0
    # Collections as coll Counter takes an iterable collapse duplicate and counts as
    # a dictionary how many equivelant items has been entered
    freqs = coll.Counter()
    freqs.update(words)
    for word in freqs:
        if freqs[word] == 2:
            count += 1

    h = count / float(len(words))
    S = count / float(len(set(words)))
    return S, h


# ---------------------------------------------------------------------------

# c(w)  = ceil (log2 (f(w*)/f(w))) f(w*) frequency of most commonly used words f(w) frequency of word w
# measure of vocabulary richness and connected to zipfs law, f(w*) const rak kay zips law say rank nikal rahay hein
def AvgWordFrequencyClass(text):
    words = RemoveSpecialCHs(text)
    # dictionary comprehension . har word kay against value 0 kardi
    freqs = {key: 0 for key in words}
    for word in words:
        freqs[word] += 1
    maximum = float(max(list(freqs.values())))
    return np.average([math.floor(math.log((maximum + 1) / (freqs[word]) + 1, 2)) for word in words])


# --------------------------------------------------------------------------
# TYPE TOKEN RATIO NO OF DIFFERENT WORDS / NO OF WORDS
def typeTokenRatio(text):
    words = word_tokenize(text)
    return len(set(words)) / len(words)


# we can convert into log because we are only comparing different texts
def BrunetsMeasureW(text):
    words = RemoveSpecialCHs(text)
    a = 0.17
    V = float(len(set(words)))
    N = len(words)
    B = (V - a) / (math.log(N))
    return B


# ------------------------------------------------------------------------
def RemoveSpecialCHs(text):
    text = word_tokenize(text)
    st = [",", ".", "'", "!", '"', "#", "$", "%", "&", "(", ")", "*", "+", "-", ".", "/", ":", ";", "<", "=", '>', "?",
          "@", "[", "\\", "]", "^", "_", '`', "{", "|", "}", '~', '\t', '\n']

    words = [word for word in text if word not in st]
    return words


# Shannon and sympsons index are basically diversity indices for any community
def ShannonEntropy(text):
    words = RemoveSpecialCHs(text)
    lenght = len(words)
    freqs = coll.Counter()
    freqs.update(words)
    arr = np.array(list(freqs.values()))
    distribution = 1. * arr
    distribution /= max(1, lenght)
    import scipy as sc
    H = sc.stats.entropy(distribution, base=2)
    return H


# ------------------------------------------------------------------
def SimpsonsIndex(text):
    words = RemoveSpecialCHs(text)
    freqs = coll.Counter()
    freqs.update(words)
    N = len(words)
    n = sum([1.0 * i * (i - 1) for i in freqs.values()])
    D = 1 - (n / (N * (N - 1)))
    return D


# -----------------------------------------------------------------
def dale_chall_readability_formula(text, NoOfSectences):
    words = RemoveSpecialCHs(text)
    difficult = 0
    adjusted = 0
    NoOfWords = len(words)
    with open('dale-chall.pkl', 'rb') as f:
        fimiliarWords = pickle.load(f)
    for word in words:
        if word not in fimiliarWords:
            difficult += 1
    percent = (difficult / NoOfWords) * 100
    if (percent > 5):
        adjusted = 3.6365
    D = 0.1579 * (percent) + 0.0496 * (NoOfWords / NoOfSectences) + adjusted
    return D


# ------------------------------------------------------------------


def Each_sentence(sequence):
    sequence = sent_tokenize(sequence)
    return sequence
def FeatureExtration(text):
    # cmu dictionary for syllables
    global cmuDictionary
    cmuDictionary = cmudict.dict()
    All_sentences_per_instance = Each_sentence(text)
    vector = []
    for line in All_sentences_per_instance:
        feature = []
        meanwl = (Avg_wordLength(line))
        feature.append(meanwl)
        meansl = (Avg_SentLenghtByCh(line))
        feature.append(meansl)
        mean = (Avg_SentLenghtByWord(line))
        feature.append(mean)
        means = CountSpecialCharacter(line)
        feature.append(means)
        p = CountPuncuation(line)
        feature.append(p)
        TTratio = typeTokenRatio(line)
        feature.append(TTratio)
        vector.append(feature)
    return vector




def text_process(text):
    log.info("data cleaning processing ......")
    pp = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
    nopunct = "".join([char for char in text if char not in pp])
    tokens = nltk.word_tokenize(nopunct)
    nopunct =" ".join([word for word in tokens])
    clean=re.sub(r'[a-zA-Z]', '', nopunct)
    return clean

print("successfully extracted all the features\n \n ")

log.info("reading all authors files ...!")
script_dir = os.path.dirname(__file__)
rel_path = "../Dataset/All_Authors_dataset.csv"
abs_file_path = os.path.join(script_dir, rel_path)

df_instance=pd.read_csv(abs_file_path, encoding='utf-8')
del df_instance['Unnamed: 0']

X = df_instance["instances"]
y =df_instance["authors_clean"]

log.info("splitting trianing and testing for ngram model..")
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2, random_state=123)

author_fvs={}
feature_vectors=[]
def creat_bin_file():
    for (text,author) in zip(X, y):
        clean_sentence = text_process(text=text)
        # author_fvs[author]=FeatureExtration(text)
        feature_vectors.append(FeatureExtration(text))
    pickle.dump(feature_vectors,open('lexical_features.pkl', 'wb'))
    return feature_vectors,y
creat_bin_file()