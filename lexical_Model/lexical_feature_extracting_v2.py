
import collections as coll
import math
import pickle
import string

import nltk
import numpy as np
import pandas as pd
from nltk.corpus import cmudict
from nltk.tokenize import sent_tokenize, word_tokenize

cmuDictionary = None
import os 
from os import path
import logging
import sys
import re

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
log = logging.getLogger('Lexical features extracting new version ....')
log.setLevel(logging.DEBUG)

log.info("importing libraries finished...")



path_of_Stopword="../Backend/Dataset/stopwords.txt"
file_path = path.relpath(path_of_Stopword)
file = open(file_path, encoding="utf8")
stop_words=file.read()
stopword = list(stop_words)
def Avg_wordLength(str):
    norm=[]
    str.translate(string.punctuation)
    tokens = word_tokenize(str)
    st = [",", ".", "'", "!", '"', "#", "$", "%", "&", "(", ")", "*", "+", "-", ".", "/", ":", ";", "<", "=", '>', "?",
          "@", "[", "\\", "]", "^", "_", '`', "{", "|", "}", '~', '\t', '\n']
    words = [word for word in tokens if word not in stopword and word not in st]
 
    return np.average([len(word) for word in words])


# ----------------------------------------------------------------------------


# returns avg number of characters in a sentence
def Avg_SentLenghtByCh(text):
    tokens = sent_tokenize(text)
    return np.average([len(token) for token in tokens])


# ----------------------------------------------------------------------------

# returns avg number of words in a sentence
def Avg_SentLenghtByWord(text):
    tokens = sent_tokenize(text)
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


# ----------------------------------------------------------------------------


# also returns Honore Measure R
def hapaxLegemena(text):
    words = RemoveSpecialCHs(text)
    V1 = 0
    # dictionary comprehension . har word kay against value 0 kardi
    freqs = {key: 0 for key in words}
    for word in words:
        freqs[word] += 1
    for word in freqs:
        if freqs[word] == 1:
            V1 += 1
    N = len(words)
    V = float(len(set(words)))
    if(N>0):
        R = 100 * math.log(N) / max(1, (1 - (V1 / V)))
        h = V1 / N
    else:
        R=0.00
    return R


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
    try:
        h = count / float(len(words))
        S = count / float(len(set(words)))
        t = S+h
    except:
        t=0.00
    return t


# ---------------------------------------------------------------------------

# c(w)  = ceil (log2 (f(w*)/f(w))) f(w*) frequency of most commonly used words f(w) frequency of word w
# measure of vocabulary richness and connected to zipfs law, f(w*) const rak kay zips law say rank nikal rahay hein
def AvgWordFrequencyClass(text):
    words = RemoveSpecialCHs(text)
    # dictionary comprehension . har word kay against value 0 kardi
    freqs = {key: 0 for key in words}
    for word in words:
        freqs[word] += 1
    try:
        maximum = float(max(list(freqs.values())))
    except:
        maximum = 0.00
    return np.average([math.floor(math.log((maximum + 1) / (freqs[word]) + 1, 2)) for word in words])


# --------------------------------------------------------------------------
# TYPE TOKEN RATIO NO OF DIFFERENT WORDS / NO OF WORDS
def typeTokenRatio(text):
    words = word_tokenize(text)
    return len(set(words)) / len(words)


# --------------------------------------------------------------------------
# logW = V-a/log(N)
# N = total words , V = vocabulary richness (unique words) ,  a=0.17
# we can convert into log because we are only comparing different texts
def BrunetsMeasureW(text):
    words = RemoveSpecialCHs(text)
    a = 0.17
    V = float(len(set(words)))
    N = len(words)
    try:
        B = (V - a) / (math.log(N))
    except:
        B=0.00
        
    return B


# ------------------------------------------------------------------------
def RemoveSpecialCHs(text):
    text = word_tokenize(text)
    st = [",", ".", "'", "!", '"', "#", "$", "%", "&", "(", ")", "*", "+", "-", ".", "/", ":", ";", "<", "=", '>', "?",
          "@", "[", "\\", "]", "^", "_", '`', "{", "|", "}", '~', '\t', '\n']

    words = [word for word in text if word not in st]
    return words


# -------------------------------------------------------------------------
# K  10,000 * (M - N) / N**2
# , where M  Sigma i**2 * Vi.
def YulesCharacteristicK(text):
    words = RemoveSpecialCHs(text)
    N = len(words)
    freqs = coll.Counter()
    freqs.update(words)
    vi = coll.Counter()
    vi.update(freqs.values())
    M = sum([(value * value) * vi[value] for key, value in freqs.items()])
    try:
        K = 10000 * (M - N) / math.pow(N, 2)
    except:
        K = 0.00
    return K


# -------------------------------------------------------------------------


# -1*sigma(pi*lnpi)
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
    # H = sum([(i/lenght)*math.log(i/lenght,math.e) for i in freqs.values()])
    return H


# ------------------------------------------------------------------
# 1 - (sigma(n(n - 1))/N(N-1)
# N is total number of words
# n is the number of each type of word
def SimpsonsIndex(text):
    words = RemoveSpecialCHs(text)
    freqs = coll.Counter()
    freqs.update(words)
    N = len(words)
    n = sum([1.0 * i * (i - 1) for i in freqs.values()])
    try:
        D = 1 - (n / (N * (N - 1)))
    except:
        D = 0.00
    return D


def Each_sentence(sequence):
    sequence = sent_tokenize(sequence)
    return sequence
def FeatureExtration(text):
    # cmu dictionary for syllables
    df = pd.DataFrame()
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
        
        #-----------new feature adding-----------
        h = hapaxLegemena(line)
        feature.append(h)
        
        B = BrunetsMeasureW(line)
        feature.append(B)
        
        Y = YulesCharacteristicK(line)
        feature.append(Y)
        
        S = ShannonEntropy(line)
        feature.append(S)
        
        SI =SimpsonsIndex(line)
        feature.append(SI)
        
        
        AWF = AvgWordFrequencyClass(line)
        feature.append(AWF)
        
        h = hapaxDisLegemena(line)
        feature.append(h)
        
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
# k------------------------------------------------------------------------------------------------


log.info("reading all authors files ...!")
script_dir = os.path.dirname(__file__)
rel_path = "../Dataset/All_Authors_dataset.csv"
abs_file_path = os.path.join(script_dir, rel_path)

df_instance=pd.read_csv(abs_file_path, encoding='utf-8')
del df_instance['Unnamed: 0']

X = df_instance["instances"]
y =df_instance["authors_clean"]

author_fvs={}
feature_vectors=[]
def creat_bin_file():
    for (text,author) in zip(X, y):
        clean_sentence = text_process(text=text)
        # author_fvs[author]=FeatureExtration(text)
        feature_vectors.append(FeatureExtration(text))
    pickle.dump(feature_vectors,open('lexical_features_new.pkl', 'wb'))
    print("successfully extracted all the features\n \n ")
    return feature_vectors

def UserInput(text):
    clean=text_process(text)
    return FeatureExtration(clean)
