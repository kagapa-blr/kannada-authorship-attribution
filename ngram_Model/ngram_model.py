import logging
import os
import pickle
import sys
import warnings

import nltk
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion
from sklearn.svm import SVC

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
log = logging.getLogger('ngram model')
log.setLevel(logging.DEBUG)
# read files
from imblearn.over_sampling import RandomOverSampler

nltk.download('punkt')
warnings.filterwarnings("ignore")

log.info("reading all authors files ...!")
script_dir = os.path.dirname(__file__)
rel_path = "../Dataset/All_Authors_dataset.csv"
abs_file_path = os.path.join(script_dir, rel_path)

df_instance = pd.read_csv(abs_file_path, encoding='utf-8')
del df_instance['Unnamed: 0']

X = df_instance["instances"]
y = df_instance["authors_clean"]

log.info("splitting trianing and testing for ngram model..")
# X, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3, random_state=456)

log.info("feature extracting from clean text")
one_word = TfidfVectorizer(analyzer="word", ngram_range=(1, 1), max_features=2000, binary=False).fit(X)
one_char = TfidfVectorizer(analyzer="char", ngram_range=(1, 1), max_features=2000, binary=False).fit(X)
two_word = TfidfVectorizer(analyzer="word", ngram_range=(2, 2), max_features=2000, binary=False).fit(X)
two_char = TfidfVectorizer(analyzer="char", ngram_range=(2, 2), max_features=2000, binary=False).fit(X)
three_word = TfidfVectorizer(analyzer="word", ngram_range=(3, 3), max_features=2000, binary=False).fit(X)
three_char = TfidfVectorizer(analyzer="char", ngram_range=(3, 3), max_features=2000, binary=False).fit(X)
four_word = TfidfVectorizer(analyzer="word", ngram_range=(4, 4), max_features=2000, binary=False).fit(X)
four_char = TfidfVectorizer(analyzer="char", ngram_range=(4, 4), max_features=2000, binary=False).fit(X)
five_word = TfidfVectorizer(analyzer="word", ngram_range=(5, 5), max_features=2000, binary=False).fit(X)
five_char = TfidfVectorizer(analyzer="char", ngram_range=(5, 5), max_features=2000, binary=False).fit(X)

log.info("calling each tf-idf vectors and union")
UnionFeater = FeatureUnion([
    ("word_1", one_word), ("word_2", two_word), ("word_3", three_word), ("word_4", four_word),
    ("word_5", five_word),
    ("char_1", one_char), ("char_2", two_char), ("char_3", three_char), ("char_4", four_char),
    ("char_5", five_char)
])

log.info("TF-IDF vectorizer...")
X_features = UnionFeater.transform(X)

ros = RandomOverSampler(random_state=42)
X_Rs, y = ros.fit_resample(X_features, y)

X_train, X_test, y_train, y_test = train_test_split(X_Rs, y, test_size=0.3, random_state=456)

log.info("RandomOverSampler completed\ntraining and testing splited\n")

log.info("Building SVM Model")
SVM = SVC(kernel='linear')
SVM = SVM.fit(X_train, y_train)

predictions = SVM.predict(X_test)
ngram_accuracy = SVM.score(X_test, y_test)

log.info("Accuracy saved in all_model_Acuuracy file : ")
saving_tofile = {"ngram_accuracy": ngram_accuracy}
with open("ngram_Model/ngram_model_Acuuracy.pkl", 'ab+') as acFile:
    pickle.dump(saving_tofile, acFile)

vector = UnionFeater.fit(X)
pickle.dump(vector, open('ngram_Model/ngram_vect.pkl', 'wb'))
pickle.dump(SVM, open('ngram_Model/ngram_attribution.pkl', 'wb'))
log.info("successfully created ngram model with SVM\n1.ngram vector file created\n2.ngram model saved with pickle")
