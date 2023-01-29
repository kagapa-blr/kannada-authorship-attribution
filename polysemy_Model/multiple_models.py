import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from collections import Counter
from sklearn.model_selection import train_test_split #split data into train and test sets
import pandas as pd
import warnings
from sklearn.metrics import classification_report
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import RandomOverSampler
import pickle
import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
log = logging.getLogger('polysemy Model')
log.setLevel(logging.DEBUG)

log.info("importing libraries finished...")
#read polysemy file
polysemy_data = pd.read_csv('polysemy_extracted.csv')
poly_emotion_df = polysemy_data
log.info("file reading finished...")
X, y = poly_emotion_df['clean_text'], poly_emotion_df['clean_authors']
original_Xdata = X.copy()
log.info("starting vectorization with TfidfVectorizer ")
vectorizer = TfidfVectorizer(analyzer='word', binary=False).fit(X)

X = vectorizer.fit_transform(X)

log.info("vectorization of the corpus finished...")

ros = RandomOverSampler(random_state=42)
X, y = ros.fit_resample(X, y)

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3, random_state=456)

log.info("RandomOverSampler completed\ntraining and testing splited\n")
model = ExtraTreesClassifier(n_estimators=100, random_state=0)
#model_Rand = RandomForestClassifier(max_depth=2, random_state=0)
model.fit(X_train, y_train)

log.info("ExtraTreesClassifier model created ")

# save the model to disk
filename_vect="polysem_vect.pkl"
x_data = vectorizer.fit(original_Xdata)
pickle.dump(x_data,open(filename_vect, 'wb'))

filename_model = 'poly_finalized_model.sav'
pickle.dump(model, open(filename_model, 'wb'))

log.info("Sucessfully created files:\n1.polysem_vect.pkl\n2.poly_finalized_model.sav\n")

poly_accuracy = model.score(X_test, y_test)
output={}
output['accuracy']=poly_accuracy
output['method']='Polysemy'
with open('accuracy.p', 'wb') as fp:
    pickle.dump(output, fp, protocol=pickle.HIGHEST_PROTOCOL)


# load the model from disk
# loaded_model = pickle.load(open(filename, 'rb'))
# result = loaded_model.score(X_test, y_test)

# log.info(result)
# pr = loaded_model.predict(X_test)

# log.info(classification_report(y_test, pr))

# this won't be run when imported
