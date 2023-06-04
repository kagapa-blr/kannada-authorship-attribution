import logging
import pickle
import sys

import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split  # split data into train and test sets

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
log = logging.getLogger('polysemy Model')
log.setLevel(logging.DEBUG)

log.info("importing libraries finished...")
#read polysemy file
polysemy_data = pd.read_csv('polysemy_Model/polysemy_extracted.csv')

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
filename_vect="polysemy_Model/polysem_vect.pkl"
x_data = vectorizer.fit(original_Xdata)
pickle.dump(x_data,open(filename_vect, 'wb'))

filename_model = 'polysemy_Model/poly_finalized_model.sav'
pickle.dump(model, open(filename_model, 'wb'))

log.info("Sucessfully created files:\n1.polysem_vect.pkl\n2.poly_finalized_model.sav\n")

poly_accuracy = model.score(X_test, y_test)
output={}
output['accuracy']=poly_accuracy
output['method']='Polysemy'
with open('polysemy_Model/accuracy.p', 'wb') as fp:
    pickle.dump(output, fp, protocol=pickle.HIGHEST_PROTOCOL)


# load the model from disk
# loaded_model = pickle.load(open(filename, 'rb'))
# result = loaded_model.score(X_test, y_test)

# log.info(result)
#pr = model.predict(X_test)

# log.info(classification_report(y_test, pr))

# this won't be run when imported
#print(classification_report(y_test, pr))