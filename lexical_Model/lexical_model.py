import logging
import os.path
import pickle
import sys
from itertools import chain

import pandas as pd
import scipy
from imblearn.over_sampling import RandomOverSampler
from scipy import sparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
log = logging.getLogger('Lexical Model')
log.setLevel(logging.DEBUG)
log.info("All required libraries imported...")
script_dir = os.path.dirname(__file__)
log.info("reading original authors data")
rel_path = "../Dataset/All_Authors_dataset.csv"
abs_file_path = os.path.join(script_dir, rel_path)

df_instance=pd.read_csv(abs_file_path, encoding='utf-8')
del df_instance['Unnamed: 0']

log.info("reading feature vector file ..")
lexical_features_vec = pickle.load(open("../lexical_Model/lexical_features_new.pkl",'rb'))

final_feature_vec=[]
for i in range(len(lexical_features_vec)):
    final_feature_vec.append(list(chain.from_iterable(lexical_features_vec[i])))
features_matrix=scipy.sparse.csr_matrix(final_feature_vec)

log.info("successfully converted to csr/sparse matrix ...")
X=features_matrix
y =df_instance["authors_clean"]

log.info("data oversampling with :RandomOverSampler")
ros = RandomOverSampler(random_state=42)
X, y = ros.fit_resample(X, y)

log.info("training and testing  split")
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2, random_state=123)
#building model

log.info("Model Buildug Started...")

model = RandomForestClassifier()  
model = model.fit(X_train, y_train)

train_acc=model.score(X_train, y_train)
test_acc=model.score(X_test, y_test)

print("Result")
print("Taining Accuracy :",train_acc)
print("Testing Accuracy: ",test_acc)


pickle.dump(model, open('lexical_attribution.pkl','wb'))
store_acc={"Training Accuracy":train_acc, "Testing Accuracy":test_acc}
pickle.dump(store_acc, open('Accuracy_saved.pkl','wb'))
#predictions = model.predict(X_test)
#print(classification_report(y_test, predictions))