from lexical_feature_extracting_v2 import UserInput
import pickle
from itertools import chain
import scipy
import pandas as pd
from scipy import sparse 
import sys
import os
#final_feature_vec=[]
# for i in range(len(vectors)):
#     final_feature_vec.append(list(chain.from_iterable(vectors[i])))
#features_matrix=scipy.sparse.csr_matrix(vectors)



root = os.path.dirname(__file__)
lexical_model_path = "..\lexical_Model"
lexi_model_file = os.path.join(root, lexical_model_path+"/lexical_attribution.pkl")

lexi_accuracy_file = os.path.join(root, lexical_model_path+"/Accuracy_saved.pkl")
def convertAndPredictAuthor(text):
    vectors = UserInput(text)
    features_matrix=scipy.sparse.csr_matrix(vectors)
    clf = pickle.load(open(lexi_model_file,'rb'))
    accuracy = pickle.load(open(lexi_accuracy_file,'rb'))
    predicted_author=clf.predict(features_matrix)
    return predicted_author[0], accuracy['Testing Accuracy']
    