import warnings

import nltk
import pandas as pd

warnings.filterwarnings("ignore")
import bz2
from sklearn.model_selection import train_test_split
import sys
import logging
import os

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
log = logging.getLogger('compression Model')
log.setLevel(logging.DEBUG)
# read files
log.info("importing all libraries for compression models ...")
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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

A = list(X_test)  # files
B = list(y_test)  # author names
# --------------------------------------------------------------#
comp_accuracy = 0.91
# accuracy tested in testing enviroment, please refer report for accuracy
log.info("bzip compression model created...")


def compression(singleFile):
    bzip_pred = []
    bzip_compress_train = {}
    bzip_concat = {}
    compressionLevel = 9
    for author, data in zip(y, X):
        bzip_compress_train[author] = bz2.compress(bytes(data, 'utf-8'), compressionLevel)
        bzip_concat[author] = data + str(singleFile)
    bzip_concat_comp = {}
    bzip_singleFileComp = bz2.compress(bytes(singleFile, 'utf-8'))
    # compress concatinated file
    for i, j in bzip_concat.items():
        bzip_concat_comp[i] = bz2.compress(bytes(j, "utf-8"))
        # calculate ncd values
    bzip_ncd_values = {}
    for i, j in bzip_compress_train.items():
        bzip_ncd_values[i] = (len(bzip_concat_comp[i]) - min(len(bzip_compress_train[i]),
                                                             len(bzip_singleFileComp))) / max(
            len(bzip_compress_train[i]), len(bzip_singleFileComp))

    tp = next(iter(dict(sorted(bzip_ncd_values.items(), key=lambda item: item[1]))))
    pred = ''.join([i for i in tp if not i.isdigit()])
    bzip_pred.append(pred)
    return bzip_pred[0]
# pickle.dump(compression(singleFile),open('compression.pkl', 'wb'))
# cmp = compression(*args)
# print(compression(str))
