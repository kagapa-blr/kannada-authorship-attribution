import pyiwn
import nltk 
import re
from nltk import word_tokenize
import pandas as pd
import logging
import sys
import os
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
log = logging.getLogger('Extract polysemy')
log.setLevel(logging.DEBUG)
#read files 

script_dir = os.path.dirname(__file__)
rel_path = "../Dataset/All_Authors_dataset.csv"
abs_file_path = os.path.join(script_dir, rel_path)

df_instance=pd.read_csv(abs_file_path, encoding='utf-8')
del df_instance['Unnamed: 0']
# df_instance
# print(df_instance.columns)

# print(df_instance['authors_clean'].value_counts())
log.info("successfully read authors data .....")

def text_process(text):
    pp = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
    nopunct = "".join([char for char in text if char not in pp])
    tokens = nltk.word_tokenize(nopunct)
    nopunct =" ".join([word for word in tokens])
    clean=re.sub(r'[a-zA-Z]', '', nopunct)
    return clean



clean_data={}
c=0
authors = []
for auhor, file in zip(df_instance['authors'] , df_instance['instances']):
    c+=1
    authors.append(auhor)
    clean_data[auhor+str(c)]=word_tokenize(file)
log.info("Data cleaning completed")

#kannada wordnet for extracting polysemy words
iwn = pyiwn.IndoWordNet(lang=pyiwn.Language.KANNADA)

log.info("polysemy words extracting started.... this will take few mins ...please wait")
def poly_non_poly_words(dict):
    authors_dict=dict
    poly_dict ={}
    np_words ={}
    for i, j in authors_dict.items():
        temp_list = []
        temp_list_non =[]
        for words in j:
            try:
                a = iwn.synsets(words)
                w = ' '.join(str(k._head_word) for k in a)
                [temp_list.append(x)for x in nltk.word_tokenize(w) if x not in temp_list ]
            except KeyError :
                temp_list_non.append(words)
                
        poly_dict[i] = temp_list+temp_list_non
        np_words [i] = temp_list_non
    return poly_dict, np_words
                 
poly_dict,np_words = poly_non_poly_words(clean_data)
log.info("polysemy wods extracted successfully .....")

def poly_count(poly_dict):
    poly_count=[]
    for i, j in poly_dict.items():
        poly_count.append(len(j))
    return poly_count
        
poly_count = poly_count(poly_dict) 


final_df = pd.DataFrame()
final_df['authors'] = authors
final_df['text'] = poly_dict.values()
final_df['poly_count'] = poly_count
final_df['original_text']=df_instance['instances']

str_list =[]
for i in final_df['text']:
#     s = re.sub(r'[A-Z|0-9|a-z| ]','',str(i))
#     print(s)
    str_list.append(text_process(str(i)))


def FinalDf():    
    final_df['clean_text']= str_list
    #final_df['clean_text'][0]
    tmp = final_df.copy()
    poly_emotion_df = tmp
    del poly_emotion_df['text']
    del poly_emotion_df['original_text']
    poly_emotion_df['clean_authors'] = df_instance['authors_clean']

    poly_emotion_df.to_csv("polysemy_Model/polysemy_extracted.csv")
    log.info("final csv file created successfully ....")
    return
FinalDf()