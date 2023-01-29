import sys
import pyiwn
import logging
import sys
import re
import nltk
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
log = logging.getLogger('input Handler')
log.setLevel(logging.DEBUG)

def text_process(text):
    log.info("data cleaning processing ......")
    pp = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
    nopunct = "".join([char for char in text if char not in pp])
    tokens = nltk.word_tokenize(nopunct)
    nopunct =" ".join([word for word in tokens])
    clean=re.sub(r'[a-zA-Z]', '', nopunct)
    return clean

def poly_non_poly_words(text):
    iwn = pyiwn.IndoWordNet(lang=pyiwn.Language.KANNADA)
    clean_sent=text_process(text)
    input_tokenize=nltk.word_tokenize(clean_sent)
    sent_token_len = len(input_tokenize)
    print(len(input_tokenize))
    log.info("polysemy words extracting started for given input sentence........")
    temp_list = []
    temp_list_non =[]
    for words in input_tokenize:
        try:
            a = iwn.synsets(words)
            w = ' '.join(str(k._head_word) for k in a)
            [temp_list.append(x)for x in nltk.word_tokenize(w) if x not in temp_list ]
        except KeyError:
            temp_list_non.append(words)
    return " ".join(temp_list+temp_list_non)
#print(text_process("ಜತೆಗೊಂದುewfwefasದಿನ!!!@@$$%%ನಾವು ನೀವು ಸಾಮಾನ್ಯವಾಗಿ ಬಳಸುವ ಫೇಸ್ಬುಕ್ ವಾಟ್ಸಾಪ್"))
# d ="ಜತೆಗೊಂದುe ದಿನ ನಾವು ನೀವು ಸಾಮಾನ್ಯವಾಗಿ ಬಳಸುವ ಫೇಸ್ಬುಕ್ ವಾಟ್ಸಾಪ್"
# print(poly_non_poly_words(d))