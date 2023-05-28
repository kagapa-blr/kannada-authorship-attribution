import logging
import os
import pickle
import sys

# import ngram_model as  nm
from flask import Flask, jsonify, request
from flask_cors import CORS

from compression_Model import compression_model as cm
from lexical_Model import ConvertInput

#Initialize the flask App
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
log = logging.getLogger('api.py')
log.setLevel(logging.DEBUG)
app = Flask(__name__)
CORS(app)



#read files 



@app.route('/test', methods = ['GET', 'POST'])
def test():
    if request.method == 'GET':
        return jsonify ({"response": "Get Request called"})
    elif request.method == 'POST':
        req_Json = request.json
        name = req_Json['name']
        return jsonify({"response": "Hi" +name})

@app.route('/ngram', methods = ['GET', 'POST'])
def ngram():
    if request.method == 'POST':
        log.info("importing ngram model and vector files")
        root_ngram = os.path.dirname(__file__)
        poly_model_path = "../ngram_Model"
    
        ngram_model_file_path = os.path.join(root_ngram, poly_model_path+"/ngram_attribution.pkl")
        ngram_clf = pickle.load(open(ngram_model_file_path,'rb'))

        ngram_loaded_vec_file = os.path.join(root_ngram, poly_model_path+"/ngram_vect.pkl")
        ngram_loaded_vec = pickle.load(open(ngram_loaded_vec_file, "rb"))

        ngram_model_accuracy_file = os.path.join(root_ngram, poly_model_path+"/ngram_model_Acuuracy.pkl")
        ngram_model_Acuuracy = pickle.load(open(ngram_model_accuracy_file, "rb"))
        
        log.info("ngram model and vector files successfully imported for prediction...")
        #taking input from the browser/client
        input_text = request.form['kannada_text']
        txt_vc = ngram_loaded_vec.transform([input_text])
        result_pred = ngram_clf.predict(txt_vc)
        accuracy = ngram_model_Acuuracy['ngram_accuracy']
        #print(accuracy, type(ngram_model_Acuuracy))
        return jsonify(
            {
                "Author_Name":'{}'.format(result_pred[0]),
                "Accuracy":str(accuracy),
                "Model": "ngram"
            }
        )

@app.route('/compression', methods = ['GET', 'POST'])
def compression():
    if request.method == 'POST':
        user_text = request.form['kannada_text']
        result  = cm.compression(user_text)
        accuracy = cm.comp_accuracy
        return jsonify({
                "Author_Name":'{}'.format(result),
                "Accuracy":str(accuracy),
                "Model": "compression"
            })

@app.route('/lexical', methods = ['GET', 'POST'])
def lexical():
     if request.method == 'POST':
        user_text = request.form['kannada_text']
        author,accuracy=ConvertInput.convertAndPredictAuthor(user_text)
        return jsonify({
                "Author_Name":'{}'.format(author),
                "Accuracy":accuracy,
                "Model": "Lexical Model"
            })

@app.route('/polysemy', methods = ['GET', 'POST'])
def polysemy():
    if request.method == 'POST':
         
        root = os.path.dirname(__file__)
        poly_model_path = "../polysemy_Model"
        poly_model_file = os.path.join(root, poly_model_path+"/poly_finalized_model.sav")
        poly_vec_file = os.path.join(root, poly_model_path+"/polysem_vect.pkl")
        poly_accuracy_file = os.path.join(root, poly_model_path+"/accuracy.p")
        
        poly_clf = pickle.load(open(poly_model_file,'rb'))
        poly_loaded_vec = pickle.load(open(poly_vec_file,'rb'))
        poly_accuracy_loaded = pickle.load(open(poly_accuracy_file,'rb'))

        input_text = request.form['kannada_text']
        txt_vc = poly_loaded_vec.transform([input_text])
        result_pred = poly_clf.predict(txt_vc)
        accuracy = poly_accuracy_loaded['accuracy']
   
        return jsonify(
            {
                "Author_Name":'{}'.format(result_pred[0]),
                "Accuracy":str(accuracy),
                "Model": "Polysemy"
            }
        )

  
    
if __name__ == "__main__":
    app.run(debug=True, port = 9090)