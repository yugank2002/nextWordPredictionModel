from flask import Flask, request, jsonify, send_from_directory
from keras.models import load_model
import numpy as np
import pickle
import os
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)


model = load_model('model/model.h5') 
with open('model/tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)


@app.route('/')
def home():
    return send_from_directory('frontend', 'index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
       
        seed_text = request.form['seed_text']
        next_words = int(request.form['next_words'])
        
       
        output_text = seed_text
        for _ in range(next_words):
            token_list = tokenizer.texts_to_sequences([seed_text])[0]
            token_list = pad_sequences([token_list], maxlen=model.input_shape[1] - 1, padding='pre')
            predicted_probs = model.predict(token_list, verbose=0)
            predicted_class = np.argmax(predicted_probs, axis=-1)[0]
            
            
            output_word = ""
            for word, index in tokenizer.word_index.items():
                if index == predicted_class:
                    output_word = word
                    break
            
            seed_text += " " + output_word  

        return jsonify({'predicted_text': seed_text})

if __name__ == '__main__':
    app.run(debug=True)
