from flask import Flask, request, render_template, redirect, url_for
import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Variabel global untuk menyimpan tokenizer dan model
input_tokenizer = None
target_tokenizer = None
model = None
history = None

import csv
import random
from flask import send_file,make_response
from io import BytesIO
from io import StringIO
# Global variables
input_tokenizer = None
target_tokenizer = None
model = None
encoder_model = None
decoder_model = None
max_encoder_seq_length = 0
num_decoder_tokens = 0
@app.route('/generate_dataset')
def generate_dataset():
    input_sentences = [
        "apa kabar", "siapa kamu", "dimana kamu", "makan apa hari ini",
        "bagaimana cuaca", "saya sedih", "terima kasih", "bisakah kamu bantu"
    ]
    output_sentences = [
        "saya baik", "saya asisten AI", "di server cloud", "saya suka nasi goreng",
        "cuaca cerah", "semangat ya", "sama-sama", "tentu, saya siap membantu"
    ]

    data = []
    for _ in range(1000):
        data.append([random.choice(input_sentences), random.choice(output_sentences)])

    si = StringIO()
    writer = csv.writer(si)
    writer.writerow(['input', 'output'])
    writer.writerows(data)

    response = make_response(si.getvalue())
    response.headers["Content-Type"] = "text/csv"
    response.headers["Content-Disposition"] = "attachment; filename=dataset_random.csv"

    return response

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['dataset']
        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            return redirect(url_for('train', filename=file.filename))
    return render_template('seq/index.html')

@app.route('/train/<filename>')
def train(filename):
    global input_tokenizer, target_tokenizer, model, encoder_model, decoder_model
    global max_encoder_seq_length, num_decoder_tokens

    # Load dataset
    df = pd.read_csv(os.path.join(app.config['UPLOAD_FOLDER'], filename), on_bad_lines='skip')
    input_texts = df.iloc[:, 0].astype(str).tolist()
    target_texts = df.iloc[:, 1].astype(str).tolist()

    # Tokenize
    input_tokenizer = Tokenizer(filters='')
    input_tokenizer.fit_on_texts(input_texts)
    input_seq = input_tokenizer.texts_to_sequences(input_texts)
    input_pad = pad_sequences(input_seq, padding='post')
    max_encoder_seq_length = input_pad.shape[1]

    target_tokenizer = Tokenizer(filters='')
    target_tokenizer.fit_on_texts(target_texts)
    target_seq = target_tokenizer.texts_to_sequences(target_texts)
    target_pad = pad_sequences(target_seq, padding='post')

    # Prepare data
    encoder_input_data = input_pad
    decoder_input_data = target_pad[:, :-1]
    decoder_target_data = target_pad[:, 1:]

    latent_dim = 256
    num_encoder_tokens = len(input_tokenizer.word_index) + 1
    num_decoder_tokens = len(target_tokenizer.word_index) + 1

    # Encoder
    encoder_inputs = Input(shape=(None,))
    encoder_emb = Embedding(num_encoder_tokens, latent_dim)(encoder_inputs)
    encoder_lstm = LSTM(latent_dim, return_state=True)
    _, state_h, state_c = encoder_lstm(encoder_emb)
    encoder_states = [state_h, state_c]

    # Decoder
    decoder_inputs = Input(shape=(None,))
    decoder_emb = Embedding(num_decoder_tokens, latent_dim)(decoder_inputs)
    decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_emb, initial_state=encoder_states)
    decoder_dense = Dense(num_decoder_tokens, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)

    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Train
    history = model.fit(
        [encoder_input_data, decoder_input_data],
        decoder_target_data,
        batch_size=32,
        epochs=10,
        validation_split=0.2
    )

    # Save encoder and decoder models for inference
    encoder_model = Model(encoder_inputs, encoder_states)

    decoder_state_input_h = Input(shape=(latent_dim,))
    decoder_state_input_c = Input(shape=(latent_dim,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

    decoder_outputs, state_h, state_c = decoder_lstm(
        decoder_emb, initial_state=decoder_states_inputs
    )
    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_dense(decoder_outputs)

    decoder_model = Model(
        [decoder_inputs] + decoder_states_inputs,
        [decoder_outputs] + decoder_states
    )

    return render_template('seq/train.html', history=history.history)

def decode_sequence(input_seq):
    states_value = encoder_model.predict(input_seq)

    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = target_tokenizer.word_index.get('start', 1)

    stop_condition = False
    decoded_sentence = []

    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)

        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_word = target_tokenizer.index_word.get(sampled_token_index, "")

        if sampled_word == '' or sampled_word == 'end' or len(decoded_sentence) > 50:
            stop_condition = True
        else:
            decoded_sentence.append(sampled_word)

        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index

        states_value = [h, c]

    return ' '.join(decoded_sentence)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    result = ""
    if request.method == 'POST':
        input_text = request.form['input_text']

        input_seq = input_tokenizer.texts_to_sequences([input_text])
        input_pad = pad_sequences(input_seq, maxlen=max_encoder_seq_length, padding='post')

        result = decode_sequence(input_pad)

    return render_template('seq/predict.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)