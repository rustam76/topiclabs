from flask import Flask, render_template, request, redirect, url_for, flash
import os
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import SimpleRNN, Dense, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import requests
from bs4 import BeautifulSoup

app = Flask(__name__)
app.secret_key = 'supersecretkey'

UPLOAD_FOLDER = 'uploads'
MODEL_FOLDER = 'models'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MODEL_FOLDER'] = MODEL_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)

# Variabel global untuk menyimpan tokenizer dan model
tokenizer = None
model = None
max_len = 0

def scrape_wikipedia(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        title = soup.find('h1', {'id': 'firstHeading'}).text.strip()
        content_div = soup.find('div', {'class': 'mw-parser-output'})
        paragraphs = content_div.find_all('p', limit=10)  # Ambil 10 paragraf pertama
        text = "\n".join([p.get_text() for p in paragraphs])
        return title, text
    except Exception as e:
        raise ValueError(f"Scraping gagal: {str(e)}")
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'url' in request.form:
            url = request.form['url']
            try:
                title, text = scrape_wikipedia(url)
                filename = f"{title.replace(' ', '_')}.txt"
                raw_filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                labeled_filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"labeled_{filename}")

                # Simpan file mentah dulu
                with open(raw_filepath, 'w', encoding='utf-8') as f:
                    f.write(text)

                # Tambahkan label dan simpan versi baru
                with open(raw_filepath, 'r', encoding='utf-8') as infile, \
                     open(labeled_filepath, 'w', encoding='utf-8') as outfile:
                    for line in infile:
                        clean_line = line.strip()
                        if clean_line:
                            outfile.write(f"1,{clean_line}\n")

                flash("Artikel berhasil di-scrape dan diberi label.")
                return redirect(url_for('train_from_file', filename=f"labeled_{filename}"))
            except Exception as e:
                flash(str(e))
        elif 'file' in request.files:
            file = request.files['file']
            if file.filename == '':
                flash('File tidak dipilih')
                return redirect(request.url)
            if file and file.filename.endswith('.txt'):
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
                file.save(filepath)

                # Tambahkan label saat upload
                labeled_filename = f"labeled_{file.filename}"
                labeled_filepath = os.path.join(app.config['UPLOAD_FOLDER'], labeled_filename)

                with open(filepath, 'r', encoding='utf-8') as infile, \
                     open(labeled_filepath, 'w', encoding='utf-8') as outfile:
                    for line in infile:
                        clean_line = line.strip()
                        if clean_line:
                            outfile.write(f"1,{clean_line}\n")

                flash("File berhasil diunggah dan diberi label.")
                return redirect(url_for('train_from_file', filename=labeled_filename))

    return render_template('rnn/index.html')

@app.route('/train/<filename>', methods=['GET', 'POST'])
def train_from_file(filename):
    global tokenizer, model, max_len
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            texts = f.readlines()

        labels = [int(t.split(',')[0]) for t in texts]  # label, teks
        docs = [t.split(',')[1].strip() for t in texts]

        # Tokenisasi
        tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
        tokenizer.fit_on_texts(docs)
        sequences = tokenizer.texts_to_sequences(docs)
        max_len = max(len(s) for s in sequences)
        X = pad_sequences(sequences, maxlen=max_len, padding='post')
        y = np.array(labels)

        # Buat model RNN
        model = Sequential()
        model.add(Embedding(input_dim=5000, output_dim=64, input_length=max_len))
        model.add(SimpleRNN(64))
        model.add(Dense(1, activation='sigmoid'))

        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.fit(X, y, epochs=5, verbose=1)

        # Simpan model dan tokenizer
        model.save(os.path.join(app.config['MODEL_FOLDER'], 'rnn_model.h5'))
        with open(os.path.join(app.config['MODEL_FOLDER'], 'tokenizer.pkl'), 'wb') as f:
            pickle.dump(tokenizer, f)
        with open(os.path.join(app.config['MODEL_FOLDER'], 'max_len.pkl'), 'wb') as f:
            pickle.dump(max_len, f)

        flash("Model berhasil dilatih!")
        return redirect(url_for('test'))

    except Exception as e:
        flash(f"Gagal melatih model: {str(e)}")
        return redirect(url_for('index'))

@app.route('/test', methods=['GET', 'POST'])
def test():
    global tokenizer, model, max_len
    result = ""
    if request.method == 'POST':
        text = request.form['text']
        if not text.strip():
            flash("Masukkan teks untuk diprediksi")
            return redirect(url_for('test'))
        if not model or not tokenizer:
            flash("Model belum dilatih")
            return redirect(url_for('index'))

        seq = tokenizer.texts_to_sequences([text])
        padded = pad_sequences(seq, maxlen=max_len, padding='post')
        prediction = model.predict(padded)[0][0]
        result = "Positif" if prediction > 0.5 else "Negatif"
    return render_template('rnn/test.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)