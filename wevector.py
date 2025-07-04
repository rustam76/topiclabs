from flask import Flask, render_template, request, redirect, url_for
import requests
from bs4 import BeautifulSoup
import re
import os
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from gensim.models import Word2Vec
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)

# Unduh tokenizer NLTK jika belum
nltk.download('punkt')

app = Flask(__name__)
DATA_FILE = 'data/scraped_text.txt'
MODEL_PATH = 'model/word2vec.model'

# --- SCRAPING ---
def scrape_wikipedia(url):
    headers = {'User-Agent': 'Mozilla/5.0'}
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, 'html.parser')
    paragraphs = soup.find_all('p')
    text = ''
    for p in paragraphs:
        text += p.get_text() + '\n'
    text = re.sub(r'\[.*?\]', '', text)  # Hapus referensi [1], [2]
    return text.strip()

# --- TRAINING ---
def train_word2vec_model(text):
    sentences = sent_tokenize(text.lower())
    tokenized_sentences = [word_tokenize(sent) for sent in sentences]
    model = Word2Vec(
        sentences=tokenized_sentences,
        vector_size=100,
        window=5,
        min_count=1,
        workers=4
    )
    model.save(MODEL_PATH)
    return model

# --- ROUTES ---
@app.route('/')
def index():
    return render_template('word-vector/index.html')

@app.route('/scrape', methods=['GET', 'POST'])
def scrape():
    if request.method == 'POST':
        url = request.form['url']
        text = scrape_wikipedia(url)
        with open(DATA_FILE, 'w', encoding='utf-8') as f:
            f.write(text)
        return redirect(url_for('train'))
    return render_template('word-vector/scrape.html')

@app.route('/train')
def train():
    if not os.path.exists(DATA_FILE):
        return "Data belum discrape!", 400
    with open(DATA_FILE, 'r', encoding='utf-8') as f:
        text = f.read()
    model = train_word2vec_model(text)
    return redirect(url_for('test'))

@app.route('/test', methods=['GET', 'POST'])
def test():
    if not os.path.exists(MODEL_PATH):
        return "Model belum dilatih!", 400
    model = Word2Vec.load(MODEL_PATH)
    similar_words = []
    vocab = list(model.wv.key_to_index.keys())[:50]  # contoh 50 kata pertama
    if request.method == 'POST':
        word = request.form['word']
        try:
            similar_words = model.wv.most_similar(word)
        except KeyError:
            similar_words = [("Kata tidak ditemukan dalam kosakata", 0)]
    return render_template('word-vector/test.html',
                           similar_words=similar_words,
                           vocab_preview=vocab)

@app.route('/model-info')
def model_info():
    if not os.path.exists(MODEL_PATH):
        return "Model belum dilatih!", 400
    model = Word2Vec.load(MODEL_PATH)
    vocab_size = len(model.wv)
    vector_size = model.vector_size
    vocab_list = list(model.wv.key_to_index.keys())
    return render_template('word-vector/model_info.html',
                           vocab_size=vocab_size,
                           vector_size=vector_size,
                           vocab_list=vocab_list)

@app.route('/retrain-model')
def retrain_model():
    if not os.path.exists(DATA_FILE):
        return "Data belum discrape!", 400
    with open(DATA_FILE, 'r', encoding='utf-8') as f:
        text = f.read()
    train_word2vec_model(text)
    return redirect(url_for('model_info'))

@app.route('/delete-model')
def delete_model():
    if os.path.exists(MODEL_PATH):
        os.remove(MODEL_PATH)
    return redirect(url_for('index'))

if __name__ == '__main__':
    os.makedirs('data', exist_ok=True)
    os.makedirs('model', exist_ok=True)
    app.run(debug=True)