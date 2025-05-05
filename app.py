from flask import Flask, request, render_template, redirect, url_for, session
from flaskext.mysql import MySQL
from scholarly import scholarly
import mysql.connector
import os
import re
from itertools import islice
from dotenv import load_dotenv
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from langdetect import detect
from gensim import corpora, models
from utils import preprocess

# Load environment variables
load_dotenv()

app = Flask(__name__)

app.static_folder = 'assets'
app.secret_key = 'aBcD1234!@#$my_secret_key_flask_app'
# Konfigurasi MySQL

def get_db_connection():
    return mysql.connector.connect(
        host=os.getenv('DB_HOST', 'localhost'),
        user=os.getenv('DB_USERNAME', 'root'),
        password=os.getenv('DB_PASS'),
        database=os.getenv('DB_NAME', 'db_lda'),
        port=os.getenv('DB_PORT', 3306)
    )

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    error = None

    if request.method == 'POST':
        abstract = request.form.get('abstract', '').strip()

        if not abstract:
            error = "Abstrak tidak boleh kosong!"
        else:
            try:
                processed_text = preprocess(abstract)
                if not processed_text:
                    error = "Tidak ada kata valid ditemukan dalam abstrak."
                else:
                    # Load model dan dictionary
                    lda_model = models.LdaModel.load('models/lda_model.pkl')
                    dictionary = corpora.Dictionary.load('models/dictionary.pkl')

                    bow = dictionary.doc2bow(processed_text)
                    topics_distribution = sorted(
                        lda_model.get_document_topics(bow),
                        key=lambda x: x[1],
                        reverse=True
                    )
                    dominant_topic_id, probability = topics_distribution[0]
                    words = lda_model.show_topic(dominant_topic_id, 5)

                    result = {
                        'topic_label': f'Topik {dominant_topic_id}',
                        'probability': round(probability * 100, 2),
                        'words': ", ".join([w[0] for w in words]),
                        'abstract': abstract
                    }

            except Exception as e:
                error = f"Terjadi kesalahan: {str(e)}"

    return render_template('index.html', result=result, error=error)

@app.route('/scrape')
def data_scrape():
    # Pagination setup
    page = request.args.get('page', 1, type=int)
    per_page = 10  # Jumlah item per halaman
    offset = (page - 1) * per_page

    connection = get_db_connection()
    cursor = connection.cursor(dictionary=True)

    # Ambil total jumlah dokumen
    cursor.execute("SELECT COUNT(*) as total FROM documents")
    total_items = cursor.fetchone()['total']
    total_pages = (total_items + per_page - 1) // per_page

    # Ambil data sesuai halaman
    cursor.execute(f"SELECT * FROM documents ORDER BY id DESC LIMIT {per_page} OFFSET {offset}")
    results = cursor.fetchall()

    cursor.close()
    connection.close()

    return render_template(
        'scrape.html',
        results=results,
        count=0,
        pagination={
            'page': page,
            'per_page': per_page,
            'total_pages': total_pages
        }
    )

@app.route('/data-scrape', methods=['GET', 'POST'])
def scrape():
    if request.method == 'POST':
        keyword = request.form['keyword']
        search_query = scholarly.search_pubs(keyword)
        results = []

        connection = get_db_connection()
        cursor = connection.cursor()

        count = 0  # <-- Inisialisasi count di sini

        for _ in range(10):
            try:
                pub = next(search_query)
                title = pub['bib']['title']
                abstract = pub['bib']['abstract']
                author = ', '.join(pub['bib']['author']) 
                year = pub['bib']['pub_year']
                venue = pub['bib']['venue']
                link = pub['pub_url'] 

                cursor.execute("SELECT COUNT(*) FROM documents WHERE title = %s", (title,))
                
                if cursor.fetchone()[0] == 0:
                    cursor.execute(
                        """
                        INSERT INTO documents (title, author, abstract, year, venue) 
                        VALUES (%s, %s, %s, %s, %s)
                        """,
                        (title, author, abstract, year, venue)
                    )
                    count += 1 

            except StopIteration:
                print("No more results available.")
                break
            except Exception as e:
                print(f"Error occurred during scraping individual result: {e}")
                continue

        connection.commit() 
        cursor.close()
        connection.close()

        return redirect(url_for('data_scrape', count=count))

@app.route('/generate-lda', methods=['POST'])
def generate_lda():
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT abstract FROM documents WHERE abstract IS NOT NULL AND abstract != ''")
    results = cursor.fetchall()
    conn.close()

    if not results:
        return "Tidak ada abstrak untuk diproses."

    texts = [preprocess(row['abstract']) for row in results]

    dictionary = corpora.Dictionary(texts)
    dictionary.filter_extremes(no_below=2, no_above=0.8)
    corpus = [dictionary.doc2bow(text) for text in texts]

    lda_model = models.LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=5,
        passes=10,
        random_state=42,
        chunksize=100,
        alpha='auto',
        per_word_topics=True
    )

    # Simpan model dan dictionary
    os.makedirs('models', exist_ok=True)
    lda_model.save('models/lda_model.pkl')
    dictionary.save('models/dictionary.pkl')

    # Ambil topik dan simpan ke session
    topics = lda_model.print_topics(num_words=5)
    session['lda_topics'] = topics

    return redirect(url_for('view_topics'))
@app.route('/view-topics')
def view_topics():
    topics = session.get('lda_topics', [])
    labeled_topics = []

    for i, topic in enumerate(topics):
        words = topic[1]
        labeled_topics.append({
            'id': i,
            'label': f'Topik {i}',
            'words': words
        })

    return render_template('topics.html', topics=labeled_topics)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=6060)
