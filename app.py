from flask import Flask, request, render_template, redirect, url_for, session,  jsonify, Response
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
import pandas as pd
from bertopic import BERTopic
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import base64
import io
import urllib.parse
import joblib
import os
import re
from nltk.corpus import stopwords
import nltk


# Load environment variables
load_dotenv()
nltk.download('stopwords')
app = Flask(__name__)

app.static_folder = 'assets'
app.secret_key = 'aBcD1234!@#$my_secret_key_flask_app'
UPLOAD_FOLDER = 'uploads'
MODEL_FOLDER = 'models'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)

# Konfigurasi MySQL

def get_db_connection():
    return mysql.connector.connect(
        host=os.getenv('DB_HOST', 'localhost'),
        user=os.getenv('DB_USERNAME', 'root'),
        password=os.getenv('DB_PASS'),
        database=os.getenv('DB_NAME', 'db_lda'),
        port=os.getenv('DB_PORT', 3306)
    )

# Fungsi preprocessing sederhana
def bertopic_preprocess(text):
    text = re.sub(r'\d+', '', text)  # Hapus angka
    text = re.sub(r'[^\w\s]', '', text)  # Hapus tanda baca
    text = text.lower()  # Lowercase
    tokens = text.split()
    
    # Stopword removal
    stop_words = set(stopwords.words('indonesian') + stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    return " ".join(tokens)
    
    return tokens

@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')


# tf inverted
@app.route('/tf-inverted', methods=['GET', 'POST'])
def index_tf():
    return render_template('TF-Inverted/index.html')


@app.route('/data-scraping')
def data_scraping():
    return render_template('TF-Inverted/dokument.html', current_path=request.path)

@app.route('/detail/<int:id>')
def detail(id):
    connection = get_db_connection()
    cursor = connection.cursor()
    cursor.execute("SELECT title, abstract, link FROM tf_inverted_dokumen WHERE id = %s", (id,))
    result = cursor.fetchone()
    if not result:
        return "Document not found", 404
    return render_template('TF-Inverted/detail_dokumen.html', title=result[0], abstract=result[1], link=result[2])


@app.route('/search', methods=['POST'])
def search():
    data = request.json
    query = data.get('query')

    if not query:
        return jsonify({'error': 'woy kamu lupa isi pencariannya'}), 400

    try:
        # Simulating a database query for inverted search
        connection = get_db_connection()
        cursor = connection.cursor()
        cursor.execute("""
            SELECT id,title, abstract, link
            FROM tf_inverted_dokumen
            WHERE MATCH(title, abstract) AGAINST (%s IN NATURAL LANGUAGE MODE)
            LIMIT 10
        """, (query,))

        results = cursor.fetchall()

        return jsonify({
            'results': [
                { 'id': row[0], 'title': row[1], 'abstract': row[2], 'link': row[3]} for row in results
            ]
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    

@app.route('/scrape-tf', methods=['POST'])
def scrape_scholar():
    if request.method == 'POST':
        data = request.json
        query = data.get('query')
        search_query = scholarly.search_pubs(query)
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

                cursor.execute("SELECT COUNT(*) FROM tf_inverted_dokumen WHERE title = %s", (title,))
                
                if cursor.fetchone()[0] == 0:
                    cursor.execute(
                        """
                        INSERT INTO tf_inverted_dokumen (title, author, abstract, year, venue, link) 
                        VALUES (%s, %s, %s, %s, %s, %s)
                        """,
                        (title, author, abstract, year, venue, link)
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
        return jsonify({'message': f'Scraping and saving completed successfully! {count} records saved.'})




@app.route('/papers', methods=['GET'])
def get_papers():
    connection = get_db_connection()
    cursor = connection.cursor()
    cursor.execute("SELECT * FROM papers")
    results = cursor.fetchall()

    papers = [{'id': row[0], 'title': row[1], 'abstract': row[2], 'link': row[3]} for row in results]
    return jsonify(papers)


@app.route('/get_data', methods=['GET'])
def get_data():
    try:
        page = int(request.args.get('page', 1))
        limit = int(request.args.get('limit', 5))
        offset = (page - 1) * limit

        connection = get_db_connection()
        cursor = connection.cursor()
        cursor.execute("SELECT COUNT(*) FROM tf_inverted_dokumen")
        total_items = cursor.fetchone()[0]

        cursor.execute("SELECT * FROM tf_inverted_dokumen LIMIT %s OFFSET %s", (limit, offset))
        rows = cursor.fetchall()

        data = [{'title': row[1], 'abstract': row[3], 'link': row[3], 'vanue': row[5], 'year': row[4], 'author': row[2]} for row in rows]
        total_pages = (total_items + limit - 1) // limit  # Ceiling division

        return jsonify({
            'results': data,
            'totalPages': total_pages
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500




# model lda

@app.route('/lda', methods=['GET', 'POST'])
def index_lda():
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

    return render_template('lda/index.html', result=result, error=error)

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
        'lda/scrape.html',
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

    return render_template('lda/topics.html', topics=labeled_topics)

# model bertopic
@app.route('/bertopic', methods=['GET', 'POST'])
def bertopic():
    accuracy = request.args.get('accuracy')
    success = request.args.get('success')
    error = request.args.get('error')
    if accuracy:
        accuracy = float(accuracy)

    # Muat hasil evaluasi jika tersedia
    conf_matrix_img = None
    class_report_img = None

    eval_path = os.path.join(MODEL_FOLDER, "evaluation_results.pkl")
    if os.path.exists(eval_path):
        try:
            eval_result = joblib.load(eval_path)
            accuracy = round(eval_result['accuracy'] * 100, 2)
            conf_matrix_img = eval_result.get('conf_matrix_img')
            class_report_img = eval_result.get('class_report_img')
        except Exception:
            pass

    if request.method == 'POST':
        if 'test' in request.files:
            file = request.files['test']
            path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(path)

            df = pd.read_csv(path)

            if 'content' not in df.columns:
                return "File harus memiliki kolom 'content'", 400

            texts = df['title'].fillna('') + " " + df['content'].fillna('')
            texts = texts.tolist()

            connection = get_db_connection()
            cursor = connection.cursor()
            cursor.execute("TRUNCATE TABLE tb_bertopic")
            connection.commit()

            for _, row in df.iterrows():
                title = str(row['title']) if pd.notna(row['title']) else ""
                content = str(row['content']) if pd.notna(row['content']) else ""
                category = str(row['category']) if pd.notna(row['category']) else ""
                tags = str(row['tags']) if pd.notna(row['tags']) else ""

                cursor.execute(
                    """
                    INSERT INTO tb_bertopic (title, content, category, tags) 
                    VALUES (%s, %s, %s, %s)
                    """,
                    (title, content, category, tags)
                )
            connection.commit()
            cursor.close()
            connection.close()
        return redirect(url_for('bertopic') + '?success=1')

    return render_template(
        'bertopic/index.html',
        accuracy=accuracy,
        success=success,
        error=error,
        conf_matrix_img=f"data:image/png;base64,{conf_matrix_img}" if conf_matrix_img else None,
        class_report_img=f"data:image/png;base64,{class_report_img}" if class_report_img else None
    )

@app.route('/train-category-model', methods=['GET', 'POST'])
def train_category_model():
    try:
        # Ambil data dari database
        connection = get_db_connection()
        query = "SELECT title, content, category FROM tb_bertopic"
        df = pd.read_sql(query, connection)
        connection.close()

        if df.empty:
             return redirect(url_for('bertopic', error="Data kosong"))

        # Gabungkan title dan content
        df['text'] = df['title'].fillna('') + " " + df['content'].fillna('')
        df['cleaned_text'] = df['text'].apply(bertopic_preprocess)

        # Siapkan X dan y
        X = df['cleaned_text']
        y = df['category']

        # Split data untuk evaluasi
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Buat pipeline model
        model_clf = Pipeline([
            ('tfidf', TfidfVectorizer()),
            ('clf', LogisticRegression(max_iter=1000))
        ])

        # Latih model
        model_clf.fit(X_train, y_train)

        # Prediksi dan evaluasi
        y_pred = model_clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        class_report = classification_report(y_test, y_pred, output_dict=True)
        conf_matrix = confusion_matrix(y_test, y_pred)

        # Gambar confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                    xticklabels=model_clf.classes_, yticklabels=model_clf.classes_)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")

        img = io.BytesIO()
        plt.savefig(img, format='png', bbox_inches='tight')
        img.seek(0)
        conf_matrix_url = base64.b64encode(img.getvalue()).decode('utf8')
        plt.close()

        # Gambar classification report
        report_df = pd.DataFrame(class_report).transpose()
        plt.figure(figsize=(10, 6))
        sns.heatmap(report_df.iloc[:-1].T, annot=True, cmap="YlGnBu", cbar=False)
        plt.title("Classification Report")

        img2 = io.BytesIO()
        plt.savefig(img2, format='png', bbox_inches='tight')
        img2.seek(0)
        class_report_url = base64.b64encode(img2.getvalue()).decode('utf8')
        plt.close()

        # Simpan model
        model_path = os.path.join(MODEL_FOLDER, "category_classifier.pkl")
        joblib.dump(model_clf, model_path)

        # Simpan hasil evaluasi untuk ditampilkan nanti
        eval_result = {
            'accuracy': acc,
            'conf_matrix_img': conf_matrix_url,
            'class_report_img': class_report_url
        }
        eval_path = os.path.join(MODEL_FOLDER, "evaluation_results.pkl")
        joblib.dump(eval_result, eval_path)

        # Redirect ke bertopic dengan query param akurasi
        return redirect(url_for('bertopic', accuracy=round(acc * 100, 2)))

    except Exception as e:
        return f"Error: {str(e)}"
    

@app.route('/download-template')
def download_template():
    # Membuat DataFrame kosong dengan kolom yang diperlukan
    template_df = pd.DataFrame(columns=["title", "content", "category", "tags"])
    
    # Simpan ke buffer CSV
    csv_data = template_df.to_csv(index=False)
    return Response(
        csv_data,
        mimetype="text/csv",
        headers={"Content-disposition": "attachment; filename=template_dataset.csv"}
    )

@app.route('/reset-model')
def reset_model():
    try:
        # Hapus data dari tabel tb_bertopic
        connection = get_db_connection()
        cursor = connection.cursor()

        # Kosongkan tabel
        cursor.execute("DELETE FROM tb_bertopic")
        connection.commit()

        cursor.close()
        connection.close()

        # Hapus file model dan hasil evaluasi jika ada
        model_path = os.path.join(MODEL_FOLDER, "category_classifier.pkl")
        eval_path = os.path.join(MODEL_FOLDER, "evaluation_results.pkl")

        if os.path.exists(model_path):
            os.remove(model_path)
        if os.path.exists(eval_path):
            os.remove(eval_path)

        return redirect(url_for('bertopic'))

    except Exception as e:
        return redirect(url_for('bertopic'))

@app.route('/test-model', methods=['GET', 'POST'])
def test_model():
    prediction = None
    if request.method == 'POST':
        text = request.form.get('text')
        model_path = os.path.join(MODEL_FOLDER, "category_classifier.pkl")

        if not os.path.exists(model_path):
            return "Model belum dilatih.", 404

        model_clf = joblib.load(model_path)
        prediction = model_clf.predict([text])[0]

    return render_template('bertopic/test_model.html', prediction=prediction)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=6060)
