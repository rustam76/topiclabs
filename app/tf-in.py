from flask import Flask, request, jsonify, render_template
from flask_mysqldb import MySQL
from scholarly import scholarly
import os
from itertools import islice

app = Flask(__name__)

app.static_folder = 'assets'

# Konfigurasi MySQL
app.config['MYSQL_HOST'] = os.getenv('MYSQL_HOST', 'localhost')
app.config['MYSQL_USER'] = os.getenv('MYSQL_USER', 'root')
app.config['MYSQL_PASSWORD'] = os.getenv('MYSQL_PASSWORD', '')
app.config['MYSQL_DB'] = os.getenv('MYSQL_DB', 'db_inverted')

mysql = MySQL(app)

@app.route('/')
def index():
    return render_template('index.html', current_path=request.path)

@app.route('/data-scraping')
def data_scraping():
    return render_template('dokument.html', current_path=request.path)

@app.route('/detail/<int:id>')
def detail(id):
    cur = mysql.connection.cursor()
    cur.execute("SELECT title, abstract, link FROM tb_dokument WHERE id = %s", (id,))
    result = cur.fetchone()
    if not result:
        return "Document not found", 404
    return render_template('detail_dokumen.html', title=result[0], abstract=result[1], link=result[2])


@app.route('/search', methods=['POST'])
def search():
    data = request.json
    query = data.get('query')

    if not query:
        return jsonify({'error': 'woy kamu lupa isi pencariannya'}), 400

    try:
        # Simulating a database query for inverted search
        cur = mysql.connection.cursor()
        cur.execute("""
            SELECT id,title, abstract, link
            FROM tb_dokument
            WHERE MATCH(title, abstract) AGAINST (%s IN NATURAL LANGUAGE MODE)
            LIMIT 10
        """, (query,))

        results = cur.fetchall()

        return jsonify({
            'results': [
                { 'id': row[0], 'title': row[1], 'abstract': row[2], 'link': row[3]} for row in results
            ]
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    

@app.route('/scrape', methods=['POST'])
def scrape_scholar():
    data = request.json
    query = data.get('query', 'sistem informasi')  # Menggunakan query dari input JSON jika tersedia
    
    if not query:
        return jsonify({'error': 'woy isi lah querynya'}), 400

    try:
        # Membuka koneksi database
        cur = mysql.connection.cursor()
        count = 0
        
        # Melakukan pencarian menggunakan scholarly
        search_query = scholarly.search_pubs(query)
        for _ in range(50):  # Batas hingga 50 hasil
            try:
                pub = next(search_query)

                # Mengambil data dari objek 'Publication'
                title = pub.bib.get('title', 'Title not found')
                abstract = pub.bib.get('abstract', 'Abstract not found')
                author = ', '.join(pub.bib.get('author', [])) 
                year = pub.bib.get('year', 'Year not found')
                venue = pub.bib.get('venue', 'Venue not found')
                link = pub.bib.get('url', 'https://www.google.com')  
                # Debugging log
                print(f"Scraped Data -> Title: {title}, Abstract: {abstract}, Link: {link}")
                
                # Periksa duplikasi berdasarkan judul
                cur.execute("SELECT COUNT(*) FROM tb_dokument WHERE title = %s", (title,))
                if cur.fetchone()[0] == 0:  # Jika tidak ada duplikasi, masukkan data
                    cur.execute("""
                        INSERT INTO tb_dokument (title, abstract, author, year, venue, link)
                        VALUES (%s, %s, %s, %s, %s, %s)
                    """, (title, abstract, author, year, venue, link))
                    count += 1

            except StopIteration:
                print("No more results available.")
                break
            except Exception as e:
                print(f"Error occurred during scraping individual result: {e}")
                continue

        # Commit transaksi database
        mysql.connection.commit()
        return jsonify({'message': f'Scraping and saving completed successfully! {count} records saved.'})

    except Exception as e:
        print(f"Error during scraping: {e}")
        return jsonify({'error': str(e)}), 500
    finally:
        cur.close()



@app.route('/papers', methods=['GET'])
def get_papers():
    cur = mysql.connection.cursor()
    cur.execute("SELECT * FROM papers")
    results = cur.fetchall()

    papers = [{'id': row[0], 'title': row[1], 'abstract': row[2], 'link': row[3]} for row in results]
    return jsonify(papers)


@app.route('/get_data', methods=['GET'])
def get_data():
    try:
        page = int(request.args.get('page', 1))
        limit = int(request.args.get('limit', 5))
        offset = (page - 1) * limit

        cursor = mysql.connection.cursor()
        cursor.execute("SELECT COUNT(*) FROM tb_dokument")
        total_items = cursor.fetchone()[0]

        cursor.execute("SELECT * FROM tb_dokument LIMIT %s OFFSET %s", (limit, offset))
        rows = cursor.fetchall()

        data = [{'title': row[1], 'abstract': row[2], 'link': row[3], 'vanue': row[4], 'year': row[5], 'author': row[6]} for row in rows]
        total_pages = (total_items + limit - 1) // limit  # Ceiling division

        return jsonify({
            'results': data,
            'totalPages': total_pages
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


