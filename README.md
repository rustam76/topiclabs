Berikut adalah perbaikan dan penyempurnaan dari file `README.md` yang jelas, rapi, dan mudah diikuti oleh siapa pun yang ingin menjalankan proyek ini:

---

# Topic Modelling dengan TF-IDF, LDA, dan BERTopic

Proyek ini merupakan implementasi dari beberapa teknik **topic modelling** seperti **TF-IDF**, **LDA**, dan **BERTopic**, dengan data yang disimpan dalam basis data MySQL. Aplikasi web dibangun menggunakan **Flask** sebagai backend dan ditampilkan dalam antarmuka sederhana.

## 📁 Struktur Proyek

- `app.py`: File utama Flask.
- `venv/`: Virtual environment Python.
- `templates/`: Folder berisi template HTML untuk frontend.
- `static/`: Folder berisi file CSS, JS, atau gambar.
- `db_lda_tf-idf_bertopic.sql`: File SQL untuk import database awal.

---

## 🔧 Persiapan Awal

### 1. Buat Database MySQL

Buat database baru dengan nama:

```sql
CREATE DATABASE db_topiclab;
```

### 2. Import File SQL

Import struktur dan data awal ke dalam database menggunakan file `db_lda_tf-idf_bertopic.sql`. Anda bisa menggunakan perintah berikut di terminal atau lewat phpMyAdmin:

```bash
mysql -u [username] -p db_topiclab < db_lda_tf-idf_bertopic.sql
```

Ganti `[username]` sesuai dengan username MySQL Anda.

---

## 🐍 Setup Lingkungan Virtual

Buka project di **VSCode**, lalu jalankan perintah berikut di terminal untuk membuat dan mengaktifkan lingkungan virtual:

### Membuat Virtual Environment

```bash
python -m venv venv
```

### Mengaktifkan Virtual Environment

Pada Windows:
```bash
venv\Scripts\Activate
```

Pada macOS/Linux:
```bash
source venv/bin/activate
```

---

## 📦 Instalasi Dependensi

Pastikan Anda telah menyalin file `requirements.txt` atau instal manual dependensi yang diperlukan seperti:

```bash
pip install flask mysql-connector-python bertopic nltk gensim scikit-learn pandas
```

---

## ▶️ Menjalankan Aplikasi

Setelah semua persiapan selesai, Anda dapat menjalankan aplikasi menggunakan salah satu cara berikut:

### Jalankan dengan Flask

```bash
flask run
```

Pastikan variabel lingkungan `FLASK_APP` sudah diatur:

```bash
set FLASK_APP=app.py
```
(Windows)

atau

```bash
export FLASK_APP=app.py
```
(macOS/Linux)

### Atau langsung jalankan app.py

```bash
python app.py
```

Akses aplikasi melalui browser:  
👉 http://127.0.0.1:5000

---

## 📝 Catatan Penting

- Pastikan koneksi ke database berhasil (cek konfigurasi di `app.py`).
- Jika terjadi error saat eksekusi model BERTopic, pastikan komputer Anda memiliki cukup RAM atau gunakan GPU.
- Beberapa model memerlukan waktu lama untuk load pertama kali.

---

## 🚀 Fitur Aplikasi

- Tampilan dashboard topic modelling.
- Pemrosesan teks dengan TF-IDF.
- Topic extraction menggunakan LDA.
- Topic modeling lanjutan dengan BERTopic.
- Data hasil disimpan ke dalam database.

---


## 📞 Dukungan & Kontak

Jika ada pertanyaan, kesulitan, atau ingin kolaborasi, silakan hubungi kami melalui email atau media sosial yang tertera.

---

## 🙏 Terima Kasih!
