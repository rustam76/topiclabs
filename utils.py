import re
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from langdetect import detect

import nltk
from nltk.corpus import stopwords
import spacy
from nltk.corpus import stopwords


nlp_en = spacy.load("en_core_web_sm")

# Buat stemmer untuk bahasa Indonesia
stemmer_factory = StemmerFactory()
stemmer_id = stemmer_factory.create_stemmer()

# Load stopword NLTK
nltk.download('stopwords')
stopwords_en = set(stopwords.words('english'))
stopwords_id = set(stopwords.words('indonesian'))

# Gabungkan dan tambahkan stopword tambahan
custom_stopwords = {'diperlukan', 'hendaknya', 'tapi', 'dimungkinkan', 'hendaklah', 'umumnya', 'tambahnya', 'usai', 'katakan', 'sebagaimana', 'sekali', 'persoalan', 'waduh', 'bermaksud', 'jelaslah', 'ditanyai', 'tiba', 'terdahulu', 'menghendaki', 'tidak', 'sangatlah', 'kalaulah', 'rata', 'tadi', 'sendirinya', 'tersampaikan', 'sekadar', 'mengakhiri', 'mempergunakan', 'sedikit', 'sekali-kali', 'katakanlah', 'karenanya', 'oleh', 'semampunya', 'diakhirinya', 'kapanpun', 'setidaknya', 'disini', 'menaiki', 'tentunya', 'terbanyak', 'tak', 'secara', 'diibaratkannya', 'mengatakan', 'hendak', 'dikarenakan', 'sekarang', 'berturut', 'ditanyakan', 'terlihat', 'diperlukannya', 'sebuah', 'cuma', 'ingat-ingat', 'sesegera', 'mengerjakan', 'keinginan', 'berlebihan', 'apalagi', 'siapapun', 'enggaknya', 'lagi', 'diungkapkan', 'bisa', 'tentu', 'bersiap', 'dia', 'ia', 'ini', 'dituturkan', 'mendatang', 'semacam', 'sebenarnya', 'terutama', 'diibaratkan', 'tunjuk', 'inilah', 'diri', 'seterusnya', 'menandaskan', 'kenapa', 'dimulailah', 'mengibaratkan', 'wong', 'disinilah', 'bahkan', 'kelihatan', 'sudahkah', 'mempertanyakan', 'dalam', 'luar', 'memulai', 'mengucapkan', 'selalu', 'waktu', 'ataukah', 'wahai', 'beberapa', 'semuanya', 'mampu', 'sebagainya', 'memungkinkan', 'bukannya', 'jadi', 'menanyakan', 'percuma', 'bolehkah', 'sekurang-kurangnya', 'yakin', 'memperbuat', 'jadinya', 'belumlah', 'terdiri', 'menjadi', 'sekalipun', 'merekalah', 'melihat', 'terakhir', 'hari', 'wah', 'sesuatu', 'sebelum', 'mendapat', 'berapa', 'dulu', 'sudah', 'tidaklah', 'kurang', 'makanya', 'ditunjuk', 'akhiri', 'bila', 'sayalah', 'buat', 'segalanya', 'berjumlah', 'perlunya', 'apatah', 'begitukah', 'itu', 'cara', 'antara', 'sampaikan', 'amat', 'mulailah', 'tertentu', 'setibanya', 'tiga', 'maka', 'semasih', 'nyaris', 'masalah', 'sebaik-baiknya', 'pasti', 'tiba-tiba', 'awal', 'bermula', 'tegasnya', 'bukanlah', 'selamanya', 'bermacam', 'satu', 'merupakan', 'disampaikan', 'sebanyak', 'menuturkan', 'segera', 'diucapkan', 'mendatangi', 'dipergunakan', 'bertanya-tanya', 'berkata', 'memintakan', 'jelas', 'kapan', 'tanyanya', 'tetapi', 'anda', 'benar', 'semula', 'sejenak', 'perlu', 'semakin', 'memang', 'begini', 'kemudian', 'serupa', 'disebutkan', 'pun', 'turut', 'bahwasanya', 'pastilah', 'nanti', 'didatangkan', 'dan', 'sedangkan', 'dikira', 'tentang', 'tersebutlah', 'diminta', 'dituturkannya', 'cukup', 'lanjutnya', 'dibuatnya', 'ucapnya', 'baru', 'haruslah', 'meminta', 'dijelaskan', 'kelihatannya', 'lainnya', 'ada', 'ibaratnya', 'ingin', 'menyangkut', 'mendapatkan', 'pentingnya', 'dirinya', 'dialah', 'diantaranya', 'terjadilah', 'ditujukan', 'bahwa', 'nah', 'mengibaratkannya', 'terhadap', 'saat', 'ditanya', 'ikut', 'mulanya', 'bakalan', 'setiba', 'tiap', 'bagaimana', 'sela', 'diberikannya', 'hanya', 'mengingat', 'meski', 'sebutlah', 'diinginkan', 'kata', 'hingga', 'usah', 'dikatakannya', 'apabila', 'per', 'manakala', 'untuk', 'sebegini', 'yakni', 'bertanya', 'olehnya', 'dipersoalkan', 'digunakan', 'ibu', 'teringat-ingat', 'adalah', 'berikan', 'sedemikian', 'sepihak', 'tandasnya', 'tegas', 'berlainan', 'bekerja', 'dini', 'inikah', 'mendatangkan', 'seringnya', 'terjadi', 'belakang', 'lalu', 'bawah', 'kedua', 'berada', 'jelaskan', 'bersiap-siap', 'awalnya', 'asal', 'daripada', 'mungkinkah', 'boleh', 'tutur', 'tengah', 'kasus', 'berikutnya', 'masing-masing', 'keadaan', 'terjadinya', 'meyakini', 'juga', 'ditunjuki', 'manalagi', 'menunjukkan', 'namun', 'bertutur', 'sehingga', 'terus', 'jadilah', 'ternyata', 'sama-sama', 'ditandaskan', 'ibaratkan', 'mirip', 'melihatnya', 'berkali-kali', 'ataupun', 'nyatanya', 'dimulai', 'bagi', 'jawabnya', 'teringat', 'aku', 'tambah', 'sudahlah', 'inginkah', 'seluruh', 'terasa', 'berakhirlah', 'dipertanyakan', 'kan', 'menyampaikan', 'saling', 'dimisalkan', 'sementara', 'beginikah', 'memastikan', 'walaupun', 'dibuat', 'kitalah', 'berkehendak', 'bilakah', 'ujar', 'pertanyakan', 'sendiri', 'jauh', 'dipunyai', 'tanpa', 'kamu', 'menyebutkan', 'berkeinginan', 'seseorang', 'pernah', 'beri', 'siapa', 'termasuk', 'pantas', 'pertama-tama', 'kelamaan', 'memperkirakan', 'semasa', 'didapat', 'belakangan', 'malahan', 'misal', 'mengungkapkan', 'yang', 'menunjuknya', 'setinggi', 'jika', 'sekalian', 'sepantasnyalah', 'kecil', 'masa', 'mau', 'bolehlah', 'lebih', 'lewat', 'betulkah', 'menanti', 'dimaksudnya', 'sebelumnya', 'jumlahnya', 'ditegaskan', 'bukan', 'di', 'mempersiapkan', 'sebesar', 'sekecil', 'bagaimanapun', 'sedikitnya', 'melalui', 'lamanya', 'benarlah', 'misalkan', 'kapankah', 'tetap', 'lagian', 'andalah', 'mengenai', 'mulai', 'mereka', 'bersama-sama', 'selama', 'ucap', 'soal', 'banyak', 'berawal', 'misalnya', 'nantinya', 'berdatangan', 'diketahui', 'jangan', 'suatu', 'biasa', 'seluruhnya', 'menantikan', 'atau', 'diberi', 'seingat', 'adapun', 'diantara', 'sering', 'ditambahkan', 'tuturnya', 'jikalau', 'berlalu', 'sebaliknya', 'begitupun', 'naik', 'diucapkannya', 'kelima', 'sepanjang', 'setiap', 'toh', 'itulah', 'sebaiknya', 'rasa', 'akhir', 'bagaikan', 'panjang', 'bagai', 'lanjut', 'benarkah', 'macam', 'sejumlah', 'menanya', 'semisalnya', 'serta', 'berujar', 'dekat', 'amatlah', 'artinya', 'bagaimanakah', 'khususnya', 'bersama', 'tandas', 'sebisanya', 'sejauh', 'sekitar', 'telah', 'balik', 'itukah', 'terlalu', 'dimaksudkan', 'sesekali', 'sebutnya', 'katanya', 'tidakkah', 'disebutkannya', 'sesudah', 'tampak', 'kalian', 'secukupnya', 'jawab', 'saya', 'masih', 'melakukan', 'pak', 'pula', 'dengan', 'menunjuk', 'sinilah', 'kembali', 'agaknya', 'antaranya', 'jelasnya', 'mengucapkannya', 'gunakan', 'diperkirakan', 'semua', 'tadinya', 'bermacam-macam', 'sebetulnya', 'jangankan', 'apaan', 'caranya', 'berapapun', 'demi', 'diperbuat', 'diperbuatnya', 'kira-kira', 'menginginkan', 'keterlaluan', 'tempat', 'bakal', 'menegaskan', 'tertuju', 'perlukah', 'sebaik', 'kita', 'agar', 'ketika', 'terkira', 'kalau', 'keseluruhannya', 'cukupkah', 'paling', 'seberapa', 'dua', 'selain', 'menyiapkan', 'setelah', 'justru', 'diingat', 'akulah', 'berkenaan', 'walau', 'lah', 'beginian', 'akhirnya', 'dikatakan', 'berapalah', 'soalnya', 'menurut', 'tanyakan', 'menjawab', 'seorang', 'ditunjukkannya', 'apakah', 'tersebut', 'makin', 'mengapa', 'sebagai', 'hanyalah', 'sebegitu', 'cukuplah', 'bukankah', 'sambil', 'dimaksudkannya', 'sesuatunya', 'selama-lamanya', 'sesama', 'hal', 'terdapat', 'apa', 'ialah', 'baik', 'belum', 'setidak-tidaknya', 'bulan', 'menambahkan', 'lama', 'masalahnya', 'mempersoalkan', 'melainkan', 'dahulu', 'berapakah', 'kiranya', 'demikian', 'lain', 'seperlunya', 'tentulah', 'meskipun', 'selaku', 'agak', 'diakhiri', 'saatnya', 'depan', 'dong', 'ungkapnya', 'guna', 'sedang', 'saja', 'kesampaian', 'berikut', 'memisalkan', 'penting', 'mengetahui', 'sekaligus', 'akankah', 'karena', 'pertanyaan', 'harus', 'kemungkinannya', 'semaunya', 'para', 'bung', 'keduanya', 'lima', 'jumlah', 'menyatakan', 'siap', 'kinilah', 'dipastikan', 'memerlukan', 'keluar', 'sama', 'sini', 'datang', 'selanjutnya', 'sajalah', 'sesaat', 'diingatkan', 'dimulainya', 'kalaupun', 'mengingatkan', 'harusnya', 'setempat', 'diperlihatkan', 'inginkan', 'mengatakannya', 'menjelaskan', 'entahlah', 'merasa', 'kini', 'ke', 'pertama', 'seolah-olah', 'berbagai', 'terhadapnya', 'jawaban', 'dapat', 'kebetulan', 'sesudahnya', 'berturut-turut', 'sangat', 'sampai', 'padanya', 'waktunya', 'menanyai', 'demikianlah', 'biasanya', 'betul', 'disebut', 'dilakukan', 'kemungkinan', 'pada', 'padahal', 'empat', 'beginilah', 'sempat', 'minta', 'menuju', 'ditunjuknya', 'se', 'sekurangnya', 'sekadarnya', 'pihak', 'dilihat', 'seolah', 'seperti', 'kepadanya', 'dimaksud', 'bagian', 'enggak', 'punya', 'keseluruhan', 'mampukah', 'adanya', 'tepat', 'menanti-nanti', 'begitulah', 'terlebih', 'maupun', 'sewaktu', 'rasanya', 'semata', 'menunjuki', 'dari', 'kamulah', 'sejak', 'kala', 'sekitarnya', 'begitu', 'seharusnya', 'kok', 'sampai-sampai', 'ditunjukkan', 'mengira', 'masing', 'supaya', 'diketahuinya', 'pukul', 'menyeluruh', 'semata-mata', 'berlangsung', 'tahun', 'diberikan', 'rupanya', 'tampaknya', 'menggunakan', 'atas', 'bisakah', 'tinggi', 'kamilah', 'mempunyai', 'pihaknya', 'berakhir', 'sepertinya', 'ujarnya', 'dikerjakan', 'sana', 'ungkap', 'berakhirnya', 'seketika', 'siapakah', 'umum', 'meyakinkan', 'sebabnya', 'membuat', 'dijelaskannya', 'kira', 'kepada', 'yaitu', 'seenaknya', 'malah', 'ibarat', 'janganlah', 'memihak', 'memberi', 'berarti', 'semampu', 'entah', 'sebut', 'segala', 'mungkin', 'memperlihatkan', 'sekiranya', 'hampir', 'tanya', 'berupa', 'sebagian', 'akan', 'semisal', 'besar', 'sebab', 'sesampai', 'dijawab', 'ingat', 'asalkan', 'sepantasnya', 'setengah', 'tahu', 'antar', 'dilalui', 'mana', 'seusai', 'masihkah', 'mula', 'memberikan', 'sendirian', 'kami', 'dimintai', 'bapak'}

all_stopwords = stopwords_en.union(stopwords_id).union(custom_stopwords)


def preprocess(text):
    if not text or len(text.strip()) == 0:
        return []

    # Bersihkan teks dasar
    text = re.sub(r'\S*@\S*\s?', '', text)     # Email
    text = re.sub(r'<[^>]+>', '', text)        # HTML tags
    text = re.sub(r'[^\w\s]', '', text)        # Tanda baca
    text = text.lower().strip()

    # Tokenisasi sederhana
    tokens = text.split()

    # Deteksi bahasa
    try:
        lang = detect(text)
    except:
        lang = 'unknown'

    # Filter stopword
    tokens = [t for t in tokens if t not in all_stopwords and len(t) > 2]

    # Lemmatisasi atau Stemming berdasarkan bahasa
    if lang == 'en':
        doc = nlp_en(text)
        tokens = [token.lemma_ for token in doc if token.text in tokens]
    elif lang == 'id':
        tokens = [stemmer_id.stem(t) for t in tokens]

    return tokens