-- phpMyAdmin SQL Dump
-- version 5.2.1
-- https://www.phpmyadmin.net/
--
-- Host: localhost:3306
-- Generation Time: May 05, 2025 at 07:07 PM
-- Server version: 8.0.30
-- PHP Version: 8.3.11

SET SQL_MODE = "NO_AUTO_VALUE_ON_ZERO";
START TRANSACTION;
SET time_zone = "+00:00";


/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8mb4 */;

--
-- Database: `db_lda`
--

-- --------------------------------------------------------

--
-- Table structure for table `documents`
--

CREATE TABLE `documents` (
  `id` int NOT NULL,
  `title` varchar(255) DEFAULT NULL,
  `author` text,
  `abstract` text,
  `year` int DEFAULT NULL,
  `venue` varchar(255) DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

--
-- Dumping data for table `documents`
--

INSERT INTO `documents` (`id`, `title`, `author`, `abstract`, `year`, `venue`) VALUES
(21, 'Analisis Sentimen Relokasi Ibukota Nusantara Menggunakan Algoritma Naïve Bayes dan KNN', 'SD Prasetyo, SS Hilabi, F Nurapriani', 'untuk melakukan analisis sentimen masyarakat Indonesia  Bayes (NB) dan K-Nearest  Neighbor (KNN). Hasil penelitian  memberikan tingkat akurasi analisis sentimen sebesar 82.27%,', 2023, 'Jurnal KomtekInfo'),
(22, 'Analisis Sentimen pada Ulasan pengguna Aplikasi Bibit Dan Bareksa dengan Algoritma KNN', 'ADA Putra, S Juanita', '“Analisis Sentimen pada ulasan Aplikasi BCA Mobile menggunakan BM25 dan Improved  K-Nearest Neighbor”  dokumen menggunakan Improved K-Nearest Neighbor dan hasil yang', 2021, 'JATISI (Jurnal Teknik Informatika dan …'),
(23, 'Analisis Sentimen Masyarakat Terhadap Penggunaan E-Commerce Menggunakan Algoritma K-Nearest Neighbor', 'IH Kusuma, N Cahyono', 'dilakukan analisis sentimen publik terkait penggunaannya.  analisis sentimen terhadap  opini publik terhadap penggunaan Shopee dengan menggunakan algoritma K-Nearest Neighbor', 2023, 'Jurnal Informatika: Jurnal …'),
(24, 'Perbandingan Metode KNN, Decision Tree, dan Naïve Bayes Terhadap Analisis Sentimen Pengguna Layanan BPJS', 'R Puspita, A Widodo', 'Penulis menggunakan metode KNN, Decision Tree, dan Naïve Bayes untuk  analisis  sentimen terhadap data Twitter terhadap layanan BPJS dengan menggunakan metode KNN', 2021, 'Jurnal Informatika Universitas …'),
(25, 'Penerapan Algoritma K-Nearest Neighbor (K-NN) Untuk Analisis Sentimen Publik Terhadap Pembelajaran Daring', 'J Supriyanto, D Alita, AR Isnain', 'Analisis; TF-IDF; Abstract: This research was conducted to apply the KNN (K-Nearest  Neighbor) algorithm in conducting sentiment analysis of Twitter users on issues related to', 2023, 'J. Inform. Dan …'),
(26, 'Perbandingan Metode Naive Bayes, Knn Dan Decision Tree Terhadap Analisis Sentimen Transportasi Krl Commuter Line', 'NT Romadloni, I Santoso', 'Oleh karena itu dalam penelitian ini melakukan analisis sentimen terhadap pengguna KRL   Pada penelitian ini menggunakan metode Naive Bayes Classifier, KNN dan Desicion Tree', 2019, 'IKRA-ITH Informatika …'),
(27, 'Analisis Sentimen dengan SVM, NAIVE BAYES dan KNN untuk Studi Tanggapan Masyarakat Indonesia Terhadap Pandemi Covid-19 pada Media Sosial Twitter', 'FS Pamungkas, I Kharisudin', 'analisis sentimen dengan algoritma machine learning. Pada penelitian ini dilakukan analisis  sentimen  (SVM), Naive Bayes, dan K-Nearest Neighbor, yang kemudian ketiga algoritma', 2021, 'PRISMA, Prosiding Seminar …'),
(28, 'Penerapan Algoritma KNN Pada Analisis Sentimen Review Aplikasi Peduli Lindungi', 'P Astuti, N Nuris', 'analisis sentiment review terhadap aplikasi PeduliLindungi menggunakan algoritma K-Nearest  Neighbor  text mining dengan metode K-Nearest Neighbor (K-NN) termasuk kedalam', 2022, 'Computer Science (Co-Science)'),
(29, 'Penerapan analisis sentimen pada pengguna twitter menggunakan metode K-Nearest Neighbor', 'A Deviyanto, MDR Wahyudi', 'algoritma KNN (K - Nearest Neighbor) dalam analisis sentimen  KNN dengan pembobotan  kata TF-IDF dan fungsi Cosine Similarity, akan dilakukan pengklasifikasian nilai sentimen ke', 2018, 'JISKA (Jurnal Informatika …'),
(30, 'Analisis sentimen mengenai vaksin sinovac menggunakan algoritma support vector machine (SVM) dan k-nearest neighbor (KNN)', 'A Baita, Y Pristyanto, N Cahyono', 'Sentiment analysis can be used to analyze public opinion on the administration of this vaccine.  In this study, the SVM and KNN algorithms were used to analyze public sentiment  KNN', 2021, 'Information System Journal'),
(31, 'Sistem Informasi Penjualan', 'A Selay, GD Andigha, A Alfarizi, MIB Wahyudi', 'Sistem Penjualan sangat mempermudah dan membantu  sebuah informasi contohnya adalah  sistem informasi penjualan,  dokumen,, dan informasi sebuah penjualan untuk keperluan', 2023, 'Karimah …'),
(32, 'Perancangan sistem informasi penjualan online studi kasus tokoku', 'FE Nugroho', 'sistem, kami menemukan beberapa kelemahan pada sistem yang sedang berjalan. Maka  pada tahap perancangan ini kami akan membuat sebuah sistem informasi penjualan  sistem', 2016, 'Jurnal Simetris'),
(33, 'Sistem informasi penjualan sandal berbasis web', 'RF Ahmad, N Hasti', 'Untuk merancang Sistem Informasi penjualan berbasis web pada toko cucko. 3. Untuk   Sistem Informasi penjualan sandal di toko cucko. 4. Unuk mengimplementasikan SItem Informasi', 2018, 'Jurnal Teknologi Dan Informasi'),
(34, 'Perancangan Sistem Informasi Penjualan Berbasis Web (Studi Kasus Pada Newbiestore)', 'D Zaliluddin, R Rohmat', 'untuk E-Comerce dapat membantu mendongkrak penjualan dengan pasar yang lebih luas.  Dalam jurnal ini diuraikan Penerapan e-commerce dalam penjualan sebuah distro pakaian', 2018, 'INFOTECH journal'),
(35, 'Penggunaan metode waterfall dalam rancang bangun sistem informasi penjualan', 'H Nur', 'dalam sebuah organisasi adalah penggunaan sistem informasi pada sebuah organisasi   sistem informasi penjualan yang berjalan pada saat ini. Hal ini bertujuan supaya sistem baru', 2019, 'Generation Journal'),
(36, 'Perancangan Sistem Informasi Penjualan Online', 'SRC Nursari, Y Immanuel', ', dan menyebarkan informasi untuk tujuan yang spesifik. Dengan demikian dapat  disimpulkan bahwa Sistem Informasi adalah teknologi informasi untuk mendukung operasi dan', 2017, 'Jurnal TAM (Technology …'),
(37, 'Sistem informasi penjualan pupuk berbasis e-commerce', 'R Novita, N Sari', '', 2015, 'Jurnal Teknoif Teknik Informatika Institut …'),
(38, 'Aplikasi Sistem Informasi Penjualan Berbasis Web Pada Toko Velg YQ', 'J Bernadi', 'menjadi faktor pembuatan sistem informasi penjualan velg,  sistem informasi penjualan  velg pada toko velg YQ, untuk mempermudah proses penjualan serta memberikan informasi', 2013, 'ComTech: Computer, Mathematics and Engineering …'),
(39, 'Metode Waterfall Untuk Sistem Informasi Penjualan', 'A Abdurrahman, S Masripah', 'penjualan masuk kedalam sistem informasi penjualan.  Sistem informasi penjualan pada  Toko Kue MANIKA dan dengan alternatif pemecahannya yaitu dengan membangun sistem', 2017, 'Information System For …');

--
-- Indexes for dumped tables
--

--
-- Indexes for table `documents`
--
ALTER TABLE `documents`
  ADD PRIMARY KEY (`id`);

--
-- AUTO_INCREMENT for dumped tables
--

--
-- AUTO_INCREMENT for table `documents`
--
ALTER TABLE `documents`
  MODIFY `id` int NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=40;
COMMIT;

/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
