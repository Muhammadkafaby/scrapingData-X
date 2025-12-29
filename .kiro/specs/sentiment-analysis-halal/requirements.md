# Requirements Document

## Introduction

Dokumen ini mendefinisikan kebutuhan untuk sistem analisis sentimen Twitter mengenai komentar masyarakat tentang sertifikasi halal di Indonesia. Penelitian ini menggunakan pendekatan hybrid dengan kombinasi:
1. **K-Means Clustering** (unsupervised) - untuk eksplorasi awal dan pengelompokan data
2. **Naive Bayes** (supervised) - untuk klasifikasi sentimen setelah labeling

Sentimen diklasifikasikan menjadi kategori: positif, negatif, dan netral. Sistem ini dirancang untuk keperluan akademis dalam konteks proposal atau skripsi penelitian.

**Status Implementasi:**
- ✅ Scraping data sudah tersedia (`Scraping.ipynb`)
- ✅ Merge data sudah tersedia (`mergeData.ipynb`)
- ✅ Data cleaning dasar sudah tersedia (`CleaningData.ipynb`)
- ⏳ Perlu enhancement: Text preprocessing, TF-IDF, K-Means, Naive Bayes, Visualisasi

## Glossary

- **Sertifikasi Halal**: Proses sertifikasi yang menjamin produk sesuai dengan syariat Islam
- **Sentiment Analysis**: Proses komputasional untuk mengidentifikasi dan mengkategorikan opini dalam teks
- **K-Means Clustering**: Algoritma unsupervised learning untuk mengelompokkan data ke dalam k cluster
- **Naive Bayes**: Algoritma supervised learning berbasis probabilitas Bayes untuk klasifikasi teks
- **TF-IDF**: Term Frequency-Inverse Document Frequency, metode pembobotan kata dalam dokumen
- **Stopword**: Kata-kata umum yang tidak memiliki makna signifikan dalam analisis teks
- **Stemming**: Proses mengubah kata ke bentuk dasarnya
- **Case Folding**: Proses mengubah semua huruf menjadi huruf kecil
- **Tokenizing**: Proses memecah teks menjadi unit-unit kecil (token)
- **Elbow Method**: Teknik untuk menentukan jumlah cluster optimal berdasarkan inertia
- **Silhouette Score**: Metrik untuk mengukur kualitas clustering
- **Confusion Matrix**: Tabel untuk mengevaluasi performa model klasifikasi
- **Precision**: Rasio prediksi positif yang benar terhadap total prediksi positif
- **Recall**: Rasio prediksi positif yang benar terhadap total data positif aktual
- **F1-Score**: Rata-rata harmonik dari precision dan recall

## Requirements

### Requirement 1: Data Collection (Scraping Twitter) ✅ SUDAH ADA

**User Story:** Sebagai peneliti, saya ingin mengumpulkan data tweet terkait sertifikasi halal, sehingga saya memiliki dataset untuk dianalisis.

**File:** `Scraping.ipynb` - menggunakan tweet-harvest dengan Node.js

#### Acceptance Criteria

1. WHEN peneliti menjalankan proses scraping THEN sistem SHALL mengumpulkan tweet berdasarkan kata kunci terkait sertifikasi halal
2. WHEN data tweet dikumpulkan THEN sistem SHALL menyimpan informasi: full_text, created_at, user_id_str, tweet_url
3. WHEN proses scraping selesai THEN sistem SHALL menyimpan data dalam format CSV
4. IF terjadi error koneksi atau rate limit THEN sistem SHALL menangani error dengan graceful

### Requirement 2: Data Merging ✅ SUDAH ADA

**User Story:** Sebagai peneliti, saya ingin menggabungkan data dari berbagai sumber scraping, sehingga dataset lebih komprehensif.

**File:** `mergeData.ipynb`

#### Acceptance Criteria

1. WHEN merge dijalankan THEN sistem SHALL menggabungkan semua file CSV dari folder data
2. WHEN merge selesai THEN sistem SHALL menghapus kolom yang tidak diperlukan (Unnamed: 0)
3. WHEN merge selesai THEN sistem SHALL reset index dan menyimpan ke `dataSertifikasiHalal.csv`

### Requirement 3: Data Cleaning ⏳ PERLU ENHANCEMENT

**User Story:** Sebagai peneliti, saya ingin membersihkan data tweet dari noise, sehingga data siap untuk preprocessing lebih lanjut.

**File:** `CleaningData.ipynb` - perlu ditambahkan fungsi cleaning lanjutan

#### Acceptance Criteria

1. WHEN data cleaning dijalankan THEN sistem SHALL menghapus URL dari teks tweet menggunakan regex
2. WHEN data cleaning dijalankan THEN sistem SHALL menghapus mention (@username) dari teks tweet
3. WHEN data cleaning dijalankan THEN sistem SHALL menghapus hashtag (#topic) dari teks tweet
4. WHEN data cleaning dijalankan THEN sistem SHALL menghapus emoji dan karakter khusus dari teks tweet
5. WHEN data cleaning dijalankan THEN sistem SHALL menghapus angka dari teks tweet
6. WHEN data cleaning dijalankan THEN sistem SHALL menghapus tweet duplikat berdasarkan konten teks
7. WHEN data cleaning dijalankan THEN sistem SHALL menghapus whitespace berlebih
8. WHEN data cleaning selesai THEN sistem SHALL menyimpan hasil cleaning ke file baru

### Requirement 4: Text Preprocessing 🆕 BARU

**User Story:** Sebagai peneliti, saya ingin melakukan preprocessing teks sesuai standar NLP bahasa Indonesia, sehingga teks siap untuk transformasi fitur.

**File:** Akan dibuat `Preprocessing.ipynb`

#### Acceptance Criteria

1. WHEN preprocessing dijalankan THEN sistem SHALL melakukan case folding (mengubah semua huruf menjadi lowercase)
2. WHEN preprocessing dijalankan THEN sistem SHALL melakukan tokenizing (memecah kalimat menjadi kata-kata)
3. WHEN preprocessing dijalankan THEN sistem SHALL menghapus stopword bahasa Indonesia menggunakan library Sastrawi atau NLTK
4. WHEN preprocessing dijalankan THEN sistem SHALL melakukan stemming menggunakan algoritma Sastrawi untuk bahasa Indonesia
5. WHEN preprocessing dijalankan THEN sistem SHALL menggabungkan token kembali menjadi teks bersih
6. WHEN preprocessing selesai THEN sistem SHALL menyimpan hasil preprocessing ke kolom baru atau file baru

### Requirement 5: Feature Extraction dengan TF-IDF 🆕 BARU

**User Story:** Sebagai peneliti, saya ingin mentransformasi teks menjadi representasi numerik menggunakan TF-IDF, sehingga data dapat diproses oleh algoritma clustering.

**File:** Akan dibuat `TF-IDF_KMeans.ipynb`

#### Acceptance Criteria

1. WHEN transformasi TF-IDF dijalankan THEN sistem SHALL menggunakan TfidfVectorizer dari scikit-learn
2. WHEN transformasi TF-IDF dijalankan THEN sistem SHALL mengatur parameter max_features untuk membatasi jumlah fitur
3. WHEN transformasi TF-IDF dijalankan THEN sistem SHALL menghasilkan matriks sparse TF-IDF
4. WHEN transformasi selesai THEN sistem SHALL menampilkan dimensi matriks dan top features

### Requirement 6: K-Means Clustering untuk Eksplorasi Data 🆕 BARU

**User Story:** Sebagai peneliti, saya ingin mengelompokkan tweet menggunakan K-Means Clustering, sehingga pola sentimen dapat diidentifikasi secara unsupervised sebagai dasar labeling.

**File:** Akan dibuat dalam `KMeans_Clustering.ipynb`

#### Acceptance Criteria

1. WHEN penentuan jumlah cluster dilakukan THEN sistem SHALL menggunakan metode Elbow dengan range k=2 sampai k=10
2. WHEN penentuan jumlah cluster dilakukan THEN sistem SHALL menghitung Silhouette Score untuk setiap nilai k
3. WHEN K-Means dijalankan THEN sistem SHALL mengelompokkan data ke dalam k cluster optimal (default k=3)
4. WHEN clustering selesai THEN sistem SHALL memberikan label cluster pada setiap tweet
5. WHEN interpretasi dilakukan THEN sistem SHALL mengekstrak top-N kata dominan di setiap cluster
6. WHEN interpretasi dilakukan THEN sistem SHALL melakukan labeling manual sentimen berdasarkan kata dominan

### Requirement 7: Naive Bayes Classification 🆕 BARU

**User Story:** Sebagai peneliti, saya ingin mengklasifikasikan sentimen tweet menggunakan Naive Bayes, sehingga model dapat memprediksi sentimen secara otomatis.

**File:** Akan dibuat dalam `NaiveBayes_Classification.ipynb`

#### Acceptance Criteria

1. WHEN data labeling selesai THEN sistem SHALL membagi data menjadi training set dan testing set (80:20 atau 70:30)
2. WHEN training dijalankan THEN sistem SHALL melatih model Multinomial Naive Bayes dengan data TF-IDF
3. WHEN evaluasi dilakukan THEN sistem SHALL menghitung accuracy, precision, recall, dan F1-score
4. WHEN evaluasi dilakukan THEN sistem SHALL menghasilkan confusion matrix
5. WHEN evaluasi dilakukan THEN sistem SHALL menghasilkan classification report per kelas sentimen
6. WHEN model selesai THEN sistem SHALL menyimpan model untuk prediksi data baru

### Requirement 8: Visualisasi Hasil 🆕 BARU

**User Story:** Sebagai peneliti, saya ingin memvisualisasikan hasil analisis sentimen, sehingga hasil penelitian mudah dipahami dan dipresentasikan.

**File:** Akan dibuat `Visualisasi.ipynb`

#### Acceptance Criteria

1. WHEN visualisasi dijalankan THEN sistem SHALL menghasilkan WordCloud untuk setiap cluster/sentimen
2. WHEN visualisasi dijalankan THEN sistem SHALL menghasilkan grafik distribusi sentimen (pie chart dan bar chart)
3. WHEN visualisasi dijalankan THEN sistem SHALL menghasilkan visualisasi Elbow Method
4. WHEN visualisasi dijalankan THEN sistem SHALL menghasilkan visualisasi Silhouette Score
5. WHEN visualisasi dijalankan THEN sistem SHALL menghasilkan scatter plot hasil clustering menggunakan PCA 2D
6. WHEN visualisasi dijalankan THEN sistem SHALL menghasilkan confusion matrix heatmap untuk Naive Bayes
7. WHEN visualisasi selesai THEN sistem SHALL menyimpan semua grafik dalam format PNG

### Requirement 9: Dokumentasi dan Kesimpulan 🆕 BARU

**User Story:** Sebagai peneliti, saya ingin mendokumentasikan hasil penelitian dengan kesimpulan yang jelas, sehingga penelitian dapat dipertanggungjawabkan secara akademis.

#### Acceptance Criteria

1. WHEN analisis selesai THEN sistem SHALL menghasilkan ringkasan statistik deskriptif dataset (jumlah data, distribusi per cluster/sentimen)
2. WHEN analisis selesai THEN sistem SHALL mendokumentasikan parameter yang digunakan (max_features TF-IDF, jumlah cluster, random_state)
3. WHEN analisis selesai THEN sistem SHALL menyajikan perbandingan hasil K-Means dan Naive Bayes
4. WHEN analisis selesai THEN sistem SHALL mencatat kelebihan dan keterbatasan masing-masing metode
5. WHEN analisis selesai THEN sistem SHALL menyimpulkan sentimen dominan masyarakat terhadap sertifikasi halal
