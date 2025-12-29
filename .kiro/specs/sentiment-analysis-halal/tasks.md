j# Implementation Plan: Analisis Sentimen Twitter Sertifikasi Halal

## Overview
Rencana implementasi untuk sistem analisis sentimen Twitter mengenai sertifikasi halal menggunakan K-Means Clustering dan Naive Bayes Classification.

---

- [x] 1. Enhancement Data Cleaning




  - [ ] 1.1 Tambahkan fungsi cleaning lanjutan di CleaningData.ipynb
    - Implementasi fungsi `remove_url()` dengan regex pattern untuk URL
    - Implementasi fungsi `remove_mention()` untuk menghapus @username
    - Implementasi fungsi `remove_hashtag()` untuk menghapus #hashtag
    - Implementasi fungsi `remove_emoji()` menggunakan regex atau library emoji
    - Implementasi fungsi `remove_numbers()` untuk menghapus angka
    - Implementasi fungsi `remove_extra_whitespace()` untuk whitespace berlebih
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.7_
  - [x]* 1.2 Write property test untuk cleaning functions


    - **Property 2: Cleaning Removes All Noise Patterns**
    - **Validates: Requirements 3.1, 3.2, 3.3, 3.4, 3.5**


  - [ ] 1.3 Implementasi fungsi `clean_text()` yang menggabungkan semua cleaning
    - Buat fungsi wrapper yang memanggil semua fungsi cleaning secara berurutan
    - Apply ke kolom full_text dan simpan ke kolom cleaned_text
    - _Requirements: 3.1-3.7_
  - [ ] 1.4 Hapus duplikat dan simpan hasil
    - Drop duplicates berdasarkan cleaned_text
    - Simpan hasil ke `data/hasil_cleaning.csv`
    - _Requirements: 3.6, 3.8_




  - [ ]* 1.5 Write property test untuk no duplicates
    - **Property 3: No Duplicates After Cleaning**
    - **Validates: Requirements 3.6**


- [ ] 2. Checkpoint - Pastikan cleaning berjalan dengan benar
  - Ensure all tests pass, ask the user if questions arise.

- [x] 3. Implementasi Text Preprocessing

  - [ ] 3.1 Buat file Preprocessing.ipynb dan setup dependencies
    - Import pandas, re, Sastrawi
    - Install Sastrawi: `pip install Sastrawi`
    - Load data dari `data/hasil_cleaning.csv`
    - _Requirements: 4.1-4.6_

  - [ ] 3.2 Implementasi fungsi case_folding()
    - Konversi semua teks ke lowercase menggunakan `.lower()`
    - _Requirements: 4.1_
  - [ ]* 3.3 Write property test untuk case folding
    - **Property 4: Case Folding Produces Lowercase**
    - **Validates: Requirements 4.1**

  - [ ] 3.4 Implementasi fungsi tokenize()
    - Pecah teks menjadi list kata menggunakan split atau NLTK word_tokenize
    - _Requirements: 4.2_

  - [ ]* 3.5 Write property test untuk tokenization
    - **Property 5: Tokenization Produces Valid Tokens**
    - **Validates: Requirements 4.2**
  - [ ] 3.6 Implementasi fungsi remove_stopwords()
    - Load stopword list bahasa Indonesia dari Sastrawi atau file custom
    - Filter token yang tidak ada dalam stopword list
    - _Requirements: 4.3_




  - [ ]* 3.7 Write property test untuk stopword removal
    - **Property 6: Stopword Removal Excludes All Stopwords**

    - **Validates: Requirements 4.3**
  - [ ] 3.8 Implementasi fungsi stem_words()
    - Gunakan StemmerFactory dari Sastrawi
    - Apply stemming ke setiap token
    - _Requirements: 4.4_
  - [ ] 3.9 Implementasi fungsi preprocess_text() dan simpan hasil
    - Gabungkan semua fungsi preprocessing

    - Join token kembali menjadi string
    - Simpan ke `data/hasil_preprocessing.csv`
    - _Requirements: 4.5, 4.6_


- [ ] 4. Checkpoint - Pastikan preprocessing berjalan dengan benar
  - Ensure all tests pass, ask the user if questions arise.

- [x] 5. Implementasi TF-IDF dan K-Means Clustering

  - [ ] 5.1 Buat file TF-IDF_KMeans.ipynb dan setup
    - Import sklearn (TfidfVectorizer, KMeans), matplotlib, pandas
    - Load data dari `data/hasil_preprocessing.csv`
    - _Requirements: 5.1-5.4, 6.1-6.6_
  - [ ] 5.2 Implementasi TF-IDF Vectorization
    - Inisialisasi TfidfVectorizer dengan max_features=1000
    - Fit dan transform preprocessed text

    - Tampilkan dimensi matriks dan top features
    - _Requirements: 5.1, 5.2, 5.3, 5.4_
  - [ ]* 5.3 Write property test untuk TF-IDF matrix dimensions
    - **Property 7: TF-IDF Matrix Dimensions**
    - **Validates: Requirements 5.2**
  - [ ] 5.4 Implementasi Elbow Method untuk penentuan k optimal
    - Loop k dari 2 sampai 10
    - Hitung inertia untuk setiap k


    - Plot Elbow curve
    - _Requirements: 6.1_
  - [ ] 5.5 Implementasi Silhouette Score analysis
    - Hitung Silhouette Score untuk setiap k
    - Plot Silhouette scores
    - Tentukan k optimal berdasarkan kedua metode
    - _Requirements: 6.2_
  - [ ] 5.6 Implementasi K-Means Clustering
    - Inisialisasi KMeans dengan k optimal (default k=3)
    - Fit dan predict cluster labels
    - Tambahkan kolom cluster_label ke DataFrame
    - _Requirements: 6.3, 6.4_
  - [ ]* 5.7 Write property test untuk valid cluster assignment
    - **Property 8: Valid Cluster Assignment**
    - **Validates: Requirements 6.2, 6.3**
  - [ ] 5.8 Ekstrak top terms per cluster dan interpretasi sentimen
    - Dapatkan cluster centers
    - Identifikasi top 10 kata per cluster
    - Lakukan labeling manual sentimen (positif/negatif/netral)
    - Simpan hasil ke `data/hasil_clustering.csv`
    - _Requirements: 6.5, 6.6_

- [ ] 6. Checkpoint - Pastikan K-Means clustering berjalan dengan benar
  - Ensure all tests pass, ask the user if questions arise.

- [x] 7. Implementasi Naive Bayes Classification

  - [x] 7.1 Buat file NaiveBayes_Classification.ipynb dan setup

    - Import sklearn (MultinomialNB, train_test_split, metrics)
    - Load data dari `data/hasil_clustering.csv` dengan label sentimen
    - _Requirements: 7.1-7.6_
  - [x] 7.2 Implementasi Train-Test Split

    - Split data 80:20 atau 70:30
    - Stratify berdasarkan label sentimen
    - _Requirements: 7.1_
  - [ ]* 7.3 Write property test untuk train-test split integrity
    - **Property 9: Train-Test Split Integrity**
    - **Validates: Requirements 7.1**
  - [x] 7.4 Training Multinomial Naive Bayes

    - Inisialisasi MultinomialNB
    - Fit model dengan training data
    - _Requirements: 7.2_
  - [x] 7.5 Evaluasi Model

    - Predict pada test data
    - Hitung accuracy, precision, recall, F1-score
    - Generate confusion matrix
    - Generate classification report
    - _Requirements: 7.3, 7.4, 7.5_
  - [ ]* 7.6 Write property test untuk valid evaluation metrics
    - **Property 10: Valid Evaluation Metrics**
    - **Validates: Requirements 7.3, 7.4**
  - [x] 7.7 Simpan model dan hasil prediksi

    - Simpan model menggunakan joblib atau pickle
    - Tambahkan kolom predicted_sentiment ke DataFrame
    - Simpan hasil ke `data/hasil_klasifikasi.csv`
    - _Requirements: 7.6_

- [ ] 8. Checkpoint - Pastikan Naive Bayes classification berjalan dengan benar
  - Ensure all tests pass, ask the user if questions arise.

- [x] 9. Implementasi Visualisasi



  - [x] 9.1 Buat file Visualisasi.ipynb dan setup


    - Import matplotlib, seaborn, wordcloud
    - Load data hasil clustering dan klasifikasi
    - Buat folder output/visualizations jika belum ada
    - _Requirements: 8.1-8.7_
  - [x] 9.2 Implementasi WordCloud per cluster/sentimen

    - Generate WordCloud untuk setiap cluster
    - Simpan sebagai PNG
    - _Requirements: 8.1_
  - [x] 9.3 Implementasi grafik distribusi sentimen

    - Buat pie chart distribusi sentimen
    - Buat bar chart distribusi sentimen
    - Simpan sebagai PNG
    - _Requirements: 8.2_
  - [x] 9.4 Implementasi visualisasi Elbow Method dan Silhouette

    - Plot Elbow curve dengan marking k optimal
    - Plot Silhouette scores
    - Simpan sebagai PNG
    - _Requirements: 8.3, 8.4_
  - [x] 9.5 Implementasi scatter plot clustering dengan PCA

    - Reduksi dimensi TF-IDF ke 2D menggunakan PCA
    - Plot scatter dengan warna per cluster
    - Simpan sebagai PNG
    - _Requirements: 8.5_
  - [x] 9.6 Implementasi confusion matrix heatmap

    - Buat heatmap dari confusion matrix Naive Bayes
    - Tambahkan annotations
    - Simpan sebagai PNG
    - _Requirements: 8.6_

- [x] 10. Dokumentasi dan Kesimpulan



  - [x] 10.1 Buat ringkasan statistik deskriptif


    - Jumlah total data
    - Distribusi per cluster dan sentimen
    - Parameter yang digunakan
    - _Requirements: 9.1, 9.2_

  - [ ] 10.2 Dokumentasikan perbandingan hasil
    - Bandingkan hasil K-Means dan Naive Bayes
    - Analisis kesesuaian cluster dengan sentimen

    - _Requirements: 9.3_
  - [ ] 10.3 Tulis kesimpulan penelitian
    - Sentimen dominan masyarakat
    - Kelebihan dan keterbatasan metode
    - Saran untuk penelitian selanjutnya
    - _Requirements: 9.4, 9.5_

- [ ] 11. Final Checkpoint - Pastikan semua komponen terintegrasi
  - Ensure all tests pass, ask the user if questions arise.
