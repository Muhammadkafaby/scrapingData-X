"""
Analisis Sentimen Data Sertifikasi Halal dengan K-Means Clustering
===================================================================
Pipeline ini mengikuti susunan yang sama seperti notebook contoh:
1. Load Data
2. Preprocessing (Case Folding, Cleaning, Tokenizing, Stopword Removal, Stemming)
3. TF-IDF Vectorization
4. K-Means Clustering
5. Evaluasi dan Visualisasi
"""

import pandas as pd
import numpy as np
import re
import string
import warnings
warnings.filterwarnings('ignore')

# NLP Libraries
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
import nltk
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('punkt_tab')

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# ML Libraries
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Visualization
import matplotlib.pyplot as plt

# =============================================================================
# 1. LOAD DATA
# =============================================================================
print("=" * 60)
print("1. LOADING DATA")
print("=" * 60)

# Load data sertifikasi halal
df = pd.read_csv('../data/dataSertifikasiHalal.csv')
print(f"Total data: {len(df)} baris")
print(f"Columns: {df.columns.tolist()}")

# Tampilkan info data
print("\nInfo Data:")
print(df.info())

# Hapus duplikat berdasarkan full_text
df_clean = df.drop_duplicates(subset=['full_text'])
print(f"\nSetelah hapus duplikat: {len(df_clean)} baris")

# Hapus baris dengan full_text kosong
df_clean = df_clean.dropna(subset=['full_text'])
print(f"Setelah hapus NaN: {len(df_clean)} baris")

# Rename kolom untuk konsistensi dengan notebook contoh
df_clean = df_clean[['full_text', 'created_at', 'username', 'favorite_count', 'retweet_count']].copy()
df_clean.columns = ['content', 'at', 'userName', 'likes', 'retweets']

print("\nContoh data:")
print(df_clean.head())

# =============================================================================
# 2. PREPROCESSING
# =============================================================================
print("\n" + "=" * 60)
print("2. PREPROCESSING DATA")
print("=" * 60)

# 2.1 Case Folding
print("\n2.1 Case Folding...")
df_clean['content_lower'] = df_clean['content'].str.lower()

# 2.2 Cleaning
print("2.2 Cleaning (removing URLs, mentions, hashtags, numbers, punctuation)...")

def clean_text(text):
    if pd.isna(text):
        return ""
    
    # Hapus URL
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    
    # Hapus mentions (@username)
    text = re.sub(r'@\w+', '', text)
    
    # Hapus hashtags (#hashtag)
    text = re.sub(r'#\w+', '', text)
    
    # Hapus angka
    text = re.sub(r'\d+', '', text)
    
    # Hapus tanda baca
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Hapus karakter spesial
    text = re.sub(r'[^\w\s]', '', text)
    
    # Hapus newline dan tab
    text = re.sub(r'[\n\t\r]', ' ', text)
    
    # Hapus spasi berlebih
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

df_clean['content_clean'] = df_clean['content_lower'].apply(clean_text)

# 2.3 Tokenizing
print("2.3 Tokenizing...")

def tokenize_text(text):
    if pd.isna(text) or text == "":
        return []
    return word_tokenize(text)

df_clean['content_tokens'] = df_clean['content_clean'].apply(tokenize_text)

# 2.4 Stopword Removal
print("2.4 Stopword Removal...")

# Gabungkan stopwords dari NLTK Indonesia dan Sastrawi
stop_words_nltk = set(stopwords.words('indonesian'))
stop_factory = StopWordRemoverFactory()
stop_words_sastrawi = set(stop_factory.get_stop_words())

# Custom stopwords untuk konteks sertifikasi halal
custom_stopwords = {
    'yang', 'dan', 'di', 'ke', 'dari', 'ini', 'itu', 'untuk', 'dengan',
    'adalah', 'pada', 'juga', 'tidak', 'ada', 'akan', 'bisa', 'sudah',
    'ya', 'ga', 'gak', 'ngga', 'nggak', 'aja', 'saja', 'kan', 'dong',
    'sih', 'nih', 'tuh', 'deh', 'yuk', 'yg', 'dgn', 'utk', 'dlm', 'krn',
    'rt', 'amp', 'via'
}

all_stopwords = stop_words_nltk.union(stop_words_sastrawi).union(custom_stopwords)

def remove_stopwords(tokens):
    return [word for word in tokens if word not in all_stopwords and len(word) > 2]

df_clean['content_no_stopword'] = df_clean['content_tokens'].apply(remove_stopwords)

# 2.5 Stemming
print("2.5 Stemming (ini membutuhkan waktu...)...")

stem_factory = StemmerFactory()
stemmer = stem_factory.create_stemmer()

def stem_tokens(tokens):
    return [stemmer.stem(word) for word in tokens]

df_clean['content_stemmed'] = df_clean['content_no_stopword'].apply(stem_tokens)

# 2.6 Join tokens menjadi string
print("2.6 Joining tokens...")
df_clean['content_final'] = df_clean['content_stemmed'].apply(lambda x: ' '.join(x))

# Hapus baris dengan content_final kosong
df_clean = df_clean[df_clean['content_final'].str.len() > 0]
print(f"\nData setelah preprocessing: {len(df_clean)} baris")

print("\nContoh hasil preprocessing:")
print(df_clean[['content', 'content_final']].head(3))

# =============================================================================
# 3. TF-IDF VECTORIZATION
# =============================================================================
print("\n" + "=" * 60)
print("3. TF-IDF VECTORIZATION")
print("=" * 60)

# TF-IDF Vectorizer
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
tfidf_matrix = tfidf.fit_transform(df_clean['content_final'])

print(f"TF-IDF Matrix shape: {tfidf_matrix.shape}")
print(f"Vocabulary size: {len(tfidf.vocabulary_)}")

# Tampilkan top features
feature_names = tfidf.get_feature_names_out()
print(f"\nTop 20 features: {feature_names[:20].tolist()}")

# =============================================================================
# 4. MENENTUKAN JUMLAH CLUSTER OPTIMAL (Elbow & Silhouette)
# =============================================================================
print("\n" + "=" * 60)
print("4. MENENTUKAN JUMLAH CLUSTER OPTIMAL")
print("=" * 60)

# Elbow Method
inertias = []
silhouette_scores = []
K_range = range(2, 11)

print("Menghitung inertia dan silhouette score untuk K=2 hingga K=10...")
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(tfidf_matrix)
    inertias.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(tfidf_matrix, kmeans.labels_))
    print(f"K={k}: Inertia={kmeans.inertia_:.2f}, Silhouette={silhouette_scores[-1]:.4f}")

# Plot Elbow dan Silhouette
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Elbow Plot
axes[0].plot(K_range, inertias, 'bo-', linewidth=2, markersize=8)
axes[0].set_xlabel('Jumlah Cluster (K)', fontsize=12)
axes[0].set_ylabel('Inertia (SSE)', fontsize=12)
axes[0].set_title('Elbow Method', fontsize=14)
axes[0].grid(True, alpha=0.3)

# Silhouette Plot
axes[1].plot(K_range, silhouette_scores, 'go-', linewidth=2, markersize=8)
axes[1].set_xlabel('Jumlah Cluster (K)', fontsize=12)
axes[1].set_ylabel('Silhouette Score', fontsize=12)
axes[1].set_title('Silhouette Score', fontsize=14)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../output/elbow_silhouette_halal.png', dpi=300, bbox_inches='tight')
plt.show()

# Pilih K optimal berdasarkan silhouette score tertinggi
optimal_k = K_range[np.argmax(silhouette_scores)]
print(f"\nK optimal berdasarkan Silhouette Score: {optimal_k}")

# =============================================================================
# 5. K-MEANS CLUSTERING
# =============================================================================
print("\n" + "=" * 60)
print("5. K-MEANS CLUSTERING")
print("=" * 60)

# Gunakan K=3 untuk sentimen (Negatif, Netral, Positif), atau K optimal
# Untuk analisis sentimen, biasanya digunakan K=3
K = 3
print(f"Menggunakan K={K} untuk clustering sentimen (Negatif, Netral, Positif)")

kmeans_final = KMeans(n_clusters=K, random_state=42, n_init=10)
df_clean['cluster'] = kmeans_final.fit_predict(tfidf_matrix)

print(f"\nDistribusi Cluster:")
print(df_clean['cluster'].value_counts().sort_index())

# =============================================================================
# 6. LABELING SENTIMEN
# =============================================================================
print("\n" + "=" * 60)
print("6. LABELING SENTIMEN")
print("=" * 60)

# Analisis centroid untuk menentukan label sentimen
centroids = kmeans_final.cluster_centers_
centroid_norms = np.linalg.norm(centroids, axis=1)

print("Centroid norms per cluster:")
for i, norm in enumerate(centroid_norms):
    print(f"  Cluster {i}: {norm:.4f}")

# Kata-kata positif dan negatif untuk labeling
kata_positif = ['bagus', 'baik', 'senang', 'puas', 'mantap', 'keren', 'hebat', 
                'excellent', 'good', 'great', 'aman', 'terjamin', 'percaya', 
                'halal', 'berkualitas', 'resmi', 'sertifikat', 'jelas']
kata_negatif = ['buruk', 'jelek', 'kecewa', 'marah', 'kesal', 'tidak', 'gagal',
                'ribet', 'susah', 'lama', 'mahal', 'curang', 'haram', 'palsu',
                'bohong', 'tipu', 'masalah', 'komplain', 'protes']

# Hitung skor sentimen per cluster
def hitung_skor_cluster(cluster_id):
    texts = df_clean[df_clean['cluster'] == cluster_id]['content_final']
    all_text = ' '.join(texts)
    
    positif_count = sum(1 for kata in kata_positif if kata in all_text)
    negatif_count = sum(1 for kata in kata_negatif if kata in all_text)
    
    return positif_count - negatif_count

cluster_scores = {i: hitung_skor_cluster(i) for i in range(K)}
print("\nSkor sentimen per cluster:")
for cluster, score in cluster_scores.items():
    print(f"  Cluster {cluster}: {score}")

# Urutkan cluster berdasarkan skor
sorted_clusters = sorted(cluster_scores.items(), key=lambda x: x[1])
label_mapping = {}
labels = ['Negatif', 'Netral', 'Positif']

for i, (cluster_id, score) in enumerate(sorted_clusters):
    label_mapping[cluster_id] = labels[i]

print(f"\nLabel Mapping: {label_mapping}")

# Apply label
df_clean['sentimen'] = df_clean['cluster'].map(label_mapping)

print("\nDistribusi Sentimen:")
print(df_clean['sentimen'].value_counts())

# =============================================================================
# 7. VISUALISASI
# =============================================================================
print("\n" + "=" * 60)
print("7. VISUALISASI")
print("=" * 60)

# Pie Chart Distribusi Sentimen
plt.figure(figsize=(10, 8))
colors = ['#ff6b6b', '#ffd93d', '#6bcb77']
sentimen_counts = df_clean['sentimen'].value_counts()

plt.pie(sentimen_counts.values, 
        labels=sentimen_counts.index, 
        autopct='%1.1f%%',
        colors=colors,
        explode=[0.02] * len(sentimen_counts),
        shadow=True,
        startangle=90)

plt.title('Distribusi Sentimen Data Sertifikasi Halal\n(K-Means Clustering)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('../output/distribusi_sentimen_halal.png', dpi=300, bbox_inches='tight')
plt.show()

# Bar Chart
plt.figure(figsize=(10, 6))
bars = plt.bar(sentimen_counts.index, sentimen_counts.values, color=colors, edgecolor='black')
plt.xlabel('Sentimen', fontsize=12)
plt.ylabel('Jumlah', fontsize=12)
plt.title('Distribusi Sentimen Data Sertifikasi Halal', fontsize=14, fontweight='bold')

# Tambahkan label di atas bar
for bar, count in zip(bars, sentimen_counts.values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10, 
             str(count), ha='center', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('../output/bar_sentimen_halal.png', dpi=300, bbox_inches='tight')
plt.show()

# =============================================================================
# 8. SIMPAN HASIL
# =============================================================================
print("\n" + "=" * 60)
print("8. MENYIMPAN HASIL")
print("=" * 60)

# Simpan hasil clustering
output_df = df_clean[['content', 'at', 'userName', 'content_final', 'cluster', 'sentimen']]
output_df.to_csv('../output/hasil_analisis_sentimen_halal.csv', index=False, encoding='utf-8-sig')
print(f"Hasil disimpan ke: ../output/hasil_analisis_sentimen_halal.csv")

# Simpan ringkasan
ringkasan = {
    'Total Data Awal': len(df),
    'Total Data Setelah Preprocessing': len(df_clean),
    'Jumlah Cluster': K,
    'Silhouette Score': silhouette_score(tfidf_matrix, kmeans_final.labels_),
    'Distribusi Sentimen': df_clean['sentimen'].value_counts().to_dict()
}

print("\n" + "=" * 60)
print("RINGKASAN HASIL ANALISIS")
print("=" * 60)
for key, value in ringkasan.items():
    print(f"{key}: {value}")

print("\n✅ Analisis selesai!")
