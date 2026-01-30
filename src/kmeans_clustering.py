"""
K-Means Clustering dengan Visualisasi N-gram per Sentimen
==========================================================
Script ini melanjutkan analisis sentimen dengan visualisasi trigram
untuk setiap kategori sentimen (Negatif, Netral, Positif).

Pipeline:
1. Load data hasil preprocessing
2. TF-IDF Vectorization
3. K-Means Clustering
4. Visualisasi Top Trigram per Sentimen
5. Word Cloud per Sentimen
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# Visualization
import matplotlib.pyplot as plt
from collections import Counter

# Optional: WordCloud
try:
    from wordcloud import WordCloud
    WORDCLOUD_AVAILABLE = True
except ImportError:
    WORDCLOUD_AVAILABLE = False
    print("⚠️ WordCloud tidak terinstall. Install dengan: pip install wordcloud")

# =============================================================================
# KONFIGURASI
# =============================================================================
DATA_PATH = '../output/hasil_analisis_sentimen_halal.csv'
OUTPUT_DIR = '../output/'

# Custom stopwords untuk Bahasa Indonesia
CUSTOM_STOPWORDS = {
    'yang', 'dan', 'di', 'ke', 'dari', 'ini', 'itu', 'untuk', 'dengan',
    'adalah', 'pada', 'juga', 'tidak', 'ada', 'akan', 'bisa', 'sudah',
    'ya', 'ga', 'gak', 'ngga', 'nggak', 'aja', 'saja', 'kan', 'dong',
    'sih', 'nih', 'tuh', 'deh', 'yuk', 'yg', 'dgn', 'utk', 'dlm', 'krn',
    'rt', 'amp', 'via', 'loh', 'nya', 'kah', 'pun', 'lah', 'mah', 
    'kok', 'emang', 'gimana', 'gitu', 'gini', 'apa', 'siapa', 'kapan',
    'bagaimana', 'kenapa', 'mana', 'sertifikasi', 'halal', 'sertifikat'
}

# Gabungkan dengan English stopwords
ALL_STOPWORDS = list(CUSTOM_STOPWORDS.union(set(ENGLISH_STOP_WORDS)))

# =============================================================================
# FUNGSI HELPER
# =============================================================================

def get_top_ngrams(corpus, n=3, top_k=10, stopwords=None):
    """
    Mendapatkan top n-grams dari corpus text.
    
    Parameters:
    -----------
    corpus : list or Series
        Kumpulan text untuk dianalisis
    n : int
        Jumlah kata dalam n-gram (1=unigram, 2=bigram, 3=trigram)
    top_k : int
        Jumlah top n-grams yang dikembalikan
    stopwords : list
        Daftar stopwords yang akan diabaikan
    
    Returns:
    --------
    list of tuples : [(ngram, count), ...]
    """
    if stopwords is None:
        stopwords = ALL_STOPWORDS
        
    vec = CountVectorizer(ngram_range=(n, n), stop_words=stopwords).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    return words_freq[:top_k]


def plot_ngrams(corpus, sentiment_label, n=3, top_k=10, stopwords=None, save_path=None):
    """
    Memvisualisasikan top n-grams untuk sentimen tertentu.
    
    Parameters:
    -----------
    corpus : list or Series
        Kumpulan text untuk dianalisis
    sentiment_label : str
        Label sentimen (Negatif, Netral, Positif)
    n : int
        Jumlah kata dalam n-gram
    top_k : int
        Jumlah top n-grams yang ditampilkan
    stopwords : list
        Daftar stopwords
    save_path : str
        Path untuk menyimpan gambar (opsional)
    """
    top_ngrams = get_top_ngrams(corpus, n=n, top_k=top_k, stopwords=stopwords)
    
    if not top_ngrams:
        print(f"⚠️ Tidak ada {n}-gram untuk sentimen {sentiment_label}")
        return
    
    ngrams, counts = zip(*top_ngrams)
    
    # Warna berdasarkan sentimen
    color_map = {
        'Negatif': '#ff6b6b',    # Merah
        'Netral': '#ffd93d',      # Kuning
        'Positif': '#6bcb77'      # Hijau
    }
    color = color_map.get(sentiment_label, '#4dabf7')
    
    # Plot
    plt.figure(figsize=(12, 7))
    bars = plt.barh(range(len(ngrams)), counts, color=color, edgecolor='black', alpha=0.8)
    plt.yticks(range(len(ngrams)), ngrams, fontsize=11)
    plt.gca().invert_yaxis()
    
    # Styling
    ngram_type = {1: 'Unigram', 2: 'Bigram', 3: 'Trigram'}.get(n, f'{n}-gram')
    plt.title(f'Top {top_k} {ngram_type} - Sentimen {sentiment_label}', 
              fontsize=16, fontweight='bold', pad=15)
    plt.xlabel('Jumlah Kemunculan', fontsize=12)
    plt.ylabel(ngram_type, fontsize=12)
    
    # Tambahkan label jumlah di ujung bar
    for bar, num in zip(bars, counts):
        plt.text(num + max(counts)*0.01, bar.get_y() + bar.get_height()/2,
                 str(int(num)), va='center', fontsize=11, color='black', fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"💾 Gambar disimpan: {save_path}")
    
    plt.show()


def plot_wordcloud(corpus, sentiment_label, save_path=None):
    """
    Membuat word cloud untuk sentimen tertentu.
    """
    if not WORDCLOUD_AVAILABLE:
        print("⚠️ WordCloud tidak tersedia.")
        return
    
    text = ' '.join(corpus)
    
    # Warna berdasarkan sentimen
    color_map = {
        'Negatif': 'Reds',
        'Netral': 'YlOrBr',
        'Positif': 'Greens'
    }
    colormap = color_map.get(sentiment_label, 'Blues')
    
    wordcloud = WordCloud(
        width=1200, 
        height=600,
        background_color='white',
        colormap=colormap,
        stopwords=CUSTOM_STOPWORDS,
        max_words=100,
        max_font_size=150,
        random_state=42
    ).generate(text)
    
    plt.figure(figsize=(14, 7))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'Word Cloud - Sentimen {sentiment_label}', fontsize=18, fontweight='bold', pad=15)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"💾 Word Cloud disimpan: {save_path}")
    
    plt.show()


def plot_cluster_distribution(df, cluster_col='cluster', label_col='sentimen'):
    """
    Visualisasi distribusi cluster dengan PCA.
    """
    # TF-IDF untuk visualisasi
    tfidf = TfidfVectorizer(max_features=1000)
    tfidf_matrix = tfidf.fit_transform(df['content_final'])
    
    # PCA untuk reduksi dimensi ke 2D
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(tfidf_matrix.toarray())
    
    df_plot = df.copy()
    df_plot['x'] = coords[:, 0]
    df_plot['y'] = coords[:, 1]
    
    # Plot
    plt.figure(figsize=(12, 8))
    colors = {'Negatif': '#ff6b6b', 'Netral': '#ffd93d', 'Positif': '#6bcb77'}
    
    for sentimen in ['Negatif', 'Netral', 'Positif']:
        subset = df_plot[df_plot[label_col] == sentimen]
        plt.scatter(subset['x'], subset['y'], 
                   c=colors[sentimen], 
                   label=sentimen, 
                   alpha=0.6, 
                   s=50,
                   edgecolors='black',
                   linewidths=0.5)
    
    plt.xlabel('Principal Component 1', fontsize=12)
    plt.ylabel('Principal Component 2', fontsize=12)
    plt.title('Distribusi Cluster K-Means (PCA 2D)', fontsize=14, fontweight='bold')
    plt.legend(title='Sentimen', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}cluster_pca_visualization.png', dpi=300, bbox_inches='tight')
    print(f"💾 Cluster visualization disimpan: {OUTPUT_DIR}cluster_pca_visualization.png")
    plt.show()


# =============================================================================
# MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("K-MEANS CLUSTERING - VISUALISASI N-GRAM PER SENTIMEN")
    print("=" * 70)
    
    # 1. Load Data
    print("\n📂 1. LOADING DATA")
    print("-" * 50)
    
    try:
        df = pd.read_csv(DATA_PATH)
        print(f"✅ Data loaded: {len(df)} baris")
        print(f"   Columns: {df.columns.tolist()}")
    except FileNotFoundError:
        print(f"❌ File tidak ditemukan: {DATA_PATH}")
        print("   Jalankan analisis_sentimen_sertifikasi_halal.py terlebih dahulu!")
        exit(1)
    
    # Pastikan kolom yang dibutuhkan ada
    required_cols = ['content_final', 'sentimen']
    if not all(col in df.columns for col in required_cols):
        print(f"❌ Kolom yang dibutuhkan tidak ditemukan: {required_cols}")
        exit(1)
    
    # Drop NaN
    df = df.dropna(subset=['content_final', 'sentimen'])
    print(f"   Data setelah drop NaN: {len(df)} baris")
    
    # Distribusi sentimen
    print("\n📊 Distribusi Sentimen:")
    for sentimen in ['Negatif', 'Netral', 'Positif']:
        count = len(df[df['sentimen'] == sentimen])
        pct = (count / len(df)) * 100
        print(f"   {sentimen}: {count} ({pct:.1f}%)")
    
    # =========================================================================
    # 2. VISUALISASI TRIGRAM PER SENTIMEN
    # =========================================================================
    print("\n" + "=" * 70)
    print("📊 2. VISUALISASI TOP 10 TRIGRAM PER SENTIMEN")
    print("=" * 70)
    
    sentiments = ['Negatif', 'Netral', 'Positif']
    
    for sentiment in sentiments:
        print(f"\n{'='*50}")
        print(f"📈 Trigram untuk Sentimen: {sentiment.upper()}")
        print(f"{'='*50}")
        
        # Filter data berdasarkan sentimen
        sentiment_data = df[df['sentimen'] == sentiment]['content_final']
        
        if len(sentiment_data) == 0:
            print(f"⚠️ Tidak ada data untuk sentimen {sentiment}")
            continue
        
        print(f"   Jumlah data: {len(sentiment_data)}")
        
        # Plot trigram
        save_path = f"{OUTPUT_DIR}trigram_{sentiment.lower()}.png"
        plot_ngrams(
            corpus=sentiment_data,
            sentiment_label=sentiment,
            n=3,  # Trigram
            top_k=10,
            stopwords=ALL_STOPWORDS,
            save_path=save_path
        )
    
    # =========================================================================
    # 3. VISUALISASI BIGRAM PER SENTIMEN
    # =========================================================================
    print("\n" + "=" * 70)
    print("📊 3. VISUALISASI TOP 10 BIGRAM PER SENTIMEN")
    print("=" * 70)
    
    for sentiment in sentiments:
        print(f"\n📈 Bigram untuk Sentimen: {sentiment}")
        
        sentiment_data = df[df['sentimen'] == sentiment]['content_final']
        
        if len(sentiment_data) == 0:
            continue
        
        save_path = f"{OUTPUT_DIR}bigram_{sentiment.lower()}.png"
        plot_ngrams(
            corpus=sentiment_data,
            sentiment_label=sentiment,
            n=2,  # Bigram
            top_k=10,
            stopwords=ALL_STOPWORDS,
            save_path=save_path
        )
    
    # =========================================================================
    # 4. WORD CLOUD PER SENTIMEN
    # =========================================================================
    if WORDCLOUD_AVAILABLE:
        print("\n" + "=" * 70)
        print("☁️  4. WORD CLOUD PER SENTIMEN")
        print("=" * 70)
        
        for sentiment in sentiments:
            print(f"\n☁️  Word Cloud: {sentiment}")
            
            sentiment_data = df[df['sentimen'] == sentiment]['content_final']
            
            if len(sentiment_data) == 0:
                continue
            
            save_path = f"{OUTPUT_DIR}wordcloud_{sentiment.lower()}.png"
            plot_wordcloud(
                corpus=sentiment_data,
                sentiment_label=sentiment,
                save_path=save_path
            )
    
    # =========================================================================
    # 5. VISUALISASI CLUSTER (PCA)
    # =========================================================================
    print("\n" + "=" * 70)
    print("🎯 5. VISUALISASI CLUSTER (PCA 2D)")
    print("=" * 70)
    
    plot_cluster_distribution(df)
    
    # =========================================================================
    # 6. RINGKASAN TOP KATA PER SENTIMEN
    # =========================================================================
    print("\n" + "=" * 70)
    print("📋 6. RINGKASAN TOP KATA PER SENTIMEN")
    print("=" * 70)
    
    for sentiment in sentiments:
        print(f"\n🔤 Top 10 Unigram - {sentiment}:")
        sentiment_data = df[df['sentimen'] == sentiment]['content_final']
        
        if len(sentiment_data) == 0:
            continue
            
        top_words = get_top_ngrams(sentiment_data, n=1, top_k=10, stopwords=ALL_STOPWORDS)
        for i, (word, count) in enumerate(top_words, 1):
            print(f"   {i}. {word}: {int(count)}")
    
    # =========================================================================
    # SELESAI
    # =========================================================================
    print("\n" + "=" * 70)
    print("✅ ANALISIS SELESAI!")
    print("=" * 70)
    print(f"\n📁 Output disimpan di: {OUTPUT_DIR}")
    print("   - trigram_negatif.png")
    print("   - trigram_netral.png")
    print("   - trigram_positif.png")
    print("   - bigram_negatif.png")
    print("   - bigram_netral.png")
    print("   - bigram_positif.png")
    if WORDCLOUD_AVAILABLE:
        print("   - wordcloud_negatif.png")
        print("   - wordcloud_netral.png")
        print("   - wordcloud_positif.png")
    print("   - cluster_pca_visualization.png")
