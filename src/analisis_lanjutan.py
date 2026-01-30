"""
Analisis Lanjutan Data Sertifikasi Halal
=========================================
1. Word Cloud per Sentimen
2. Naive Bayes Classification
3. Confusion Matrix & Classification Report
4. Top Words per Sentimen
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Fix for numpy 2.0 compatibility
np.asarray_orig = np.asarray
def asarray_fix(*args, **kwargs):
    kwargs.pop('copy', None)
    return np.asarray_orig(*args, **kwargs)
np.asarray = asarray_fix

# ML Libraries
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Word Cloud
from wordcloud import WordCloud
from collections import Counter

# =============================================================================
# 1. LOAD DATA HASIL CLUSTERING
# =============================================================================
print("=" * 60)
print("ANALISIS LANJUTAN DATA SERTIFIKASI HALAL")
print("=" * 60)

df = pd.read_csv('../output/hasil_analisis_sentimen_halal.csv')
print(f"Total data: {len(df)}")
print(f"\nDistribusi Sentimen:")
print(df['sentimen'].value_counts())

# =============================================================================
# 2. WORD CLOUD PER SENTIMEN
# =============================================================================
print("\n" + "=" * 60)
print("1. MEMBUAT WORD CLOUD PER SENTIMEN")
print("=" * 60)

def generate_wordcloud(text, title, color, filename):
    """Generate word cloud untuk teks tertentu"""
    wordcloud = WordCloud(
        width=800, 
        height=400,
        background_color='white',
        colormap=color,
        max_words=100,
        min_font_size=10,
        max_font_size=100,
        random_state=42
    ).generate(text)
    
    plt.figure(figsize=(12, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(f'../output/{filename}', dpi=300, bbox_inches='tight')
    plt.show()
    print(f"✅ Word cloud disimpan: ../output/{filename}")

# Word Cloud untuk semua data
all_text = ' '.join(df['content_final'].dropna().astype(str))
generate_wordcloud(all_text, 'Word Cloud - Semua Data Sertifikasi Halal', 'viridis', 'wordcloud_all.png')

# Word Cloud per sentimen
sentimen_colors = {
    'Negatif': 'Reds',
    'Netral': 'Greys',
    'Positif': 'Greens'
}

for sentimen, color in sentimen_colors.items():
    text = ' '.join(df[df['sentimen'] == sentimen]['content_final'].dropna().astype(str))
    if text.strip():
        generate_wordcloud(
            text, 
            f'Word Cloud - Sentimen {sentimen}', 
            color, 
            f'wordcloud_{sentimen.lower()}.png'
        )

# =============================================================================
# 3. TOP WORDS PER SENTIMEN
# =============================================================================
print("\n" + "=" * 60)
print("2. TOP WORDS PER SENTIMEN")
print("=" * 60)

def get_top_words(df, sentimen, n=20):
    """Get top n words untuk sentimen tertentu"""
    text = ' '.join(df[df['sentimen'] == sentimen]['content_final'].dropna().astype(str))
    words = text.split()
    word_counts = Counter(words)
    return word_counts.most_common(n)

for sentimen in ['Positif', 'Netral', 'Negatif']:
    print(f"\n📊 Top 15 Kata - Sentimen {sentimen}:")
    print("-" * 40)
    top_words = get_top_words(df, sentimen, 15)
    for i, (word, count) in enumerate(top_words, 1):
        print(f"  {i:2}. {word:20} : {count}")

# =============================================================================
# 4. NAIVE BAYES CLASSIFICATION
# =============================================================================
print("\n" + "=" * 60)
print("3. NAIVE BAYES CLASSIFICATION")
print("=" * 60)

# Prepare data
X = df['content_final'].fillna('')
y = df['sentimen']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training data: {len(X_train)}")
print(f"Testing data: {len(X_test)}")

# TF-IDF Vectorization
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Train Naive Bayes
print("\nMelatih model Naive Bayes...")
nb_classifier = MultinomialNB(alpha=0.1)
nb_classifier.fit(X_train_tfidf, y_train)

# Predict
y_pred = nb_classifier.predict(X_test_tfidf)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
print(f"\n✅ Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

# Classification Report
print("\n📋 Classification Report:")
print("-" * 60)
print(classification_report(y_test, y_pred))

# =============================================================================
# 5. CONFUSION MATRIX
# =============================================================================
print("\n" + "=" * 60)
print("4. CONFUSION MATRIX")
print("=" * 60)

# Create confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=['Positif', 'Netral', 'Negatif'])

# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Positif', 'Netral', 'Negatif'],
            yticklabels=['Positif', 'Netral', 'Negatif'],
            annot_kws={'size': 14})
plt.xlabel('Predicted', fontsize=12)
plt.ylabel('Actual', fontsize=12)
plt.title('Confusion Matrix - Naive Bayes Classification\nData Sertifikasi Halal', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('../output/confusion_matrix_halal.png', dpi=300, bbox_inches='tight')
plt.show()
print("✅ Confusion matrix disimpan: ../output/confusion_matrix_halal.png")

# =============================================================================
# 6. VISUALISASI PERBANDINGAN
# =============================================================================
print("\n" + "=" * 60)
print("5. VISUALISASI PERBANDINGAN")
print("=" * 60)

# Bar chart perbandingan actual vs predicted
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Actual distribution
actual_counts = y_test.value_counts()
colors = ['#ff6b6b', '#ffd93d', '#6bcb77']
axes[0].bar(actual_counts.index, actual_counts.values, color=colors, edgecolor='black')
axes[0].set_title('Distribusi Aktual (Test Set)', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Sentimen', fontsize=12)
axes[0].set_ylabel('Jumlah', fontsize=12)
for i, (idx, v) in enumerate(zip(actual_counts.index, actual_counts.values)):
    axes[0].text(i, v + 5, str(v), ha='center', fontsize=11, fontweight='bold')

# Predicted distribution
pred_counts = pd.Series(y_pred).value_counts()
axes[1].bar(pred_counts.index, pred_counts.values, color=colors, edgecolor='black')
axes[1].set_title('Distribusi Prediksi (Naive Bayes)', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Sentimen', fontsize=12)
axes[1].set_ylabel('Jumlah', fontsize=12)
for i, (idx, v) in enumerate(zip(pred_counts.index, pred_counts.values)):
    axes[1].text(i, v + 5, str(v), ha='center', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('../output/perbandingan_actual_vs_predicted.png', dpi=300, bbox_inches='tight')
plt.show()
print("✅ Perbandingan disimpan: ../output/perbandingan_actual_vs_predicted.png")

# =============================================================================
# 7. SIMPAN MODEL DAN HASIL
# =============================================================================
print("\n" + "=" * 60)
print("6. MENYIMPAN HASIL")
print("=" * 60)

import joblib

# Simpan model dan vectorizer
joblib.dump(nb_classifier, '../output/models/naive_bayes_model.pkl')
joblib.dump(tfidf, '../output/models/tfidf_vectorizer.pkl')
print("✅ Model disimpan: ../output/models/naive_bayes_model.pkl")
print("✅ TF-IDF Vectorizer disimpan: ../output/models/tfidf_vectorizer.pkl")

# Simpan hasil prediksi
df_test = pd.DataFrame({
    'content_final': X_test,
    'actual': y_test,
    'predicted': y_pred
})
df_test.to_csv('../output/hasil_prediksi_naive_bayes.csv', index=False, encoding='utf-8-sig')
print("✅ Hasil prediksi disimpan: ../output/hasil_prediksi_naive_bayes.csv")

# =============================================================================
# RINGKASAN
# =============================================================================
print("\n" + "=" * 60)
print("📊 RINGKASAN HASIL ANALISIS LANJUTAN")
print("=" * 60)
print(f"""
1. Word Cloud:
   - wordcloud_all.png (semua data)
   - wordcloud_positif.png
   - wordcloud_netral.png
   - wordcloud_negatif.png

2. Naive Bayes Classification:
   - Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)
   - Model tersimpan di: ../output/models/

3. Confusion Matrix:
   - confusion_matrix_halal.png

4. Perbandingan Actual vs Predicted:
   - perbandingan_actual_vs_predicted.png
""")

print("✅ SEMUA ANALISIS SELESAI!")
