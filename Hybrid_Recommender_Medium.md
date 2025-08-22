# E-Ticaret için Hibrit Ürün Öneri Sistemi: İçerik Tabanlı + İşbirlikçi Filtreleme

Bu projede, e-ticaret ortamları için **hibrit ürün öneri sistemi** geliştirdim.  
Sistem hem **içerik tabanlı** hem de **işbirlikçi filtreleme (collaborative filtering)** tekniklerini birleştiriyor.

Dataset olarak **MovieLens 100k** kullanıldı; kullanıcı-film etkileşimleri, film türleri ve etiketler üzerinden öneri sistemi tasarlandı.

---

## 🔹 Gün 1 – Veri Hazırlığı ve Basit Öneri Sistemi

### 1. Veri Seti ve Hazırlık

- `movies.csv`, `ratings.csv`, `links.csv`, `tags.csv` dosyaları okundu.
- Kullanıcı-film etkileşimleri ve film türleri incelendi.
- Basit istatistikler: popüler filmler, kullanıcı başına izlenen film sayısı.

### 2. Baseline Öneri Modelleri

- **Popülerlik tabanlı öneri:** En çok izlenen veya en yüksek puan alan filmler.
- **İçerik tabanlı öneri:** Film türlerine göre cosine similarity ile benzer filmler önerildi.

```python
# Örnek içerik tabanlı öneri
print("Önerilen Filmler:", content_based_recommender("Toy Story (1995)"))
```

### 3. Başarı Metrikleri

- Precision@K ve Recall@K hesaplanarak baseline modellerin performansı değerlendirildi.

---

## 🔹 Gün 2 – İleri Düzey Modeller

### 1. İşbirlikçi Filtreleme (Collaborative Filtering)

- Kullanıcı-ürün matrisi oluşturuldu.
- Matris faktorizasyonu (SVD) ile bilinmeyen kullanıcı-film etkileşimleri tahmin edildi.

```python
# Kullanıcı-ürün matrisi
user_item_matrix = movie_ratings.pivot_table(index='userId', columns='title', values='rating').fillna(0)
```

### 2. Hibrit Sistem

- İçerik tabanlı ve işbirlikçi tahminler birleştirildi.
- Tek bir fonksiyon ile hem içerik hem de kullanıcı etkileşimleri dikkate alınarak öneriler üretildi.

```python
def hybrid_recommender(user_id, title, top_n=5, alpha=0.5):
    content_recs = content_based_recommender(title, top_n=top_n*2)
    collab_recs = collaborative_recommender(preds_df, user_id, movies, ratings, top_n=top_n*2)
    combined = list(dict.fromkeys(content_recs + collab_recs))[:top_n]
    return combined
```

### 3. Model Değerlendirmesi

- Cross-validation veya kullanıcı bazlı örnekler ile öneriler gözlemlendi.
- Hibrit yaklaşımın doğruluğu baseline modellerden daha yüksek bulundu.

---

## 🔹 Gün 3 – Ürünleştirme ve Yayınlama

### 1. Pipeline Kurulumu

- Veri yükleme, modelleme ve öneri üretimi tek fonksiyonlar halinde toplandı.
- Kullanıcı başına öneri üretmek artık **tek satırda** mümkün.

### 2. Başarı Metrikleri ve Görselleştirme

- Precision@K, Recall@K ve NDCG hesaplandı.
- Tüm kullanıcılar için ortalamalar alındı.
- Örnek bir kullanıcı için önerilen ve gerçek izlenen filmler karşılaştırıldı.

```python
print("Precision@10:", sum(precisions)/len(precisions))
print("Recall@10:", sum(recalls)/len(recalls))
print("NDCG@10:", sum(ndcgs)/len(ndcgs))
```

### 3. Sonuçlar

- Hibrit sistem, hem içerik hem de işbirlikçi filtrelemeyi kullandığı için **daha isabetli öneriler** sunuyor.
- Kullanıcıların sevdiği içeriklerin çoğunu yakalayabiliyor.
- Precision ve Recall değerleri baseline sistemlerden daha yüksek.
