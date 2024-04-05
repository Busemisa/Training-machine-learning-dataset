import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Veri setini yükleyin
data = pd.read_csv('../../diyabetSet_1.csv')

# Veri setindeki bağımsız değişkenleri ve hedef değişkeni ayırın
X = data.drop(columns=['Outcome'])
y = data['Outcome']

# PCA modelini eğitin
pca = PCA(n_components=2)
pca.fit(X)
X_pca = pca.transform(X)

# LDA modelini eğitin
lda = LinearDiscriminantAnalysis(n_components=1)
lda.fit(X, y)
X_lda = lda.transform(X)

# PCA ve LDA'ya göre dönüştürülmüş veriyi yeni DataFrame'lere aktarın
pca_df = pd.DataFrame(X_pca, columns=['PCA1', 'PCA2'])
lda_df = pd.DataFrame(X_lda, columns=['LDA'])

# PCA ve LDA sonuçlarını birleştirin
result_df = pd.concat([pca_df, lda_df], axis=1)

# Sonuçları CSV dosyasına kaydedin
result_df.to_csv('pca_lda_results.csv', index=False)
