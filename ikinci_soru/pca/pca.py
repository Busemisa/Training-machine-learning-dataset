import pandas as pd
from sklearn.decomposition import PCA

# Veri setini yükleyin
data = pd.read_csv('../../diyabetSet_1.csv')

# Veri setindeki bağımsız değişkenleri ve bağımlı değişkeni seçin
X = data.drop(columns=['Outcome'])
y = data['Outcome']

# PCA modelini eğitin ve uygulayın
pca = PCA(n_components=2)
pca.fit(X)
X_pca = pca.transform(X)

# PCA sonuçlarını içeren yeni bir DataFrame oluşturun
pca_df = pd.DataFrame(X_pca, columns=['PCA1', 'PCA2'])

# "Outcome" sütununu PCA sonuçları DataFrame'ine ekleyin
pca_df['Outcome'] = y

# PCA sonuçlarını CSV dosyasına kaydedin
pca_df.to_csv('pca_results_with_outcome.csv', index=False)
