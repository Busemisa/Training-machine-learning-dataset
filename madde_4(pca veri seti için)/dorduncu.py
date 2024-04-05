import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Veri setini yükleyin (örneğin, "pca_results_with_outcome.csv" dosyası olarak varsayalım)
veri = pd.read_csv("pca_results_with_outcome.csv")

# Bağımsız değişkenler ve hedef değişken arasında ayırma yapın
X = veri.drop(columns=["Outcome"])
y = veri["Outcome"]

# Veri setini eğitim ve test setlerine ayırın (70% eğitim, 30% test)
X_egitim, X_test, y_egitim, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Karar ağacı sınıflandırma modelini oluşturun ve eğitin
decision_tree_model = DecisionTreeClassifier(criterion="gini", random_state=42)
decision_tree_model.fit(X_egitim, y_egitim)

# Test verisiyle modeli değerlendirin
y_tahmin = decision_tree_model.predict(X_test)

# Doğruluk skorunu hesaplayın
accuracy = accuracy_score(y_test, y_tahmin)
print("Doğruluk Skoru:", accuracy)

# Sınıflandırma raporunu yazdırın
classification_rep = classification_report(y_test, y_tahmin, output_dict=True)
classification_df = pd.DataFrame(classification_rep).transpose()

# Karışıklık matrisini yazdırın
conf_matrix = confusion_matrix(y_test, y_tahmin)
conf_matrix_df = pd.DataFrame(conf_matrix, index=["Gerçek 0", "Gerçek 1"], columns=["Tahmin 0", "Tahmin 1"])

# Görselleştirme: Karar Ağacı Yapısı
plt.figure(figsize=(15, 10))
plot_tree(decision_tree_model, filled=True, feature_names=X.columns, class_names=["0", "1"])
plt.savefig("decision_tree_structure_pca.png")

# Performans metriklerini bir DataFrame'e ekleyin
performans_df = pd.DataFrame({"Doğruluk Skoru": accuracy}, index=[0])

# Sonuçları CSV dosyasına kaydedin
classification_df.to_csv("classification_report_decision_tree_pca.csv")
conf_matrix_df.to_csv("confusion_matrix_decision_tree_pca.csv")
performans_df.to_csv("model_performance_decision_tree_pca.csv", index=False)
