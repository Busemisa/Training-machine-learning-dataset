from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Veri setini yükle
diabetes_data = pd.read_csv("../diyabetSet_1.csv")

# Bağımsız değişkenler (X) ve bağımlı değişken (y) olarak ayır
X = diabetes_data.drop("Outcome", axis=1)
y = diabetes_data["Outcome"]

# Veri setini eğitim ve test olarak ayır
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Karar ağacı sınıflandırma modelini oluştur ve eğit
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, y_train)

# Test verisi için tahmin yap
y_pred = decision_tree.predict(X_test)

# Performans metriklerini hesapla
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred, output_dict=True)

# Sınıflandırma raporunu DataFrame'e dönüştür
classification_df = pd.DataFrame(classification_rep).transpose()

# Performans metriklerini DataFrame'e ekle
performance_df = pd.DataFrame({"Accuracy": [accuracy]})

# Karışıklık matrisini hesapla ve DataFrame'e dönüştür
confusion = confusion_matrix(y_test, y_pred)
confusion_df = pd.DataFrame(confusion, index=["Actual 0", "Actual 1"], columns=["Predicted 0", "Predicted 1"])

# Sonuçları birleştir
result_df = pd.concat([performance_df, classification_df], axis=1)

# Sonuçları ve karışıklık matrisini CSV dosyasına kaydet
result_df.to_csv("decision_tree_results.csv", index=True)
confusion_df.to_csv("confusion_matrix.csv")

# Ağaç yapısını görselleştirme
plt.figure(figsize=(20,10))
plot_tree(decision_tree, filled=True, feature_names=X.columns, class_names=['0', '1'], fontsize=3)
plt.savefig("decision_tree_structure.png", bbox_inches='tight')
plt.show()
