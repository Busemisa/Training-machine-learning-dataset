import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Veri setini yükle
diabetes_data = pd.read_csv("../diyabetSet_1.csv")

# Bağımsız değişkenler (X) ve bağımlı değişken (y) olarak ayır
X = diabetes_data.drop("Outcome", axis=1)
y = diabetes_data["Outcome"]

# Veri setini eğitim ve test olarak ayır
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Naive Bayes sınıflandırıcısını oluştur ve eğit
naive_bayes = GaussianNB()
naive_bayes.fit(X_train, y_train)

# Eğitim verisi üzerinde tahmin yap
train_predictions = naive_bayes.predict(X_train)

# Eğitim verisi için performans metriklerini hesapla
train_accuracy = accuracy_score(y_train, train_predictions)
train_classification_report = classification_report(y_train, train_predictions)
train_conf_matrix = confusion_matrix(y_train, train_predictions)

# Test verisi üzerinde tahmin yap
test_predictions = naive_bayes.predict(X_test)

# Test verisi için performans metriklerini hesapla
test_accuracy = accuracy_score(y_test, test_predictions)
test_classification_report = classification_report(y_test, test_predictions)
test_conf_matrix = confusion_matrix(y_test, test_predictions)

# Confusion matrixi terminalde göster
print("Train Confusion Matrix:")
print(train_conf_matrix)
print("\nTest Confusion Matrix:")
print(test_conf_matrix)

# Sonuçları bir DataFrame'e dönüştür
results_df = pd.DataFrame({
    "Dataset": ["Train", "Test"],
    "Accuracy": [train_accuracy, test_accuracy],
    "Classification Report": [train_classification_report, test_classification_report]
})

# Sonuçları CSV dosyasına kaydet
results_df.to_csv("naive_bayes_results.csv", index=False)
