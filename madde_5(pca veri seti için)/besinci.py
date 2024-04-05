import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc

# Veri setini yükleyin (örneğin, "veri.csv" dosyası olarak varsayalım)
veri = pd.read_csv("pca_results_with_outcome.csv")

# Bağımsız değişkenler ve hedef değişken arasında ayırma yapın
X = veri.drop(columns=["Outcome"])
y = veri["Outcome"]

# Veri setini eğitim ve test setlerine ayırın (70% eğitim, 30% test)
X_egitim, X_test, y_egitim, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Naive Bayes sınıflandırıcısını oluşturun ve eğitin
naive_bayes_model = GaussianNB()
naive_bayes_model.fit(X_egitim, y_egitim)

# Test verisiyle modeli değerlendirin
y_tahmin = naive_bayes_model.predict(X_test)

# Doğruluk skorunu hesaplayın
accuracy = accuracy_score(y_test, y_tahmin)

# Sınıflandırma raporunu ve karışıklık matrisini hesaplayın
classification_rep = classification_report(y_test, y_tahmin, output_dict=True)
conf_matrix = confusion_matrix(y_test, y_tahmin)

# Confusion Matrix'ten değerleri çıkar
TN, FP, FN, TP = conf_matrix.ravel()

# Hassasiyet (Sensitivity) ve Özgüllük (Specificity) hesaplayın
sensitivity = TP / (TP + FN)
specificity = TN / (TN + FP)

# TP, TN, FP, FN değerlerini DataFrame'e ekleyin
results_df = pd.DataFrame({'Actual Positive': [TP + FN], 'Actual Negative': [FP + TN],
                           'Predicted Positive': [TP + FP], 'Predicted Negative': [FN + TN]})

# ROC eğrisini ve AUC'yi hesaplayın
fpr, tpr, thresholds = roc_curve(y_test, y_tahmin)
auc_score = auc(fpr, tpr)

# Sonuçları bir DataFrame'e kaydedin
sonuclar = pd.DataFrame({"Gerçek Değerler": y_test, "Tahmin Edilen Değerler": y_tahmin})

# Sonuçları CSV dosyasına kaydedin
sonuclar.to_csv("naive_bayes_sonuclar.csv", index=False)

# Performans metriklerini bir DataFrame'e ekleyin
performance_df = pd.DataFrame({"Accuracy": [accuracy], "Sensitivity": [sensitivity],
                                "Specificity": [specificity], "AUC": [auc_score]})

# Sonuçları CSV dosyasına kaydedin
performance_df.to_csv("naive_bayes_performance_metrics.csv", index=False)

# Confusion Matrix sonuçlarını CSV dosyasına kaydedin
results_df.to_csv("naive_bayes_confusion_matrix.csv", index=False)
