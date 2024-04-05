import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, accuracy_score, confusion_matrix, classification_report, roc_curve, roc_auc_score

# Veri setini yükleyin (örneğin, "veri.csv" dosyası olarak varsayalım)
veri = pd.read_csv("../pca_results_with_outcome.csv")

# Bağımsız değişkenler ve hedef değişken arasında ayırma yapın
X = veri.drop(columns=["Outcome"])
y = veri["Outcome"]

# Veri setini eğitim ve test setlerine ayırın (70% eğitim, 30% test)
X_egitim, X_test, y_egitim, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Çoklu doğrusal regresyon modelini uygulayın
lin_reg_model = LinearRegression()
lin_reg_model.fit(X_egitim, y_egitim)

# Katsayıları raporlayın
print("Çoklu Doğrusal Regresyon Katsayıları:")
for i, coef in enumerate(lin_reg_model.coef_):
    print(f"Beta_{i+1}: {coef}")

# Multinominal lojistik regresyon modelini uygulayın
log_reg_model = LogisticRegression(multi_class='multinomial', solver='lbfgs')
log_reg_model.fit(X_egitim, y_egitim)

# Katsayıları raporlayın
print("\nMultinominal Lojistik Regresyon Katsayıları:")
for i, coef in enumerate(log_reg_model.coef_):
    print(f"Katsayılar Sınıf {i}: {coef}")

# Test seti için tahminler yapın
y_tahmin_lin_reg = lin_reg_model.predict(X_test)
y_tahmin_log_reg = log_reg_model.predict(X_test)

# Çoklu doğrusal regresyon için performans metriklerini hesaplayın (ör. RMSE)
lin_reg_rmse = np.sqrt(mean_squared_error(y_test, y_tahmin_lin_reg))

# Multinominal lojistik regresyon için doğruluk skorunu hesaplayın
log_reg_accuracy = accuracy_score(y_test, y_tahmin_log_reg)

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_tahmin_log_reg)
print("\nConfusion Matrix:")
print(conf_matrix)

# Sınıflandırma raporu
class_report = classification_report(y_test, y_tahmin_log_reg)
print("\nSınıflandırma Raporu:")
print(class_report)

# ROC eğrisi ve AUC
y_probs = log_reg_model.predict_proba(X_test)
fpr, tpr, thresholds = roc_curve(y_test, y_probs[:, 1])
auc = roc_auc_score(y_test, y_probs[:, 1])

print("\nROC AUC Değeri:", auc)

# Performans metriklerini bir DataFrame'e ekleyin
performans_df = pd.DataFrame({
    "Model": ["Çoklu Doğrusal Regresyon", "Multinominal Lojistik Regresyon"],
    "Performans Metriği": [lin_reg_rmse, log_reg_accuracy]
})

# Performans metriklerini CSV dosyasına kaydedin
performans_df.to_csv("performans_metrikleri.csv", index=False)
