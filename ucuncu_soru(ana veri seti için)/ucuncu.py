import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report
import numpy as np

# Veri setini yükle
diabetes_data = pd.read_csv("../diyabetSet_1.csv")

# Bağımsız değişkenler (X) ve bağımlı değişken (y) olarak ayır
X = diabetes_data.drop("Outcome", axis=1)
y = diabetes_data["Outcome"]

# Veri setini eğitim ve test olarak ayır
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Çoklu Doğrusal Regresyon Analizi
linear_reg = LinearRegression()
linear_reg.fit(X_train, y_train)

# Katsayıları raporla
linear_coefs = pd.DataFrame({'Feature': X.columns, 'Coefficient': linear_reg.coef_})

# Multinominal Lojistik Regresyon Analizi
logistic_reg = LogisticRegression(max_iter=1000)
logistic_reg.fit(X_train, y_train)

# Katsayıları raporla
logistic_coefs = pd.DataFrame({'Feature': X.columns, 'Coefficient': logistic_reg.coef_[0]})

# Test kümesi için performans metriklerini hesapla
linear_pred = linear_reg.predict(X_test)
logistic_pred = logistic_reg.predict(X_test)

linear_rmse = np.sqrt(mean_squared_error(y_test, linear_pred))
logistic_accuracy = accuracy_score(y_test, logistic_pred)

# Sınıflandırma raporunu al
classification_rep = classification_report(y_test, logistic_pred, output_dict=True)
classification_df = pd.DataFrame(classification_rep).transpose()

# Sonuçları bir DataFrame olarak topla
results = pd.DataFrame({'Model': ['Linear Regression', 'Logistic Regression'],
                        'Performance': [linear_rmse, logistic_accuracy]})

# Tüm sonuçları birleştir
all_results = pd.concat([linear_coefs, logistic_coefs, results, classification_df], axis=1)

# Sonuçları CSV dosyasına kaydet
all_results.to_csv("regression_classification_results.csv", index=False)
