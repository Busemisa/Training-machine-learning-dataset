Bu kodlar, bir veri seti üzerinde çoklu doğrusal regresyon ve multinominal lojistik regresyon modellerini uygular ve bu modellerin performansını değerlendirir.

İlk olarak, varsayılan olarak "pca_results_with_outcome.csv" dosyasından bir veri seti yüklenir. Bağımsız değişkenler (`X`) ve hedef değişken (`y`) olarak ayrılır, daha sonra veri seti eğitim ve test setlerine ayrılır.

Çoklu doğrusal regresyon modeli ve multinominal lojistik regresyon modeli uygulanır ve eğitilir. Bu modellerin katsayıları raporlanır.

Daha sonra, test seti üzerinde tahminler yapılır ve her iki model için performans metrikleri hesaplanır. Çoklu doğrusal regresyon için Root Mean Squared Error (RMSE) hesaplanırken, multinominal lojistik regresyon için doğruluk skoru hesaplanır.

Ayrıca, multinominal lojistik regresyon modeli için confusion matrix (karışıklık matrisi) ve sınıflandırma raporu da hesaplanır ve terminalde gösterilir.

Son olarak, ROC eğrisi ve Area Under Curve (AUC) hesaplanır ve bu metrikler bir DataFrame'e eklenerek `performans_metrikleri.csv` adlı bir CSV dosyasına kaydedilir.

Bu kodlar, iki farklı regresyon modelinin performansını değerlendirir ve sonuçları bir dosyada raporlar.
