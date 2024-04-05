Bu kodlar, PCA sonuçları üzerinde Naive Bayes sınıflandırıcısını uygular ve performansını değerlendirir.

Öncelikle, varsayılan olarak "pca_results_with_outcome.csv" dosyasından bir veri seti yüklenir ve bağımsız değişkenler ile hedef değişken ayrılır.

Veri seti eğitim ve test setlerine ayrılır, ardından Naive Bayes sınıflandırıcısı oluşturulur ve eğitilir.

Modelin performansı test seti üzerinde yapılan tahminlerle değerlendirilir. Doğruluk skoru, hassasiyet, özgüllük, AUC (Area Under Curve) gibi performans metrikleri hesaplanır.

Ayrıca, sınıflandırma raporu ve karışıklık matrisi de hesaplanır. Karışıklık matrisinden TP (True Positive), TN (True Negative), FP (False Positive), FN (False Negative) değerleri elde edilir.

ROC eğrisi ve AUC hesaplanır, sonuçlar bir DataFrame'e kaydedilir ve CSV dosyalarına yazdırılır.

Sonuçlar, "naive_bayes_sonuclar.csv", "naive_bayes_performance_metrics.csv" ve "naive_bayes_confusion_matrix.csv" adlı dosyalara kaydedilir.
