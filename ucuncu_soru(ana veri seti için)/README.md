Bu kodlar, diyabet veri setini yükler, bağımsız değişkenler ile hedef değişkeni ayırır ve veri setini eğitim ve test kümelerine ayırır.

İlk olarak, veri seti yüklenir ve bağımsız değişkenler (`X`) ile hedef değişken (`y`) olarak ayrılır.

Ardından, veri seti eğitim ve test kümelerine ayrılır. Bu kodlarda, veri setinin %70'i eğitim için kullanılırken, %30'u test için kullanılır.

Çoklu Doğrusal Regresyon Analizi için bir model oluşturulur ve eğitilir. Eğitilmiş modelin katsayıları raporlanır.

Aynı şekilde, Multinominal Lojistik Regresyon Analizi için bir model oluşturulur ve eğitilir. Eğitilmiş modelin katsayıları raporlanır.

Eğitilmiş modeller test kümesi üzerinde değerlendirilir. Çoklu Doğrusal Regresyon Analizi için tahminler yapılır ve bu tahminlerin hata oranı hesaplanarak RMSE (Root Mean Squared Error - Kök Ortalama Kare Hata) değeri elde edilir. Multinominal Lojistik Regresyon Analizi için ise tahminler yapılır ve bu tahminlerin doğruluğu hesaplanır.

Ayrıca, sınıflandırma raporu alınır ve bu rapor bir DataFrame'e dönüştürülerek performans metrikleri elde edilir.

Tüm bu sonuçlar birleştirilerek bir DataFrame oluşturulur ve son olarak bu sonuçlar `regression_classification_results.csv` adlı bir CSV dosyasına kaydedilir.

Bu kodlar, iki farklı regresyon modelinin performansını değerlendirir ve sonuçları bir dosyada raporlar.
