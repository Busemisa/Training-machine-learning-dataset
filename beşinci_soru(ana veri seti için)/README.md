Bu kodlar, Naive Bayes sınıflandırıcısını kullanarak diyabet veri seti üzerinde eğitim ve test verisi üzerinde tahminler yapmayı içerir.

İlk olarak, diyabet veri seti yüklenir ve bağımsız değişkenler (`X`) ile hedef değişken (`y`) ayrılır. Veri seti daha sonra eğitim ve test kümelerine ayrılır.

Naive Bayes sınıflandırıcısı oluşturulur ve eğitilir. Eğitilmiş model, hem eğitim hem de test verisi üzerinde tahminler yapar.

Eğitim ve test verisi için doğruluk (accuracy), sınıflandırma raporu ve karışıklık matrisi gibi çeşitli performans metrikleri hesaplanır.

Son olarak, elde edilen performans metrikleri bir DataFrame'e dönüştürülür ve `naive_bayes_results.csv` adlı bir CSV dosyasına kaydedilir.

Bu kodlar, Naive Bayes sınıflandırıcısının eğitim ve test performansını değerlendirir ve sonuçları bir dosyada raporlar.
