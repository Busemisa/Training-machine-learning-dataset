Bu kodlar, bir karar ağacı sınıflandırma modeli oluşturmayı, eğitmeyi ve test verisi üzerinde tahmin yapmayı içerir. Ayrıca, modelin performansını değerlendirmek için çeşitli performans metriklerini hesaplar ve sonuçları bir dosyada raporlar.

İlk olarak, diyabet veri seti yüklenir ve bağımsız değişkenler (`X`) ile hedef değişken (`y`) ayrılır. Veri seti daha sonra eğitim ve test kümelerine ayrılır.

Karar ağacı sınıflandırma modeli oluşturulur ve eğitilir. Ardından, bu model kullanılarak test verisi üzerinde tahminler yapılır.

Modelin performansını değerlendirmek için doğruluk (accuracy), sınıflandırma raporu ve karışıklık matrisi gibi çeşitli metrikler hesaplanır.

Sonuçlar bir DataFrame'e dönüştürülerek birleştirilir ve `decision_tree_results.csv` adlı bir CSV dosyasına kaydedilir. Ayrıca, karışıklık matrisi de `confusion_matrix.csv` adlı bir CSV dosyasına kaydedilir.

Ağaç yapısını görselleştirmek için `plot_tree` fonksiyonu kullanılır. Bu şekilde, karar ağacının yapısı görsel olarak temsil edilir ve "decision_tree_structure.png" adlı bir dosyaya kaydedilir.

Bu kodlar, bir karar ağacı sınıflandırma modelinin eğitimini, testini ve performans değerlendirmesini sağlar, ardından sonuçları dosyalara kaydeder ve ağaç yapısını görselleştirir.
