Bu kodlar, bir karar ağacı sınıflandırma modelini PCA sonuçları üzerinde eğitmeyi ve test etmeyi sağlar.

Veri seti olarak varsayılan olarak "pca_results_with_outcome.csv" dosyası yüklenir ve bağımsız değişkenler ile hedef değişken ayrıştırılır.

Veri seti eğitim ve test setlerine ayrılır, ardından karar ağacı sınıflandırma modeli oluşturulur ve eğitilir.

Modelin performansı, test seti üzerinde yapılan tahminlerle değerlendirilir. Doğruluk skoru hesaplanır ve yazdırılır. Ayrıca sınıflandırma raporu ve karışıklık matrisi oluşturulur ve bu sonuçlar CSV dosyalarına kaydedilir.

Karar ağacının yapısını görselleştirmek için `plot_tree` fonksiyonu kullanılır ve "decision_tree_structure_pca.png" adlı bir dosyaya kaydedilir.

Son olarak, performans metrikleri bir DataFrame'e eklenir ve CSV dosyalarına kaydedilir.

Bu kodlar, bir karar ağacı modelinin PCA sonuçları üzerindeki performansını değerlendirir ve sonuçları dosyalara kaydeder.
