Bu kodlar, diyabet veri setinin yüklenmesini, min-max normalizasyonu işleminin gerçekleştirilmesini ve sonrasında normalleştirilmiş verinin başka bir CSV dosyası olarak kaydedilmesini içerir.

İlk olarak, kodlar Pandas kütüphanesini kullanarak `diyabetSet_1.csv` adlı bir veri setini yükler.

Daha sonra, min-max normalizasyonunu gerçekleştirmek için bir fonksiyon tanımlanır. Bu fonksiyon, bir sütunun en küçük ve en büyük değerlerini kullanarak normalizasyon işlemini yapar.

'Outcome' adlı hedef değişken sütunu ayrı bir değişkende saklanır ve veri setinden çıkarılır.

Tüm sütunlar üzerinde min-max normalizasyonu uygulanır ve sonuç normalize edilmiş veri olarak `normalized_data` değişkenine atanır.

'Outcome' sütunu normalize edilmiş veri setine geri eklenir.

Son olarak, normalize edilmiş veri, `normalized_diabetes_data.csv` adlı yeni bir CSV dosyası olarak kaydedilir.

Bu kodlar, veri setinin özelliklerini normalize etmek için kullanılır ve daha sonra makine öğrenimi modellerine uygulanabilir hale getirir.
