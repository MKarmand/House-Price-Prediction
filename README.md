# Prediksi Harga Rumah

[Dataset](https://www.kaggle.com/datasets/wisnuanggara/daftar-harga-rumah?select=DATA+RUMAH.xlsx).
[Streamlit](https://house-price-prediction-mka.streamlit.app/).


## Domain Proyek

Rumah merupakan kebutuhan utama manusia sebagai tempat tinggal, berkumpul, dan beristirahat, serta memiliki nilai investasi karena harga properti cenderung meningkat, terutama di lokasi strategis. Selain sebagai tempat tinggal, rumah juga berfungsi sebagai tempat singgah bagi wisatawan. Seiring meningkatnya kebutuhan, banyak pengusaha properti membangun rumah untuk memenuhi permintaan pasar, sehingga masyarakat harus lebih selektif dalam memilih rumah yang sesuai kebutuhan dan bernilai investasi.

**Rubrik/Kriteria Tambahan (Opsional)**:
Masalah ini perlu diselesaikan karena banyaknya pilihan rumah membuat calon pembeli sulit menentukan pilihan terbaik. Dengan pengolahan data yang sistematis berdasarkan faktor harga, luas, dan fasilitas, calon pembeli dapat mengambil keputusan yang lebih tepat, efisien, dan menguntungkan.
  
  Format Referensi: [Haryanto, C., Rahaningsih, N., & Muhammad Basysyar, F. (2023). KOMPARASI ALGORITMA MACHINE LEARNING DALAM MEMPREDIKSI HARGA RUMAH. JATI (Jurnal Mahasiswa Teknik Informatika), 7(1), 533–539. https://doi.org/10.36040/jati.v7i1.6343](https://doi.org/10.36040/jati.v7i1.6343) 

## Business Understanding

### Problem Statements

Menjelaskan pernyataan masalah latar belakang:
- Bagaimana memprediksi harga rumah secara akurat berdasarkan fitur-fitur properti seperti luas bangunan, luas tanah, jumlah kamar tidur, kamar mandi, dan fasilitas lainnya?
- Bagaimana membangun model prediksi harga rumah yang efektif untuk membantu calon pembeli dan investor mengambil keputusan berdasarkan estimasi harga properti?

### Goals

Menjelaskan tujuan dari pernyataan masalah:
- Mengembangkan model prediksi harga rumah dengan memanfaatkan data fitur properti yang tersedia sehingga dapat membantu pengguna memperkirakan nilai pasar properti yang diminati.
- Meningkatkan akurasi prediksi harga rumah dengan memilih metode predictive analytics yang tepat serta menerapkan teknik evaluasi dan validasi model.

**Rubrik/Kriteria Tambahan (Opsional)**:


    ### Solution statements
    - Menerapkan algoritma machine learning untuk regresi seperti Linear Regression, Random Forest Regressoruntuk membangun model prediksi harga rumah.
    - Membandingkan performa beberapa model prediktif berdasarkan metrik evaluasi seperti Mean Absolute Error (MAE), Mean Squared Error (MSE), dan R-squared (R²) untuk memilih model terbaik.

## Data Understanding
Pada proyek ini, kami menggunakan dataset yang berisi informasi terkait harga rumah yang dijual. Dataset ini mencakup berbagai fitur properti, seperti harga rumah, luas bangunan (LB), luas tanah (LT), jumlah kamar tidur (KT), jumlah kamar mandi (KM), jumlah kapasitas mobil dalam garasi (GRS). Dataset ini terdiri dari 1010 entri dengan 8 kolom yang mewakili karakteristik utama yang mempengaruhi harga rumah. Data ini akan digunakan untuk menganalisis pola-pola harga properti berdasarkan faktor-faktor yang ada, serta untuk membangun model prediktif yang dapat memperkirakan harga rumah di masa depan.




### Variabel-variabel pada Kaggle dataset adalah sebagai berikut:
- NO         : nomor data.
- NAMA RUMAH : title rumah.
- HARGA      : harga dari rumah.
- LB         : jumlah luas bangunan.
- LT         : jumlah luas tanah.
- KT         : jumlah kamar tidur
- KM         : jumlah kamar mandi.
- GRS        : jumlah kapasitas mobil dalam garasi.


**Rubrik/Kriteria Tambahan (Opsional)**:
# Informasi Struktur Data
Setelah data berhasil dimuat, kita menggunakan `df.info()` untuk memahami struktur dataset. Output ini mengonfirmasi bahwa kita memiliki 1010 entri (baris) dan 8 kolom. Kita juga dapat melihat tipe data untuk setiap kolom. Mayoritas fitur numerik (NO, HARGA, LB, LT, KT, KM, GRS) terdeteksi sebagai integer atau float, yang sesuai dengan sifatnya. Kolom 'NAMA RUMAH' memiliki tipe objek, yang mengindikasikan tipe data string atau campuran, dan kemungkinan memerlukan penanganan lebih lanjut jika akan digunakan dalam pemodelan. Yang menggembirakan, semua kolom memiliki 1010 nilai non-null, yang berarti tidak ada *missing values* yang perlu kita atasi.

# Statistik Deskriptif Data Numerik
Dengan `df.describe()`, kita mendapatkan ringkasan statistik penting dari fitur-fitur numerik. Beberapa poin menarik untuk diperhatikan:
- **Rentang Harga yang Luas:** Statistik menunjukkan rentang harga yang sangat lebar, dari 4.3e+08 hingga 6.5e+10. Standar deviasi harga yang tinggi (7.34e+09) mengindikasikan variabilitas harga yang signifikan dalam dataset.
- **Sebaran Luas Tanah dan Bangunan:** Mirip dengan harga, fitur 'LB' dan 'LT' juga memiliki rentang dan standar deviasi yang cukup besar, menunjukkan variasi ukuran properti yang signifikan.
- **Distribusi Jumlah Kamar:** Fitur 'KT', 'KM', dan 'GRS' memiliki rentang nilai yang lebih kecil dan standar deviasi yang moderat, mencerminkan variasi jumlah kamar dan garasi yang lebih terbatas.
- **Potensi Skewness:** Perbandingan antara nilai mean dan median untuk 'HARGA', 'LB', dan 'LT' menunjukkan adanya potensi *skewness*. Misalnya, mean harga (7.63e+09) jauh lebih tinggi daripada median harga (5.00e+09), yang mengindikasikan distribusi yang miring ke kanan, di mana terdapat beberapa properti dengan harga yang sangat tinggi yang menarik rata-rata ke atas. Hal serupa terlihat pada luas tanah dan bangunan.

# Pemeriksaan Nilai yang Hilang
Pemeriksaan dengan `df.isnull().sum()` mengonfirmasi bahwa tidak ada nilai yang hilang (*missing values*) dalam dataset. Ini menghilangkan kebutuhan untuk imputasi atau penghapusan data terkait *missing values*, dan memungkinkan kita untuk melanjutkan ke tahap analisis berikutnya dengan dataset yang lengkap.

# Pemeriksaan Data Duplikat
Analisis dengan `df.duplicated().sum()` menunjukkan bahwa tidak ada baris yang terduplikasi dalam dataset. Setiap entri dalam dataset bersifat unik, sehingga kita tidak perlu khawatir tentang bias atau redundansi yang disebabkan oleh data yang berulang.

# Visualisasi Distribusi Fitur Numerik
Histogram dari setiap fitur numerik memberikan wawasan visual tentang sebaran datanya:
- **'NO':** Histogram menunjukkan distribusi yang hampir seragam, yang sesuai dengan sifatnya sebagai indeks.
- **'HARGA', 'LB', 'LT':** Histogram untuk ketiga fitur ini jelas menunjukkan *positive skewness*. Sebagian besar data terkumpul di nilai yang lebih rendah, dengan ekor yang memanjang ke nilai yang lebih tinggi. Distribusi ini mengindikasikan bahwa properti dengan harga dan ukuran yang sangat besar relatif jarang, tetapi ada. Transformasi data mungkin diperlukan untuk menangani *skewness* ini dalam beberapa model.
- **'KT', 'KM', 'GRS':** Histogram untuk jumlah kamar tidur, kamar mandi, dan garasi menunjukkan distribusi diskrit. 'KT' dan 'KM' cenderung terpusat pada nilai tertentu (misalnya, 4-5 kamar tidur dan 3 kamar mandi adalah yang paling umum), dengan frekuensi yang menurun untuk nilai yang lebih ekstrem. 'GRS' didominasi oleh 0, 1, dan 2 garasi/carport. Distribusi diskrit ini perlu dipertimbangkan saat memilih model.

# Identifikasi Fitur Kategorikal
Satu-satunya fitur yang jelas bersifat kategorikal dalam dataset ini adalah kolom 'NAMA RUMAH'. Kolom ini berisi deskripsi tekstual properti. Untuk pemodelan numerik saat ini, kolom ini tidak akan digunakan secara langsung. Namun, penting untuk dicatat bahwa informasi dalam kolom ini berpotensi diekstraksi dan diubah menjadi fitur numerik atau kategorikal baru melalui teknik *Natural Language Processing* atau *feature engineering* di masa mendatang.

# Pemeriksaan Outlier pada Data Target ('HARGA')
Meskipun statistik deskriptif mengindikasikan potensi *outlier* harga, visualisasi menggunakan *boxplot* tidak secara eksplisit menunjukkan adanya titik-titik yang jauh melampaui *whisker*. Ini bisa berarti bahwa distribusi harga memang memiliki rentang yang lebar dan *skewness* yang signifikan, daripada adanya nilai-nilai ekstrem yang terisolasi. Namun, perlu diingat bahwa definisi *outlier* bisa subjektif dan tergantung pada konteks masalah. Distribusi harga yang miring tetap perlu dipertimbangkan dalam pemilihan dan evaluasi model.

# Analisis Korelasi Antar Fitur Numerik
Heatmap korelasi memberikan wawasan tentang hubungan linear antar fitur numerik:
- **Korelasi Positif dengan Harga:** 'LB' (0.75) dan 'LT' (0.81) menunjukkan korelasi positif yang kuat dengan 'HARGA'. Ini secara intuitif masuk akal bahwa rumah dengan luas bangunan dan tanah yang lebih besar cenderung memiliki harga yang lebih tinggi. 'KT' (0.32), 'KM' (0.40), dan 'GRS' (0.48) juga berkorelasi positif dengan harga, tetapi dengan kekuatan yang lebih rendah.
- **Korelasi Antar Fitur Prediktor:** Terdapat korelasi positif yang cukup kuat antara 'LB' dan 'LT' (0.74), yang menunjukkan bahwa rumah dengan luas bangunan yang besar juga cenderung memiliki luas tanah yang besar. Selain itu, 'KT' dan 'KM' juga berkorelasi positif (0.67), yang logis karena rumah dengan lebih banyak kamar tidur biasanya juga memiliki lebih banyak kamar mandi.
- **Implikasi untuk Modeling:** Korelasi yang tinggi antar fitur prediktor ('LB' dan 'LT') dapat mengindikasikan adanya *multikolinearitas*, yang dapat mempengaruhi interpretasi koefisien dalam model linear. Untuk model seperti Random Forest, *multikolinearitas* biasanya tidak menjadi masalah besar dalam hal performa prediksi, tetapi tetap perlu dipertimbangkan dalam analisis fitur. Korelasi fitur dengan target ('HARGA') memberikan indikasi awal tentang fitur mana yang mungkin menjadi prediktor yang kuat.
Pemahaman tentang korelasi ini akan membantu dalam pemilihan fitur dan jenis model yang akan digunakan dalam tahap pemodelan selanjutnya.

## Data Preparation
Karena algoritma Random Forest dan Linear Regression tidak memerlukan normalisasi atau scaling fitur, tahap Data Preparation dalam proyek ini cukup dengan memilih fitur yang relevan dan membagi data menjadi data latih dan data uji tanpa perlu melakukan transformasi lebih lanjut terhadap data.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Memilih Fitur dan Target: Langkah pertama adalah memilih fitur yang akan digunakan dalam model prediksi harga rumah, serta menentukan target yang akan diprediksi. Dalam hal ini, fitur yang dipilih mencakup kolom-kolom yang berhubungan dengan karakteristik rumah seperti luas bangunan (LB), luas tanah (LT), jumlah kamar tidur (KT), jumlah kamar mandi (KM), dan keberadaan garasi (GRS). Target yang diprediksi adalah harga rumah (HARGA).
- Membagi Data Menjadi Data Latih dan Data Uji: Data kemudian dibagi menjadi dua bagian: data latih (training data) dan data uji (test data). Proporsi yang digunakan adalah 80% untuk data latih dan 20% untuk data uji. Pembagian ini dilakukan untuk memastikan model dapat dilatih dengan data yang cukup, sementara data uji digunakan untuk mengevaluasi performa model yang telah dilatih. Pembagian ini dilakukan secara acak dengan menetapkan random_state=42 untuk memastikan hasil yang konsisten setiap kali eksperimen dijalankan

## Modeling
Pada tahap ini, kami menggunakan dua model machine learning untuk memprediksi harga rumah, yaitu Random Forest Regressor dan Linear Regression. Kedua algoritma ini memiliki karakteristik yang berbeda dan dipilih untuk melihat mana yang paling cocok dengan dataset yang ada.
## Model 1: Random Forest Regressor

### Cara Kerja

Random Forest adalah algoritma *ensemble learning* yang bekerja dengan membangun banyak pohon keputusan (*decision trees*) selama proses pelatihan. Untuk melakukan prediksi, setiap pohon dalam hutan akan memberikan prediksi dan hasil akhir prediksi model adalah rata-rata (untuk regresi) atau mayoritas (untuk klasifikasi) dari prediksi semua pohon.

### Parameter

Dalam kode yang diberikan, model Random Forest Regressor diinisialisasi dengan parameter berikut:

* `n_estimators=100`: Parameter ini menentukan jumlah pohon keputusan yang akan dibangun dalam hutan. Semakin banyak pohon biasanya meningkatkan kinerja model dan membuatnya lebih stabil, tetapi juga meningkatkan waktu pelatihan. Dalam kasus ini, 100 pohon akan dibuat.
* `random_state=42`: Parameter ini digunakan untuk mengontrol keacakan dalam proses pembangunan hutan. Dengan menetapkan `random_state` ke nilai tertentu (dalam hal ini 42), kita memastikan bahwa hasil yang diperoleh akan selalu sama jika kode dijalankan kembali dengan data yang sama. Ini penting untuk reproduktifitas eksperimen.

### Kelebihan (Opsional)

* Cenderung memberikan akurasi prediksi yang baik.
* Mampu menangani dataset dengan dimensi tinggi.
* Kurang rentan terhadap *overfitting* dibandingkan dengan satu pohon keputusan tunggal, terutama dengan jumlah pohon yang banyak.
* Dapat memberikan estimasi pentingnya fitur.
* Tidak memerlukan banyak pra-pemrosesan data.

### Kekurangan (Opsional)

* Model yang dihasilkan bisa sulit diinterpretasikan dibandingkan dengan satu pohon keputusan.
* Waktu pelatihan bisa lebih lama, terutama dengan jumlah pohon yang besar.

## Model 2: Linear Regression

### Cara Kerja

Linear Regression adalah algoritma model linear yang bertujuan untuk memodelkan hubungan antara variabel dependen (target) dan satu atau lebih variabel independen (fitur) dengan menyesuaikan persamaan linear ke data yang diamati. 

### Parameter



### Kelebihan (Opsional)

* Mudah diimplementasikan dan diinterpretasikan.
* Secara komputasi relatif murah, terutama untuk dataset yang tidak terlalu besar.
* Berfungsi dengan baik ketika hubungan antara variabel dependen dan independen bersifat linear.

### Kekurangan (Opsional)

* Mengasumsikan hubungan linear antara fitur dan target. Jika hubungannya non-linear, performanya mungkin buruk.
* Sangat sensitif terhadap *outlier*.
* Dapat menderita *multikolinearitas* (korelasi tinggi antar fitur independen), yang dapat mempengaruhi stabilitas koefisien.

**Rubrik/Kriteria Tambahan (Opsional)**: 
Random Forest Regressor adalah model ensemble yang membangun beberapa pohon keputusan dan menggabungkan hasil prediksi dari pohon-pohon tersebut. Setiap pohon dibangun menggunakan subset acak dari data dan fitur, yang meningkatkan robusta dan akurasi model. Random Forest dapat menangani hubungan non-linier dengan baik dan lebih tahan terhadap overfitting.

Pada tahap pemodelan, dua algoritma regresi digunakan: Linear Regression dan Random Forest. Linear Regression mencoba memodelkan hubungan secara linear dan dipengaruhi oleh asumsi linearitas dan outlier. Parameter fit_intercept memungkinkan model untuk memiliki konstanta. Random Forest, sebagai ensemble dari pohon keputusan, lebih baik dalam menangkap hubungan non-linear dan lebih tahan terhadap outlier. Berdasarkan evaluasi, Random Forest menunjukkan performa prediksi yang lebih baik (MAE dan RMSE lebih rendah, R-squared lebih tinggi), sehingga direkomendasikan untuk tugas prediksi harga rumah ini karena kemampuannya menangani kompleksitas data.

## Evaluation
Pada tahap evaluasi menggunakan beberapa metrik evaluasi untuk mengukur kinerja kedua model yang diterapkan, yaitu Random Forest Regressor dan Linear Regression. Metrik yang digunakan adalah Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE), dan R-squared (R²). Metrik-metrik ini dipilih karena relevan untuk masalah regresi, yaitu prediksi harga rumah.

Evaluasi Model Random Forest vs Linear Regression:

Mean Absolute Error (MAE) - Random Forest: 1759550222.1703582
Mean Squared Error (MSE) - Random Forest: 1.0635204265435824e+19
Root Mean Squared Error (RMSE) - Random Forest: 3261166089.8267393
R-squared (R²) - Random Forest: 0.772181577855092

Mean Absolute Error (MAE) - Linear Regression: 1980345761.1519253
Mean Squared Error (MSE) - Linear Regression: 1.0675729074057431e+19
Root Mean Squared Error (RMSE) - Linear Regression: 3267373421.2754793
R-squared (R²) - Linear Regression: 0.7713134894077545

Secara keseluruhan, Random Forest lebih unggul dibandingkan dengan Linear Regression berdasarkan MAE, MSE, dan RMSE. Meskipun R² keduanya hampir sama, Random Forest menunjukkan performa yang lebih baik dalam hal kesalahan prediksi rata-rata dan penalti terhadap kesalahan besar, yang membuatnya menjadi pilihan model yang lebih baik untuk proyek ini.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- MAE mengukur rata-rata selisih absolut antara nilai prediksi dan nilai aktual. Metrik ini memberi gambaran tentang seberapa besar rata-rata kesalahan dalam prediksi, tanpa memperhitungkan arah kesalahan (positif atau negatif).
- MSE mengukur rata-rata kuadrat selisih antara nilai prediksi dan nilai aktual. Metrik ini lebih sensitif terhadap kesalahan besar (outliers), karena kesalahan besar dihitung kuadratnya.
- RMSE adalah akar kuadrat dari MSE dan memberikan gambaran seberapa jauh prediksi rata-rata dari nilai sebenarnya dalam satuan yang sama dengan data. RMSE memberikan penalti yang lebih besar pada kesalahan besar.
- R² mengukur seberapa baik model dapat menjelaskan variansi dalam data target. Nilai R² berkisar antara 0 dan 1, di mana semakin dekat nilai R² ke 1, semakin baik model dalam menjelaskan variansi data.

**Hubungan dengan Business Understanding:**

* **Problem Statement:** Kedua model menjawab pertanyaan prediksi harga berdasarkan fitur. Random Forest lebih akurat (MAE, RMSE lebih rendah).
* **Goals:** Tujuan mengembangkan model prediksi tercapai. Peningkatan akurasi melalui pemilihan Random Forest juga tercapai.
* **Solution Statement:**
    * Penerapan algoritma regresi berhasil membangun model prediksi.
    * Perbandingan model efektif dalam memilih Random Forest yang berkinerja lebih baik secara kuantitatif.

**Kesimpulan:**

Random Forest Regressor menunjukkan performa prediksi harga rumah yang lebih unggul dibandingkan Linear Regression berdasarkan metrik evaluasi. Pemilihan dan perbandingan model ini secara langsung menjawab *problem statement* dan mencapai *goals* yang ditetapkan dalam *business understanding*.


