
# Laporan Proyek Predictive Analytics - Machine Learning Terapan - Muhammad Rizki

## 📌 Domain Proyek

Diabetes merupakan salah satu penyakit kronis yang semakin banyak diderita oleh penduduk dunia. Menurut data Organisasi Kesehatan Dunia (WHO), terdapat sekitar 422 juta orang di dunia yang menderita diabetes, dan angka ini terus meningkat setiap tahunnya [1]. Komplikasi dari diabetes dapat menyebabkan berbagai masalah kesehatan serius seperti penyakit jantung, gagal ginjal, kebutaan, dan bahkan kematian.

Deteksi dini diabetes sangat penting untuk pengelolaan dan pencegahan komplikasi penyakit. Dengan kemajuan teknologi machine learning, kita dapat mengembangkan model prediktif yang membantu mengidentifikasi individu yang berisiko terkena diabetes berdasarkan faktor-faktor klinis tertentu. Hal ini memungkinkan tenaga medis untuk melakukan intervensi lebih awal dan membantu pasien dalam mengelola kondisi mereka dengan lebih efektif.

Proyek ini bertujuan untuk mengembangkan model machine learning yang dapat memprediksi kemungkinan seorang individu mengidap diabetes berdasarkan berbagai parameter klinis. Dataset yang digunakan adalah **Pima Indians Diabetes Database**, yang berisi informasi medis dari populasi penduduk asli Amerika (Pima Indians), yang dikenal memiliki tingkat prevalensi diabetes yang tinggi [2].

---

## 🎯 Business Understanding

### ❓ Problem Statements

1. Bagaimana kita dapat memprediksi apakah seseorang menderita diabetes berdasarkan faktor-faktor klinis dengan tingkat akurasi yang tinggi?
2. Faktor klinis apa yang memiliki pengaruh terbesar terhadap kemungkinan seseorang menderita diabetes?
3. Bagaimana menangani ketidakseimbangan kelas dalam data diabetes untuk menghasilkan model prediksi yang lebih baik?

### 🎯 Goals

1. Mengembangkan model klasifikasi yang dapat memprediksi apakah seseorang menderita diabetes dengan akurasi minimal 80%.
2. Mengidentifikasi dan menganalisis faktor-faktor klinis yang paling berpengaruh terhadap perkembangan diabetes.
3. Menerapkan teknik penanganan ketidakseimbangan kelas untuk meningkatkan performa model, terutama dalam hal recall.

### 💡 Solution Statements

1. Mengembangkan dan membandingkan beberapa model klasifikasi:
   - Logistic Regression
   - Decision Tree
   - Random Forest
   - Support Vector Machine (SVM)
   - Gradient Boosting
2. Menerapkan teknik SMOTE untuk menangani ketidakseimbangan kelas.
3. Melakukan hyperparameter tuning pada model terbaik.
4. Mengevaluasi model menggunakan metrik: Accuracy, Precision, Recall, F1-score, ROC-AUC.

---

## 📊 Data Understanding

Dataset: [Pima Indians Diabetes Database - Kaggle](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
Dataset ini berisi data medis dari 768 wanita dari populasi asli Amerika (Pima Indians). Data ini dikumpulkan oleh National Institute of Diabetes and Digestive and Kidney Diseases.
### 🔎 Variabel

| Nama Fitur               | Deskripsi |
|--------------------------|-----------|
| Pregnancies              | Jumlah kehamilan |
| Glucose                  | Konsentrasi glukosa plasma (mg/dL) |
| BloodPressure            | Tekanan darah diastolik (mm Hg) |
| SkinThickness            | Ketebalan lipatan kulit trisep (mm) |
| Insulin                  | Insulin serum 2 jam (mu U/ml) |
| BMI                      | Indeks Massa Tubuh |
| DiabetesPedigreeFunction | Riwayat keluarga diabetes |
| Age                      | Usia (tahun) |
| Outcome                  | Target: 1 = diabetes, 0 = tidak |

### 📈 Distribusi Kelas Target 
![image](https://github.com/user-attachments/assets/66b6bfb9-52f4-4f7e-9e9b-91584691cc00)

- Kelas 0 (Tidak diabetes): 500 sampel (65.1%)
- Kelas 1 (Diabetes): 268 sampel (34.9%)

### ⚠️ Nilai Tidak Valid

| Fitur         | Jumlah Nol |
|---------------|-------------|
| Glucose       | 5           |
| BloodPressure | 35          |
| SkinThickness | 227         |
| Insulin       | 374         |
| BMI           | 11          |

### Korelasi antar Variabel 
![image](https://github.com/user-attachments/assets/ad42403b-eff5-4229-961f-37055790f1e1)
![image](https://github.com/user-attachments/assets/b5aa4433-a1f0-490c-8b6b-ffa3e80f8de7)

Analisis korelasi menunjukkan bahwa beberapa fitur memiliki korelasi yang signifikan dengan status diabetes (*Outcome*):
- **Glucose** memiliki korelasi positif tertinggi dengan diabetes, yang sesuai dengan pengetahuan medis.
- **BMI** dan **Age** juga menunjukkan korelasi positif yang cukup kuat.
- **DiabetesPedigreeFunction**, yang mencerminkan riwayat keluarga, juga berkorelasi dengan status diabetes.

beberapa insight yang dapat kita peroleh:

1. **Distribusi Kelas**: Dataset tidak seimbang, dengan lebih banyak kasus non-diabetes (kelas 0) dibandingkan
   kasus diabetes (kelas 1). Hal ini perlu diperhatikan dalam pemodelan.

2. **Glukosa dan Outcome**: Terdapat korelasi positif yang kuat antara kadar glukosa dan diabetes,
   yang sesuai dengan pengetahuan medis bahwa kadar glukosa tinggi merupakan indikator utama diabetes.

3. **BMI dan Diabetes**: BMI juga menunjukkan korelasi yang cukup dengan outcome diabetes,
   mengkonfirmasi bahwa obesitas merupakan faktor risiko diabetes.

4. **Usia**: Pasien yang lebih tua cenderung memiliki risiko diabetes lebih tinggi, ditunjukkan dengan
   korelasi positif antara usia dan outcome.

5. **Kehamilan**: Jumlah kehamilan memiliki korelasi positif dengan diabetes, menunjukkan bahwa
   wanita dengan lebih banyak riwayat kehamilan mungkin memiliki risiko lebih tinggi.

6. **Outliers**: Terdapat outlier di beberapa fitur seperti Insulin dan SkinThickness yang perlu ditangani
   dalam tahap pra-pemrosesan data.

---
## 🧹 Data Preparation
Beberapa teknik persiapan data yang diterapkan dalam proyek ini:

### 1. Penanganan Nilai yang Tidak Valid

Nilai nol pada fitur klinis seperti *Glucose*, *BloodPressure*, *SkinThickness*, *Insulin*, dan *BMI* dianggap tidak valid secara medis karena fitur-fitur ini tidak seharusnya bernilai nol pada manusia yang hidup.

Nilai-nilai nol tersebut diganti dengan nilai median dari masing-masing kelompok berdasarkan status diabetes (*Outcome*) untuk menjaga konsistensi distribusi data antar kelas.

> Pendekatan ini lebih baik daripada mengganti dengan median keseluruhan karena mempertahankan karakteristik distribusi dari masing-masing kelompok, sehingga hasil pelatihan model tidak bias terhadap kelompok tertentu.

### 2. Pembagian Data

Data dibagi menjadi data training (80%) dan data testing (20%) dengan menggunakan teknik stratifikasi berdasarkan variabel target (*Outcome*).

> Stratifikasi memastikan bahwa proporsi antara kelas positif dan negatif tetap seimbang di kedua subset data, yang penting untuk menjaga performa model terutama pada data yang memiliki ketidakseimbangan kelas.

### 3. Standarisasi Data

Standarisasi dilakukan terhadap semua fitur numerik menggunakan _StandardScaler_, yang mengubah distribusi data agar memiliki rata-rata 0 dan standar deviasi 1.

> Ini penting karena beberapa algoritma pembelajaran mesin seperti *Logistic Regression* dan *SVM* sangat sensitif terhadap skala fitur, sehingga standarisasi membantu model dalam melakukan optimasi dengan lebih efektif.

### 4. Penanganan Ketidakseimbangan Kelas dengan SMOTE

Ketidakseimbangan jumlah sampel antara kelas diabetes dan non-diabetes pada data training diatasi dengan menggunakan teknik **SMOTE (Synthetic Minority Over-sampling Technique)**.

> SMOTE menghasilkan sampel sintetis baru dari kelas minoritas dengan cara interpolasi terhadap tetangga terdekat, sehingga menghindari duplikasi langsung dan memperkaya variasi data. Hal ini meningkatkan kemampuan model dalam mendeteksi kelas minoritas secara adil.
```python
# Menerapkan SMOTE pada data training
smote = SMOTE(random_state=42)
```
X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)
Setelah penerapan SMOTE, distribusi kelas menjadi seimbang:
![image](https://github.com/user-attachments/assets/329f7abc-7d23-43a6-adf7-d475d2bfdb60)

- **Kelas 0 (Tidak diabetes)**: 400 sampel  
- **Kelas 1 (Diabetes)**: 400 sampel
---
## Modeling

### Pengembangan Model

Dalam proyek ini, lima algoritma klasifikasi berbeda dibandingkan untuk menemukan model terbaik:

- **Logistic Regression**: Model linear yang sederhana namun mudah diinterpretasikan, cocok sebagai baseline model untuk masalah klasifikasi biner.
- **Decision Tree**: Model berbasis aturan yang dapat menangkap pola non-linear dan memberikan interpretasi yang jelas melalui struktur pohon keputusan.
- **Random Forest**: Ensemble dari pohon keputusan yang dapat menangani kompleksitas data dan mengurangi overfitting yang sering terjadi pada Decision Tree tunggal.
- **Support Vector Machine (SVM)**: Algoritma yang efektif untuk data berdimensi tinggi dan bekerja dengan mencari hyperplane optimal yang memisahkan kelas.
- **Gradient Boosting**: Teknik ensemble yang membangun model secara sekuensial, di mana setiap model bertujuan memperbaiki kesalahan dari model sebelumnya.

#### Kelebihan dan Kekurangan Model

##### 1. **Logistic Regression**
**Kelebihan:**
- Sederhana dan cepat untuk dilatih
- Mudah diinterpretasikan (koefisien menunjukkan pengaruh fitur)
- Cocok untuk klasifikasi biner
- Tidak rentan terhadap overfitting jika jumlah fitur tidak terlalu banyak

**Kekurangan:**
- Hanya menangkap hubungan linear antar fitur
- Performa menurun pada data dengan pola non-linear
- Sensitif terhadap outlier

##### 2. **Decision Tree**
**Kelebihan:**
- Dapat menangkap hubungan non-linear
- Mudah diinterpretasikan melalui struktur pohon
- Tidak memerlukan normalisasi data
- Menangani fitur numerik dan kategorik

**Kekurangan:**
- Rentan terhadap overfitting
- Performa tidak stabil pada data yang sedikit berubah
- Kurang efektif pada data berdimensi tinggi


##### 3. **Random Forest**
**Kelebihan:**
- Mengurangi risiko overfitting dari pohon tunggal
- Dapat menangani dataset besar dan kompleks
- Menyediakan informasi pentingnya fitur
- Cukup robust terhadap outlier dan noise

**Kekurangan:**
- Lebih sulit untuk diinterpretasikan dibanding Decision Tree
- Memerlukan sumber daya komputasi lebih besar
- Waktu pelatihan lebih lama dari model sederhana

##### 4. **Support Vector Machine (SVM)**
**Kelebihan:**
- Efektif pada data berdimensi tinggi
- Dapat menangani margin klasifikasi sempit dengan kernel trick
- Cocok untuk data dengan pemisahan kelas yang jelas

**Kekurangan:**
- Tidak cocok untuk dataset besar (komputasi mahal)
- Sulit diinterpretasikan
- Sensitif terhadap pemilihan kernel dan parameter

##### 5. **Gradient Boosting**
**Kelebihan:**
- Akurasi tinggi karena belajar dari kesalahan model sebelumnya
- Dapat menangani data yang kompleks dan tidak seimbang
- Memiliki fleksibilitas tinggi dengan berbagai fungsi loss

**Kekurangan:**
- Rentan terhadap overfitting jika tidak dituning dengan baik
- Memerlukan waktu pelatihan yang lama
- Parameter tuning lebih kompleks

#### Definisi Model

Setiap model didefinisikan dengan parameter dasar sebagai berikut:

- **Logistic Regression**: `max_iter=1000`, `random_state=42`  
  (Parameter `max_iter` ditingkatkan untuk memastikan konvergensi)
  
- **Decision Tree**: `random_state=42`  
  (Digunakan untuk menghindari hasil yang acak antar eksekusi)

- **Random Forest**: `random_state=42`  
  (Secara default menggunakan 100 pohon dan pembobotan fitur secara acak)

- **SVM**: `probability=True`, `random_state=42`  
  (`probability=True` digunakan untuk menghasilkan probabilitas prediksi, yang berguna untuk evaluasi lebih lanjut)

- **Gradient Boosting**: `random_state=42`  
  (Digunakan default setting dengan booster berbasis pohon)

#### Evaluasi Model

Setiap model dievaluasi menggunakan **5-Fold Cross-Validation** dengan metrik **accuracy**. Teknik ini membagi data training menjadi lima bagian, melatih model pada empat bagian, dan mengujinya pada satu bagian, lalu mengulang proses tersebut lima kali untuk menghasilkan skor yang lebih stabil dan representatif.

> Cross-validation penting untuk mengukur kinerja model secara lebih objektif dan menghindari overfitting terhadap data training.


#### Hasil Evaluasi Awal (Cross-Validation 5-Fold)

| Model                | Akurasi Rata-rata ± Deviasi Standar |
|----------------------|--------------------------------------|
| Logistic Regression  | 0.7862 ± 0.0214                      |
| Decision Tree        | 0.8562 ± 0.0198                      |
| Random Forest        | 0.9000 ± 0.0240                      |
| SVM                  | 0.8712 ± 0.0085                      |
| Gradient Boosting    | 0.8812 ± 0.0230                      |

### Pemilihan Model Terbaik

Berdasarkan hasil cross-validation, **Random Forest** memberikan performa terbaik dengan akurasi rata-rata **0.9000**. Model ini kemudian dipilih untuk dioptimalkan lebih lanjut melalui proses *hyperparameter tuning*.

**Alasan pemilihan Random Forest**:

- Memberikan akurasi tertinggi pada validasi silang
- Mampu menangani berbagai jenis data dengan baik
- Lebih tahan terhadap overfitting dibandingkan Decision Tree tunggal
- Menyediakan informasi mengenai pentingnya fitur (*feature importance*)

### Hyperparameter Tuning

*Grid Search* dilakukan untuk menemukan kombinasi parameter optimal untuk model Random Forest. Parameter yang dioptimalkan meliputi:

- `n_estimators`: Jumlah pohon dalam forest
- `max_depth`: Kedalaman maksimum setiap pohon
- `min_samples_split`: Jumlah minimum sampel untuk membelah simpul internal
- `min_samples_leaf`: Jumlah minimum sampel yang diperlukan di simpul daun

**Hasil terbaik dari tuning parameter**:

- `n_estimators`: 300  
- `max_depth`: None  
- `min_samples_split`: 2  
- `min_samples_leaf`: 1  

Dengan akurasi cross-validation akhir sebesar **0.9037**.

---
## Evaluation

### Metrik Evaluasi

Beberapa metrik evaluasi digunakan untuk mengukur performa model:

- **Accuracy**: Persentase total prediksi yang benar.  
  - **Formula**: (TP + TN) / (TP + TN + FP + FN)  
  - **Keterangan**:  
    - TP = True Positive  
    - TN = True Negative  
    - FP = False Positive  
    - FN = False Negative  

- **Precision**: Persentase prediksi positif yang benar.  
  - **Formula**: TP / (TP + FP)  
  - Mengukur seberapa akurat model dalam mengidentifikasi kasus diabetes.

- **Recall (Sensitivity)**: Persentase kasus positif aktual yang berhasil diidentifikasi.  
  - **Formula**: TP / (TP + FN)  
  - Mengukur kemampuan model untuk mendeteksi semua kasus diabetes.

- **F1-score**: Rata-rata harmonik dari precision dan recall.  
  - **Formula**: 2 * (Precision * Recall) / (Precision + Recall)  
  - Memberikan keseimbangan antara precision dan recall.

- **ROC-AUC**: Area di bawah kurva ROC, mengukur kemampuan model untuk membedakan antara kelas.  
  - Nilai berkisar dari 0 hingga 1, dengan 1 menunjukkan performa klasifikasi sempurna.


### Hasil Evaluasi
Setelah mendefinisikan lima model klasifikasi, dilakukan evaluasi menggunakan **5-fold cross-validation** pada data hasil SMOTE. Hasil rata-rata akurasi (mean accuracy) dan standar deviasi dari setiap model adalah sebagai berikut:

| Model                | Mean CV Accuracy | Std Dev |
|----------------------|------------------|---------|
| Logistic Regression  | 0.7862           | ± 0.0214 |
| Decision Tree        | 0.8562           | ± 0.0198 |
| Random Forest        | **0.9000**       | ± 0.0240 |
| SVM                  | 0.8712           | ± 0.0085 |
| Gradient Boosting    | 0.8812           | ± 0.0230 |

Dari hasil tersebut, **Random Forest** menunjukkan performa terbaik sebelum dilakukan tuning hyperparameter.
### Hyperparameter Tuning dengan Grid Search

Berdasarkan hasil evaluasi awal, **Random Forest** dipilih sebagai kandidat model terbaik. Untuk meningkatkan performanya, dilakukan **Grid Search** menggunakan kombinasi parameter berikut:

- `n_estimators`: [100, 200, 300]
- `max_depth`: [None, 10, 20, 30]
- `min_samples_split`: [2, 5, 10]
- `min_samples_leaf`: [1, 2, 4]

Hasil terbaik dari tuning adalah:

- **Best Parameters**: `{'max_depth': None, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 300}`
- **Best CV Accuracy**: `0.9037`
Hasil evaluasi model **Random Forest terbaik** pada data testing:

- **Accuracy**: 0.8571  
- **Precision**: 0.7963  
- **Recall**: 0.7963  
- **F1-score**: 0.7963  
- **ROC-AUC**: 0.94  

### Analisis Feature Importance

Analisis feature importance dari model Random Forest menunjukkan fitur-fitur yang paling berpengaruh dalam prediksi diabetes:

| Fitur                     | Importance |
|---------------------------|------------|
| Insulin                   | 0.3621     |
| SkinThickness             | 0.1610     |
| Glucose                   | 0.1467     |
| BMI                       | 0.0949     |
| Age                       | 0.0753     |
| DiabetesPedigreeFunction  | 0.0673     |
| BloodPressure             | 0.0501     |
| Pregnancies               | 0.0427     |

Hasil ini menunjukkan bahwa kadar **Insulin**, **ketebalan kulit**, dan **glukosa** merupakan faktor paling penting dalam prediksi diabetes, yang sejalan dengan pengetahuan medis tentang penyakit ini.

---

## Kesimpulan

Proyek ini berhasil mengembangkan model machine learning untuk memprediksi diabetes berdasarkan faktor-faktor klinis. Beberapa kesimpulan utama yang diperoleh:

- Model **Random Forest** yang telah dioptimalkan menunjukkan performa terbaik, dengan akurasi sebesar **85.71%** pada data testing.
- Penggunaan teknik **SMOTE** untuk menangani ketidakseimbangan kelas terbukti efektif, khususnya dalam meningkatkan **recall** model.
- Faktor-faktor klinis yang paling berpengaruh dalam prediksi diabetes meliputi:
  - Kadar **Insulin**
  - **SkinThickness** (ketebalan kulit)
  - Kadar **Glukosa**
- Model ini memiliki potensi sebagai **alat bantu skrining awal** bagi tenaga medis dalam mendeteksi diabetes, namun **bukan sebagai pengganti diagnosis profesional**.

---

### Rekomendasi untuk Pengembangan Lebih Lanjut

- Mengumpulkan data tambahan dengan fitur yang lebih lengkap, seperti **pola makan**, **aktivitas fisik**, dan **riwayat medis**, untuk meningkatkan performa model.
- Mengeksplorasi penggunaan **deep learning** seperti **Neural Networks**, terutama jika dataset diperbesar.
- Menerapkan **model ensemble** yang menggabungkan prediksi dari beberapa model terbaik untuk meningkatkan **robustness** dan **akurasi**.
- Melakukan validasi pada **dataset eksternal** untuk memastikan kemampuan generalisasi model di populasi yang berbeda.

---

## Referensi

1. World Health Organization. "Diabetes," 2021. [Online]. Tersedia di: [https://www.who.int/news-room/fact-sheets/detail/diabetes](https://www.who.int/news-room/fact-sheets/detail/diabetes)  
2. Smith, J. W., Everhart, J. E., Dickson, W. C., Knowler, W. C., & Johannes, R. S. (1988). *Using the ADAP learning algorithm to forecast the onset of diabetes mellitus*. Proceedings of the Annual Symposium on Computer Application in Medical Care, 261.  
3. Kaggle. "Pima Indians Diabetes Database". [Online]. Tersedia di: [https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)  
4. Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). *SMOTE: Synthetic Minority Over-sampling Technique*. Journal of Artificial Intelligence Research, 16, 321–357.  
5. Breiman, L. (2001). *Random Forests*. Machine Learning, 45(1), 5–32.

