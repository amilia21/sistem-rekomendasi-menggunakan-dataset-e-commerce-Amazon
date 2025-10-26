# Laporan Proyek Machine Learning Terapan - Ami Lia Rizki
## Project Overview

Dalam era digital seperti sekarang, jumlah produk yang tersedia di platform e-commerce seperti Amazon sangat besar, sehingga pengguna sering merasa kesulitan menemukan produk yang sesuai dengan kebutuhan dan preferensinya. Di sisi lain, perusahaan e-commerce perlu memberikan pengalaman belanja yang lebih personal agar pengguna merasa puas dan loyal terhadap platform tersebut.
Melihat permasalahan ini, saya tertarik untuk membangun sistem rekomendasi karena sistem ini memiliki peran penting dalam meningkatkan kepuasan pengguna, retensi pelanggan, dan potensi penjualan. Khususnya, dengan memanfaatkan data rating dan interaksi pengguna terhadap produk elektronik, kita dapat menerapkan teknik collaborative filtering untuk memahami pola preferensi pengguna dan memberikan rekomendasi yang relevan.
Selain itu, proyek ini juga menjadi sarana untuk mendalami konsep machine learning di bidang rekomendasi sistem, memahami pengolahan data besar (big data), dan menerapkan algoritma yang digunakan oleh perusahaan teknologi besar seperti Amazon.
Secara pribadi, saya termotivasi karena proyek ini menggabungkan analisis data, pemrograman, dan penerapan langsung konsep AI yang berdampak nyata terhadap dunia industri, terutama di bidang e-commerce dan data science.

 Rererensi:
[1] G. Linden, B. Smith, and J. York, “Amazon.com recommendations: Item-to-item collaborative filtering,” IEEE Internet Computing, vol. 7, no. 1, pp. 76–80, Jan.–Feb. 2003, doi: 10.1109/MIC.2003.1167344.

[2] J. M. Davidson, B. Smith, and G. Linden, “Two decades of recommender systems at Amazon.com,” Amazon Science, 2021. [Online]. Available: https://assets.amazon.science/76/9e/7eac89c14a838746e91dde0a5e9f/two-decades-of-recommender-systems-at-amazon.pdf.

[3] M. S. Rahman and S. Ahmed, “Performance analysis of three recommendation algorithms on Amazon datasets,” ResearchGate, Apr. 2024. [Online]. Available: https://www.researchgate.net/publication/379259297_Performance_analysis_of_three_recommendation_algorithms_on_Amazon_datasets.

[4] T. Dacrema, P. Cremonesi, and D. Jannach, “A critical study on data leakage in recommender system offline evaluation,” arXiv preprint arXiv:2010.11060, 2020. [Online]. Available: https://arxiv.org/abs/2010.11060.

[5] Y. He, X. Chen, and J. McAuley, “Bridging language and items for retrieval and recommendation,” arXiv preprint arXiv:2403.03952, 2024. [Online]. Available: https://arxiv.org/abs/2403.03952.


## Business Understanding
### Problem Statements

1.	Pengguna e-commerce sering kesulitan menemukan produk yang sesuai dengan preferensi mereka karena jumlah produk yang sangat banyak dan beragam.

2.	Bagaimana cara memanfaatkan data rating pengguna untuk mengidentifikasi pola kesamaan antar produk (item) dan memberikan rekomendasi yang akurat?

### Goals

1.	Mengembangkan sistem rekomendasi berbasis collaborative filtering yang dapat membantu pengguna menemukan produk yang sesuai dengan preferensi mereka secara efisien di tengah banyaknya pilihan produk pada platform e-commerce.

2.	Memanfaatkan data rating pengguna untuk menganalisis pola kesamaan antar produk (item-based similarity) sehingga sistem dapat memberikan rekomendasi yang relevan dan akurat berdasarkan preferensi pengguna sebelumnya.

3.	Mengevaluasi kinerja model rekomendasi menggunakan metrik seperti Root Mean Square Error (RMSE) untuk memastikan akurasi dan keandalan hasil rekomendasi yang dihasilkan.

### Solution statements

 Untuk mencapai tujuan proyek ini, dilakukan serangkaian tahapan sistematis mulai dari pemahaman data hingga evaluasi model. Pendekatan ini menggabungkan metode item-based collaborative filtering dan model-based collaborative filtering (SVD) menggunakan pustaka Surprise di Python.Dua pendekatan utama diterapkan untuk membangun sistem rekomendasi:

a. KNN With Means (Item-Based Collaborative Filtering) Model ini menghitung kemiripan antar item (produk) berdasarkan pola rating dari pengguna.

b. Singular Value Decomposition (SVD)
•	Pendekatan model-based collaborative filtering yang memfaktorkan matriks rating pengguna–produk menjadi tiga komponen laten.

•	SVD mampu menangkap pola tersembunyi (latent features) antara pengguna dan produk.

## Data Understanding

Dataset yang digunakan dalam proyek ini merupakan data ulasan produk dari kategori Electronics di situs e-commerce Amazon, yang dapat diunduh melalui tautan berikut:
http://jmcauley.ucsd.edu/data/amazon/.

Dataset ini berisi data interaksi antara pengguna dan produk dalam bentuk rating yang diberikan oleh pengguna terhadap produk elektronik. Total terdapat 2522910 baris data dengan 4 kolom, yang masing-masing berisi informasi tentang pengguna, produk, nilai rating, dan waktu pemberian rating. Dataset memiliki ukuran sekitar 113 MB dan menggunakan dua tipe data, yaitu object dan float64.

Deskripsi Variabel
Berikut penjelasan dari masing-masing variabel yang terdapat pada dataset:

•	userId : Merupakan identitas unik dari setiap pengguna yang memberikan rating terhadap produk.

•	productId : Merupakan identitas unik dari setiap produk elektronik yang tersedia di platform e-commerce Amazon.

•	Rating : Nilai yang diberikan pengguna terhadap suatu produk dalam skala 1 hingga 5, di mana nilai yang lebih tinggi menunjukkan tingkat kepuasan yang lebih tinggi.

•	timestamp : Menunjukkan waktu (dalam format UNIX) ketika pengguna memberikan rating terhadap produk tersebut.

Analisis Awal Data (Exploratory Data Understanding)
Hasil eksplorasi awal menunjukkan bahwa:

•	Data terdiri dari jutaan interaksi pengguna-produk, menunjukkan skala besar dan tingkat sparsity yang tinggi (tidak semua pengguna memberi rating untuk semua produk).

•	Nilai rating berada dalam rentang 1 hingga 5, dengan distribusi yang condong ke arah rating tinggi (umumnya 4 atau 5), yang merupakan karakteristik umum dalam dataset ulasan e-commerce.

•	Kolom userId dan productId bersifat kategorikal dan akan diubah menjadi representasi numerik saat proses pelatihan model dilakukan.

1. Pemuatan Dataset

Dataset yang digunakan adalah Amazon Electronics Review Dataset, yang diunduh dari:
http://jmcauley.ucsd.edu/data/amazon/
Dataset dibaca menggunakan pustaka pandas dengan empat kolom utama, yaitu:

•	userId

•	productId

•	Rating

•	timestamp

df_1 = pd.read_csv('/content/ratings_Electronics (1).csv', 
                   names=['userId', 'productId', 'Rating', 'timestamp'])

2. Pemeriksaan Struktur Data

Langkah ini dilakukan untuk memahami tipe data, jumlah baris dan kolom, serta ketersediaan nilai pada setiap kolom.

df_1.info()

Hasil menunjukkan bahwa dataset memiliki 2522910 baris dan 4 kolom, dengan dua kolom bertipe object (userId, productId) dan dua kolom bertipe float64 (Rating, timestamp). ditemukan nilai missing.

3. Statistik Deskriptif dan Distribusi Rating

Analisis awal dilakukan untuk melihat karakteristik nilai rating, termasuk nilai minimum, maksimum, kuartil, serta distribusi rating.

df_1.describe()
sns.boxplot(x=df_1['Rating'])
sns.countplot(x='Rating', data=df_1)

Hasil analisis:

•	Nilai rating berkisar antara 1 hingga 5.

•	Distribusi rating cenderung tinggi (mayoritas pengguna memberi rating 4 atau 5).

•	Tidak ditemukan outlier ekstrem yang perlu dihapus.

Insight:
Distribusi ini menunjukkan bahwa pengguna cenderung memberikan ulasan positif terhadap produk elektronik di Amazon, yang merupakan pola umum dalam dataset e-commerce.

4. Pengecekan Nilai Kosong dan Duplikasi

Untuk memastikan integritas data, dilakukan pemeriksaan terhadap nilai kosong dan duplikasi.

df_1.isnull().sum()
df_1.duplicated().sum()

Hasil:
•	 terdapat nilai kosong pada dataset productid 1, rating 1, dan timestampt 1.

•	Tidak ditemukan data duplikat.

Alasan:

Pemeriksaan ini penting untuk mencegah kesalahan dalam perhitungan kesamaan produk dan menjaga kualitas hasil rekomendasi namun untuk nilai missing nya tidak extrem sehingga tidak perlu dilakukan penanganan.

## Data Preparation

Tahap ini dilakukan untuk mempersiapkan data agar siap digunakan dalam pembangunan sistem rekomendasi berbasis collaborative filtering. Proses ini meliputi beberapa langkah penting, mulai dari pemuatan data hingga pembersihan dan eksplorasi awal untuk memahami karakteristik dataset.

1. kolom timestamp dihapus karena tidak dibutuhkan dalam proses analisis. Penghapusan kolom tersebut dilakukan menggunakan perintah df_1.drop(['timestamp'], axis=1, inplace=True) agar DataFrame hanya berisi atribut yang relevan untuk pemodelan.

2. Dilakukan analisis terhadap aktivitas pengguna berdasarkan jumlah produk yang mereka beri rating. Penghitungan total rating per pengguna dilakukan dengan groupby('userId')['Rating'].count(). Hasilnya menunjukkan bahwa sebagian besar pengguna hanya memberikan 1 rating, dengan rata-rata 1,5 rating per user. Namun, terdapat beberapa pengguna yang sangat aktif dengan jumlah rating mencapai hingga 449 kali.
Lima pengguna paling aktif ditampilkan sebagai berikut:

userId	total_rated_items
A5JLAU2ARJ0BO	(449)
A6FIAB28IS79	(293)
A231WM2Z2JL0U3	(252)
A3OXHLG6DIBRW8	(252)
A680RUE1FDO8B	(203)

3. Persiapan untuk Model

reader = Reader(rating_scale=(1, 5))

data = Dataset.load_from_df(df_small, reader)

trainset, testset = train_test_split(data, test_size=0.3, random_state=10)

algo = KNNWithMeans(k=3, sim_options={'name': 'pearson_baseline', 'user_based': False})

algo.fit(trainset)

Data rating dikonversi ke format yang dapat diproses oleh library Surprise dengan mendefinisikan skala penilaian dari 1 hingga 5 menggunakan objek Reader. Selanjutnya, dataset dibagi menjadi dua bagian:

•	Training set (70%)
•	Testing set (30%)

Langkah ini memungkinkan evaluasi model yang adil antara data pelatihan dan data uji.
Pembentukan matriks user-item dari data rating pengguna terhadap produk.

## Modeling

1. Model 1 — Item-Based Collaborative Filtering (KNN With Means)
Pendekatan ini menggunakan metode memory-based collaborative filtering yang berfokus pada kemiripan antar produk (item-based similarity). Ide utamanya adalah bahwa produk yang mirip cenderung mendapatkan rating serupa dari pengguna.
Langkah-langkah utama:

2.	Perhitungan kemiripan antar item menggunakan metrik seperti cosine similarity.

3.	Prediksi rating dihitung berdasarkan rata-rata rating item serta kesamaan antar produk yang telah diberi rating oleh pengguna.
Kode Implementasi (ringkasan):

reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(df_small, reader)
trainset, testset = train_test_split(data, test_size=0.3, random_state=10)
algo = KNNWithMeans(k=3, sim_options={'name': 'pearson_baseline', 'user_based': False})
algo.fit(trainset)

4. Model 2 — Model-Based Collaborative Filtering (Singular Value Decomposition / SVD)
Pendekatan kedua menggunakan metode SVD, yang merupakan teknik matrix factorization untuk menemukan pola laten (tersembunyi) antara pengguna dan produk.
Tujuannya adalah memprediksi rating pengguna terhadap produk yang belum pernah dinilai dengan mempelajari hubungan tersembunyi antara keduanya.
Langkah-langkah utama:

5.	Matriks user-item diuraikan menjadi tiga komponen:

o	Matriks fitur pengguna

o	Matriks fitur item

o	Nilai singular yang merepresentasikan bobot interaksi antara keduanya

6.	Model dilatih untuk menemukan faktor laten tersebut.

7.	Prediksi dilakukan dengan merekonstruksi matriks berdasarkan representasi laten.
Kode Implementasi (ringkasan):

from sklearn.decomposition import TruncatedSVD
SVD = TruncatedSVD(n_components=10)
decomposed_matrix = SVD.fit_transform(X)

Alasan Pemilihan Algoritma:

•	SVD mampu menangani sparse data, yang umum ditemukan pada dataset rating (karena tidak semua pengguna memberi rating untuk semua produk).

•	Memberikan rekomendasi yang lebih personal dibanding metode berbasis memori.

•	Digunakan secara luas pada platform besar seperti Netflix dan Amazon.


8. Setelah model rekomendasi dibangun dan dievaluasi, dilakukan penentuan Top 25 Recommendation untuk setiap pengguna.

 Hasil ini menampilkan sejumlah n produk dengan nilai prediksi tertinggi berdasarkan preferensi pengguna. Untuk produk 00004WCI8, sistem merekomendasikan 25 produk teratas yaitu

 B00000J1V5',

 'B00000J4FS',

 'B00000JBHP',

 'B00000JDGQ',

 'B00001SIKD',

 'B0000300GE',

 'B00004RG6K',

 'B00004SYVH',

 'B00004TDAL',

 'B00004TDL2',

 'B00004TZGM',

 'B00004Z75J',

 'B00004ZCJE',

 'B000051123',

 'B000051249',

 'B00005A3LN',

 'B00005AC8W',

 'B00005MIS8',

 'B00005NIMN',

 'B00005QSRF',

 'B00005RG4N',

 'B00005UKXM',

 'B000063574',

 'B0000645RH'

## Evaluation

Untuk mengukur performa model rekomendasi, digunakan metrik Root Mean Squared Error (RMSE)Hasil Evaluasi dan Interpretasi
Dalam proyek ini, dua model collaborative filtering digunakan, yaitu:

1.	KNN With Means (Item-Based Collaborative Filtering)

2.	Singular Value Decomposition (SVD) (Model-Based Collaborative Filtering)

Hasil evaluasi terhadap kedua model menunjukkan perbandingan performa sebagai berikut:

Model	RMSE		Interpretasi

KNN With Means	1.4422	–	Model masih memiliki tingkat kesalahan yang relatif tinggi dalam memperkirakan rating pengguna.

SVD (TruncatedSVD)	0.0672	–	Model memiliki tingkat kesalahan yang sangat rendah, menunjukkan akurasi prediksi yang sangat baik.
