# Import library yang dibutuhkan untuk melakukan data preprocessing
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import streamlit as st
import librosa
import scipy.stats
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from streamlit_option_menu import option_menu
from scipy.stats import skew, kurtosis, mode, iqr

st.title("Prediksi Emosi dari Audio")

# Membaca data dari file csv
df = pd.read_csv('hasil_statistik_pertemuan4.csv')
# Memisahkan kolom target (label) dari kolom fitur
x = df.drop(columns=['Label'], axis =1)  # Kolom fitur
y = df['Label']  # Kolom target

st.write("# Ekstraksi Ciri Audio Untuk Klasifikasi Audio")

with st.sidebar:
  selected = option_menu(
      menu_title="Main Menu",
      options=["Dataset", "Normalisasi Data", "Hasil Akurasi", "Reduksi Data","Upload Audio"],
      default_index=0
  )

if selected == "Prediksi Audio":
    st.write("""
    <h1>Prediksi Data Audio </h1>
    <br>
    """, unsafe_allow_html=True)
    def calculate_statistics(audio_path):
        x, sr = librosa.load(audio_path)

        mean = np.mean(x)
        std = np.std(x)
        maxv = np.amax(x)
        minv = np.amin(x)
        median = np.median(x)
        skewness = skew(x)
        kurt = kurtosis(x)
        q1 = np.quantile(x, 0.25)
        q3 = np.quantile(x, 0.75)
        mode_v = mode(x)[0]
        iqr = q3 - q1

        zcr = librosa.feature.zero_crossing_rate(x)
        mean_zcr = np.mean(zcr)
        median_zcr = np.median(zcr)
        std_zcr = np.std(zcr)
        kurtosis_zcr = kurtosis(zcr, axis=None)
        skew_zcr = skew(zcr, axis=None)

        n = len(x)
        mean_rms = np.sqrt(np.mean(x**2) / n)
        median_rms = np.sqrt(np.median(x**2) / n)
        skew_rms = np.sqrt(skew(x**2) / n)
        kurtosis_rms = np.sqrt(kurtosis(x**2) / n)
        std_rms = np.sqrt(np.std(x**2) / n)

        return [mean, median, mode_v, maxv, minv, std, skewness, kurt, q1, q3, iqr, mean_zcr, median_zcr, std_zcr, kurtosis_zcr, skew_zcr, mean_rms, median_rms, std_rms, kurtosis_rms, skew_rms]

    uploaded_file = st.file_uploader("Pilih file audio...", type=["wav","mp3"])

    scaler = st.radio(
    "Prediksi Class Data Audio",
    ('Prediksi Z-Score', 'Prediksi MinMax'))

    if uploaded_file is not None:
        st.audio(uploaded_file, format="audio/wav")
        if scaler == 'Prediksi Z-Score':
            st.title("Prediksi Class Data Audio Menggunakan Z-Score")

            if st.button("Cek Nilai Statistik"):
                # Simpan file audio yang diunggah
                audio_path = "audio_diunggah.wav"
                with open(audio_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                # Hitung statistik untuk file audio yang diunggah
                statistik = calculate_statistics(audio_path)

                results = []
                result = {
                    'Audio Mean': statistik[0],
                    'Audio Median': statistik[1],
                    'Audio Mode': statistik[2],
                    'Audio Maxv': statistik[3],
                    'Audio Minv': statistik[4],
                    'Audio Std': statistik[5],
                    'Audio Skew': statistik[6],
                    'Audio Kurtosis': statistik[7],
                    'Audio Q1': statistik[8],
                    'Audio Q3': statistik[9],
                    'Audio IQR': statistik[10],
                    'ZCR Mean': statistik[11],
                    'ZCR Median': statistik[12],
                    'ZCR Std': statistik[13],
                    'ZCR Kurtosis': statistik[14],
                    'ZCR Skew': statistik[15],
                    'RMS Energi Mean': statistik[16],
                    'RMS Energi Median': statistik[17],
                    'RMS Energi Std': statistik[18],
                    'RMS Energi Kurtosis': statistik[19],
                    'RMS Energi Skew': statistik[20],
                }
                results.append(result)
                df = pd.DataFrame(results)
                st.write(df)

                # Hapus file audio yang diunggah
                os.remove(audio_path)

            if st.button("Deteksi Audio"):

                # Memuat data audio yang diunggah dan menyimpannya sebagai file audio
                audio_path = "audio_diunggah.wav"
                with open(audio_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                # Menghitung statistik untuk file audio yang diunggah (gunakan fungsi calculate_statistics sesuai kebutuhan)
                audio_features = calculate_statistics(audio_path)
                results = []
                result = {
                    'Audio Mean': audio_features[0],
                    'Audio Median': audio_features[1],
                    'Audio Mode': audio_features[2],
                    'Audio Maxv': audio_features[3],
                    'Audio Minv': audio_features[4],
                    'Audio Std': audio_features[5],
                    'Audio Skew': audio_features[6],
                    'Audio Kurtosis': audio_features[7],
                    'Audio Q1': audio_features[8],
                    'Audio Q3': audio_features[9],
                    'Audio IQR': audio_features[10],
                    'ZCR Mean': audio_features[11],
                    'ZCR Median': audio_features[12],
                    'ZCR Std': audio_features[13],
                    'ZCR Kurtosis': audio_features[14],
                    'ZCR Skew': audio_features[15],
                    'RMS Energi Mean': audio_features[16],
                    'RMS Energi Median': audio_features[17],
                    'RMS Energi Std': audio_features[18],
                    'RMS Energi Kurtosis': audio_features[19],
                    'RMS Energi Skew': audio_features[20],
                }
                results.append(result)
                data_tes = pd.DataFrame(results)


                # Load the model and hyperparameters
                with open('gridsearchknnzscoremodel.pkl', 'rb') as model_file:
                    saved_data = pickle.load(model_file)

                df = pd.read_csv('hasil_statistik_pertemuan4.csv')

                # Memisahkan kolom target (label) dari kolom fitur
                X = df.drop(columns=['Label'])  # Kolom fitur
                y = df['Label']  # Kolom target

                # Normalisasi data menggunakan StandardScaler
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)

                # Memisahkan data menjadi data latih dan data uji
                X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

                # Access hyperparameters
                best_n_neighbors = saved_data['hyperparameters']['best_n_neighbors']
                best_weights = saved_data['hyperparameters']['best_weights']
                best_metric = saved_data['hyperparameters']['best_metric']
                best_comp = saved_data['hyperparameters']['best_comp']

                # Melakukan PCA pada data audio yang diunggah
                pca = PCA(n_components=best_comp)

                # Memanggil metode fit dengan data pelatihan sebelum menggunakan transform
                zscore_scaler = StandardScaler()
                X_test_zscore = zscore_scaler.fit_transform(data_tes)

                X_train_pca = pca.fit_transform(X_train)
                X_test_pca = pca.transform(X_test_zscore)

                # Membuat model KNN dengan hyperparameter terbaik
                best_knn_model = KNeighborsClassifier(n_neighbors=best_n_neighbors, weights=best_weights, metric=best_metric)
                best_knn_model.fit(X_train_pca, y_train)

                predicted_label = best_knn_model.predict(X_test_pca)

                # Menampilkan hasil prediksi
                st.write("Emosi Terdeteksi:", predicted_label)

                # Menghapus file audio yang diunggah
                os.remove(audio_path)

        elif scaler == 'Prediksi MinMax':
            st.title("Prediksi Class Data Audio Menggunakan MinMax")

            if st.button("Cek Nilai Statistik"):
                # Simpan file audio yang diunggah
                audio_path = "audio_diunggah.wav"
                with open(audio_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                # Hitung statistik untuk file audio yang diunggah
                statistik = calculate_statistics(audio_path)

                results = []
                result = {
                    'Audio Mean': statistik[0],
                    'Audio Median': statistik[1],
                    'Audio Mode': statistik[2],
                    'Audio Maxv': statistik[3],
                    'Audio Minv': statistik[4],
                    'Audio Std': statistik[5],
                    'Audio Skew': statistik[6],
                    'Audio Kurtosis': statistik[7],
                    'Audio Q1': statistik[8],
                    'Audio Q3': statistik[9],
                    'Audio IQR': statistik[10],
                    'ZCR Mean': statistik[11],
                    'ZCR Median': statistik[12],
                    'ZCR Std': statistik[13],
                    'ZCR Kurtosis': statistik[14],
                    'ZCR Skew': statistik[15],
                    'RMS Energi Mean': statistik[16],
                    'RMS Energi Median': statistik[17],
                    'RMS Energi Std': statistik[18],
                    'RMS Energi Kurtosis': statistik[19],
                    'RMS Energi Skew': statistik[20],
                }
                results.append(result)
                df = pd.DataFrame(results)
                st.write(df)

                # Hapus file audio yang diunggah
                os.remove(audio_path)

            if st.button("Deteksi Audio"):

                # Memuat data audio yang diunggah dan menyimpannya sebagai file audio
                audio_path = "audio_diunggah.wav"
                with open(audio_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                # Menghitung statistik untuk file audio yang diunggah (gunakan fungsi calculate_statistics sesuai kebutuhan)
                audio_features = calculate_statistics(audio_path)
                results = []
                result = {
                    'Audio Mean': audio_features[0],
                    'Audio Median': audio_features[1],
                    'Audio Mode': audio_features[2],
                    'Audio Maxv': audio_features[3],
                    'Audio Minv': audio_features[4],
                    'Audio Std': audio_features[5],
                    'Audio Skew': audio_features[6],
                    'Audio Kurtosis': audio_features[7],
                    'Audio Q1': audio_features[8],
                    'Audio Q3': audio_features[9],
                    'Audio IQR': audio_features[10],
                    'ZCR Mean': audio_features[11],
                    'ZCR Median': audio_features[12],
                    'ZCR Std': audio_features[13],
                    'ZCR Kurtosis': audio_features[14],
                    'ZCR Skew': audio_features[15],
                    'RMS Energi Mean': audio_features[16],
                    'RMS Energi Median': audio_features[17],
                    'RMS Energi Std': audio_features[18],
                    'RMS Energi Kurtosis': audio_features[19],
                    'RMS Energi Skew': audio_features[20],
                }
                results.append(result)
                data_tes = pd.DataFrame(results)


                # Load the model and hyperparameters
                with open('gridsearchknnminmaxmodel.pkl', 'rb') as model_file:
                    saved_data = pickle.load(model_file)

                df = pd.read_csv('hasil_statistik_pertemuan4.csv')

                # Memisahkan kolom target (label) dari kolom fitur
                X = df.drop(columns=['Label'])  # Kolom fitur
                y = df['Label']  # Kolom target

                # Normalisasi data menggunakan StandardScaler
                scaler = MinMaxScaler()
                x_scaled = scaler.fit_transform(X)

                # Memisahkan data menjadi data latih dan data uji
                X_train, X_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=42)

                # Access hyperparameters
                best_n_neighbors = saved_data['hyperparameters']['best_n_neighbors']
                best_weights = saved_data['hyperparameters']['best_weights']
                best_metric = saved_data['hyperparameters']['best_metric']
                best_comp = saved_data['hyperparameters']['best_comp']

                # Melakukan PCA pada data audio yang diunggah
                pca = PCA(n_components=best_comp)

                # Memanggil metode fit dengan data pelatihan sebelum menggunakan transform
                minmax_scaler = MinMaxScaler()
                X_test_minmax = minmax_scaler.fit_transform(data_tes)

                X_train_pca = pca.fit_transform(X_train)
                X_test_pca = pca.transform(X_test_minmax)

                # Membuat model KNN dengan hyperparameter terbaik
                best_knn_model = KNeighborsClassifier(n_neighbors=best_n_neighbors, weights=best_weights, metric=best_metric)
                best_knn_model.fit(X_train_pca, y_train)

                predicted_label = best_knn_model.predict(X_test_pca)

                # Menampilkan hasil prediksi
                st.write("Emosi Terdeteksi:", predicted_label)


                # Menghapus file audio yang diunggah
                os.remove(audio_path)

if selected == "Dataset":
    st.write('''## Dataset''')
    st.write(df)
    st.write('''Dataset ini merupakan hasil Ekstraksi Ciri Audio yang mana audio yang digunakan berasal dari website Kaggle.''')
    st.write('''Dataset ini memiliki jumlah data sebanyak 2800 dengan 22 fitur.''')
    st.write('''#### Fitur-Fitur Pada Dataset''')
    st.info('''
    Fitur yang akan digunakan adalah sebagai berikut :
    1. Mean Audio
    2. Median Audio
    3. Modus Audio
    4. Nilai maksimum Audio
    5. Nilai minimum Audio
    6. Standar deviasi Audio
    7. Nilai kemiringan (skewness) Audio
    8. Nilai Keruncingan (kurtosis) Audio
    9. Nilai kuartil bawah (Q1) Audio
    10. Nilai kuartil atas (Q3) Audio
    11. Nilai IQR Audio
    12. Mean ZCR
    13. Median ZCR
    14. Std ZCR
    15. Kurtosis ZCR
    16. Skew ZCR
    17. Mean Energy RMSE
    18. Median Energy RMSE
    19. Kurtosis Energy RMSE
    20. Std Energy RMSE
    21. Skew Energy RMSE
    22. Label
     ''')

if selected == "Normalisasi Data":

  # Membagi data menjadi data training dan data testing
  x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

  st.write('''## Membagi Data Menjadi Data Uji Dan Data Testing''')
  st.write('Data dibagi menjadi 30% sebagai data uji dan 70% data testing')
  st.success(f'''
  ##### Diperoleh:
  - Banyaknya Data : {x.shape[0]}
  - Banyak Data Testing : {x_train.shape[0]}
  - Banyak Data Uji : {x_test.shape[0]}
  - Banyaknya fitur yang digunakan : {x.shape[1]}
  ''')

  st.write('''## Normalisasi Menggunakan Z-Score''')
  st.write('''Normalisasi Z-Score merupakan teknik yang mana nilai pada atribut akan dinormalisasikan berdasarkan mean dan standar deviasi.
  Normalisasi Z-Score mentransformasikan data dari nilai ke skala umum dimana mean sama dengan nol dan standar deviasi adalah satu.''')
  st.write('Untuk melakukan normalisasi Z-Score pada python bisa menggunakan Library StandardScaler().')

  # Membagi data menjadi data training dan data testing
  x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


  #membuat variabel untuk normalisasi menggunakan z-score dan minmax
  z_scaler = StandardScaler()
  zscore_x_train = z_scaler.fit_transform(x_train)
  zscore_x_test = z_scaler.fit_transform(x_test)

  # Simpan normalisasi zscore ke dalam file
  pickle.dump(zscore_x_train, open('z_score_train.pkl','wb'))
  pickle.dump(zscore_x_test, open('z_score_test.pkl','wb'))

  #Membaca dataframe hasil dari normalisasi zscore Scalling sebelumnya
  pickled_z_score_train = pickle.load(open('z_score_train.pkl','rb'))
  pickled_z_score_test = pickle.load(open('z_score_test.pkl','rb'))

  st.write("#### Hasil Normalisasi Z-Score pada X-train ")
  st.write(pickled_z_score_train)

  st.write("#### Hasil Normalisasi Z-Score pada X-test ")
  st.write(pickled_z_score_test)

  st.write('''## Normalisasi Menggunakan MinMaxScaler''')
  st.write('''MinMaxScaler merupakan teknik yang digunakan untuk mengubah nilai-niali kedalam suatu fitur atau kolom menjadi rentang tertentu,
  biasanya dalam rentang 0 hingga 1. Rentang skala pada MinMaxScaler bisa diatur sesuai yang diinginkan. ''')
  st.write('Untuk melakukan normalisasi Z-Score pada python bisa menggunakan Library MinMaxScaler().')

  #membuat variabel untuk normalisasi menggunakan minmax
  minmax_scaler = MinMaxScaler()
  minmax_x_train = minmax_scaler.fit_transform(x_train)
  minmax_x_test = minmax_scaler.fit_transform(x_test)

  # Simpan normalisasi minmax ke dalam file
  pickle.dump(minmax_x_train, open('minmax_x_train,pkl','wb'))
  pickle.dump(minmax_x_test, open('minmax_x_test.pkl','wb'))

  #Membaca dataframe hasil dari normalisasi minmax Scalling sebelumnya
  pickled_min_max_train = pickle.load(open('min_max_train.pkl','rb'))
  pickled_min_max_test = pickle.load(open('min_max_test.pkl','rb'))

  st.write("#### Hasil Normalisasi MinMaxScaler pada X-train ")
  st.write(pickled_min_max_train)

  st.write("#### Hasil Normalisasi MinMaxScaler pada X-test ")
  st.write(pickled_min_max_test)

if selected == "Hasil Akurasi":
  st.write("## Hasil Akurasi Dari Normalisasi Z-Score Dengan Model KNN")

  # Membagi data menjadi data training dan data testing
  x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

  #Membaca dataframe hasil dari normalisasi zscore Scalling sebelumnya
  pickled_z_score_train = pickle.load(open('z_score_train.pkl','rb'))
  pickled_z_score_test = pickle.load(open('z_score_test.pkl','rb'))

  akurasi_tertinggi = 0
  k_terbaik = []

  for k in list(range(1, 51)):

      # membangun model KNN
      knn = KNeighborsClassifier(n_neighbors = k)
      knn.fit(pickled_z_score_train, y_train)
      y_pred_knn = knn.predict(pickled_z_score_test )

      # akurasi
      akurasi_knn = accuracy_score(y_test, y_pred_knn)
      st.write(f"Hasil akurasi dengan k = {k} : {akurasi_knn}")

      if akurasi_knn > akurasi_tertinggi:
          akurasi_tertinggi = akurasi_knn
          k_terbaik = [k]
      elif akurasi_knn == akurasi_tertinggi:
          k_terbaik.append(k)

  st.success(f"Hasil akurasi tertinggi adalah {akurasi_tertinggi} pada k = {k_terbaik}")


  st.write("## Hasil Akurasi Dari Normalisasi MinMaxScaler Dengan Model KNN")

  #Membaca dataframe hasil dari normalisasi minmax Scalling sebelumnya
  pickled_min_max_train = pickle.load(open('min_max_train.pkl','rb'))
  pickled_min_max_test = pickle.load(open('min_max_test.pkl','rb'))

  akurasi_tertinggi = 0
  k_terbaik = []

  for k in list(range(1, 51)):

      # membangun model KNN
      knn = KNeighborsClassifier(n_neighbors = k)
      knn.fit(pickled_min_max_train, y_train)
      y_pred_knn = knn.predict(pickled_min_max_test)

      # akurasi
      akurasi_knn = accuracy_score(y_test, y_pred_knn)
      st.write(f"Hasil akurasi dengan k = {k} : {akurasi_knn}")

      if akurasi_knn > akurasi_tertinggi:
          akurasi_tertinggi = akurasi_knn
          k_terbaik = [k]
      elif akurasi_knn == akurasi_tertinggi:
          k_terbaik.append(k)

  st.success(f"Hasil akurasi tertinggi adalah {akurasi_tertinggi} pada k = {k_terbaik}")

if selected == "Reduksi Data":
  st.write("## Mereduksi Data Berdasarkan Pencarian Parameter K- Manual ")

  # Membagi data menjadi data training dan data testing
  x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

  #Membaca dataframe hasil dari normalisasi zscore Scalling sebelumnya
  pickled_z_score_train = pickle.load(open('z_score_train.pkl','rb'))
  pickled_z_score_test = pickle.load(open('z_score_test.pkl','rb'))

  #Membaca dataframe hasil dari normalisasi minmax Scalling sebelumnya
  pickled_min_max_train = pickle.load(open('min_max_train.pkl','rb'))
  pickled_min_max_test = pickle.load(open('min_max_test.pkl','rb'))

  pca = PCA(n_components = 20)
  pca_1 = pca.fit(pickled_z_score_train)
  pca_2 = pca.fit(pickled_min_max_train)

  # Menggunakan data normalisasi zscore
  pca_train_zscore = pca_1.transform(pickled_z_score_train)
  pca_test_zscore = pca_1.transform(pickled_z_score_test)

  # Menggunakan data normalisasi minmaxscaler
  pca_train_minmax = pca_2.transform(pickled_min_max_train)
  pca_test_minmax = pca_2.transform(pickled_min_max_test)

  st.write("### Reduksi Data Menggunakan Data Hasil Normalisasi Z-Score")
  st.info("Dikarenakan pada hasil akurasi dari normalisasi z-score dengan model KNN mendapat akurasi terbaik pada k-11, maka pada model PCA ini untuk nilai k diisi dengan 11")


  #z-score

  akurasi_tertinggi = 0
  k_terbaik = []

  # List untuk menyimpan nilai akurasi
  akurasi_list = []

  for k in list(range(1, 51)):

      # membangun model KNN
      knn = KNeighborsClassifier(n_neighbors = k)
      knn.fit(pca_train_zscore, y_train)
      y_pred_knn = knn.predict(pca_test_zscore)


      # akurasi
      akurasi_knn = accuracy_score(y_test, y_pred_knn)
      akurasi_list.append(akurasi_knn)

      st.write(f"Hasil akurasi dengan k = {k} : {akurasi_knn}")

      if akurasi_knn > akurasi_tertinggi:
          akurasi_tertinggi = akurasi_knn
          k_terbaik = [k]
      elif akurasi_knn == akurasi_tertinggi:
          k_terbaik.append(k)

  st.success(f"Hasil akurasi tertinggi adalah {akurasi_tertinggi} pada k = {k_terbaik}")



  akurasi_list1 = []

  for n_components in range(20, 0, -1):

      # Membangun model PCA dengan jumlah komponen utama yang sesuai
      pca = PCA(n_components = n_components)
      pca.fit(pickled_z_score_train)

      pca_train = pca.transform(pickled_z_score_train)
      pca_test = pca.transform(pickled_z_score_test)

      # Sesuai pencarian nilai k- terbaik sebelumnya, maka digunakan k = 9

      knn = KNeighborsClassifier(n_neighbors = 9)
      knn.fit(pca_train, y_train)
      y_pred_knn = knn.predict(pca_test)

      # Menghitung akurasi
      akurasi = accuracy_score(y_test, y_pred_knn)

      akurasi_list1.append(akurasi)

      st.write(f"Jumlah komponen utama  {n_components}, dengan akurasi  {akurasi}")

  st.write("#### Tampilan Dalam Grafik ")

  # Membuat grafik

  plt.figure(figsize=(16, 5))
  plt.plot(list(range(1, 21)), akurasi_list1, marker='o', linestyle='-', color='green')
  plt.title('Grafik Akurasi Normalisasi Z-score - PCA')
  plt.xlabel('Nilai k')
  plt.ylabel('Akurasi')
  plt.grid(True)
  plt.xticks(list(range(1, 21)))

  plt.savefig('normalisasi_zscore.png')

  st.pyplot(plt)

  st.write("### Reduksi Data Menggunakan Data Hasil Normalisasi MinMaxScaler")
  st.info("Dikarenakan pada hasil akurasi dari normalisasi minmaxscaler dengan model KNN mendapat akurasi terbaik pada k-8, maka pada model PCA ini untuk nilai k diisi dengan 8")


  #minmax

  akurasi_tertinggi = 0
  k_terbaik = []

  # List untuk menyimpan nilai akurasi
  akurasi_list = []

  for k in list(range(1, 51)):

      # membangun model KNN
      knn = KNeighborsClassifier(n_neighbors = k)
      knn.fit(pca_train_minmax, y_train)
      y_pred_knn = knn.predict(pca_test_minmax)


      # akurasi
      akurasi_knn = accuracy_score(y_test, y_pred_knn)
      akurasi_list.append(akurasi_knn)

      st.write(f"Hasil akurasi dengan k = {k}: {akurasi_knn}")

      if akurasi_knn > akurasi_tertinggi:
          akurasi_tertinggi = akurasi_knn
          k_terbaik = [k]
      elif akurasi_knn == akurasi_tertinggi:
          k_terbaik.append(k)

  st.success(f"Hasil akurasi tertinggi adalah {akurasi_tertinggi} pada k = {k_terbaik}")


  akurasi_list2 = []

  for n_components in range(20, 0, -1):

      # Membangun model PCA dengan jumlah komponen utama yang sesuai
      pca = PCA(n_components = n_components)
      pca.fit(pickled_min_max_train)

      pca_train = pca.transform(pickled_min_max_train)
      pca_test = pca.transform(pickled_min_max_test)

      # Sesuai pencarian nilai k- terbaik sebelumnya, maka digunakan k = 5
      knn = KNeighborsClassifier(n_neighbors = 5)
      knn.fit(pca_train, y_train)
      y_pred_knn = knn.predict(pca_test)

      # Menghitung akurasi
      akurasi = accuracy_score(y_test, y_pred_knn)

      akurasi_list2.append(akurasi)

      st.write(f"Jumlah komponen utama  {n_components}, dengan akurasi  {akurasi}")

  st.write("#### Tampilan Dalam Grafik ")

  # Menampilkan grafik

  plt.figure(figsize=(16, 5))
  plt.plot(range(0,20), akurasi_list2, marker='p', linestyle='-', color='blue')
  plt.title('Grafik Akurasi Normalisasi Min Max - PCA')
  plt.xlabel('Jumlah komponen PCA')
  plt.ylabel('Akurasi')
  plt.grid(True)
  plt.xticks(range(0,20))

  plt.savefig('normalisasi_minmax.png')

  st.pyplot(plt)


  st.write("#### grid ")
  # Mendefinisikan parameter yang ingin diuji
  param_grid = {
      'n_neighbors': list(range(1, 51)),  # untuk nilai k-
      'weights': ['uniform', 'distance'],  # pengaturan cara bobot jarak antara tetangga-tetangga dalam proses prediksi
      'metric': ['euclidean', 'manhattan']  # metode pengukuran jarak antara titik data
  }

  grid_search_zscore = GridSearchCV(estimator = knn, param_grid = param_grid)
  grid_search_zscore.fit(pca_train_zscore, y_train)

  grid_search_minmaxscaler = GridSearchCV(estimator = knn, param_grid = param_grid)
  grid_search_minmaxscaler.fit(pca_train_minmax, y_train)

  # Menampilkan parameter dan akurasi terbaik dari data
  st.write(f"Parameter terbaik untuk data normalisasi zscore: {grid_search_zscore.best_params_}")
  st.write(f"Akurasi terbaik untuk data normalisasi zscore :  {grid_search_zscore.best_score_}")

  # Menampilkan akurasi terbaik
  st.write(f"Parameter terbaik untuk data normalisasi minmaxscaler: {grid_search_minmaxscaler.best_params_}")
  st.write(f"Akurasi terbaik untuk data normalisasi minmaxscaler : {grid_search_minmaxscaler.best_score_}")



  akurasi_list3 = []

  for n_components in range(20, 0, -1):

      # Membangun model PCA dengan jumlah komponen utama yang sesuai
      pca = PCA(n_components = n_components)
      pca.fit(pickled_min_max_train)

      pca_train = pca.transform(pickled_min_max_train)
      pca_test = pca.transform(pickled_min_max_test)

      # Sesuai hasil grid search sebelumnya, {'metric': 'euclidean', 'n_neighbors': 6, 'weights': 'distance'}
      knn = KNeighborsClassifier(n_neighbors = 6, metric = 'euclidean', weights = 'distance')
      knn.fit(pca_train, y_train)
      y_pred_knn = knn.predict(pca_test)

      # Menghitung akurasi
      akurasi = accuracy_score(y_test, y_pred_knn)

      akurasi_list3.append(akurasi)

      st.write(f"Jumlah komponen utama  {n_components}, dengan akurasi  {akurasi}")


  st.write("#### Tampilan Dalam Grafik ")

  plt.figure(figsize=(16, 5))
  plt.plot(range(0,20), akurasi_list3, marker='o', linestyle='-', color='blue')
  plt.title('Grafik Akurasi Normalisasi Min Max - grid')
  plt.xlabel('Jumlah komponen PCA')
  plt.ylabel('Akurasi')
  plt.grid(True)
  plt.xticks(range(0,20))

  plt.savefig('grid_normalisasi_minmax.png')

  st.pyplot(plt)