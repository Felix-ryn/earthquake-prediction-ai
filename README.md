# Dokumentasi Kode Sistem *Earthquake Prediction*

Dokumen ini menjelaskan struktur dan isi kode pada proyek **earthquake-prediction** secara rinci per file / modul:

- alur data dari awal sampai prediksi ROI,
- peran masing-masing modul,
- bagaimana backend dan web visualisasi saling terhubung.

---

## 1. Struktur Direktori Proyek

Secara garis besar, isi zip:

```text
earthquake-prediction/
├─ main.py
├─ .env                # opsional, konfigurasi environment
├─ src/
│  ├─ __init__.py
│  ├─ config.py
│  ├─ utils.py
│  ├─ pipeline.py
│  ├─ models/
│  │  ├─ cnn.py
│  │  └─ lstm.py
│  └─ modules/
│     ├─ ingest.py
│     ├─ aco.py
│     ├─ ga.py
│     ├─ train.py
│     ├─ eval.py
│     └─ predict.py
└─ web/
   ├─ index.html
   └─ data/
      ├─ events_aco.geojson
      ├─ ga_vectors.geojson
      └─ inference_roi.geojson
```

Alurnya:

1. `` menjalankan pipeline end-to-end.
2. `` mengatur urutan langkah (ingest → ACO → GA → train → eval → predict).
3. `` menyimpan konfigurasi (path I/O, hyperparameter ACO/GA/train/predict) dan membangunnya dari environment / `.env`.
4. `` berisi step-step pipeline yang konkret.
5. `` berisi arsitektur model deep learning (CNN, LSTM, hybrid).
6. `` membaca hasil GeoJSON dan menampilkan peta interaktif dengan Leaflet.

---

## 2. Root: `main.py`

```python
from src.config import build_pipeline_from_env
from src.pipeline import EarthquakePipeline

pipe_cfg = build_pipeline_from_env()
pipeline = EarthquakePipeline(pipe_cfg)

result = pipeline.run_all()
```

Fungsinya sangat sederhana:

1. `` (dari `config.py`) membaca `.env` / environment untuk menentukan:

   - lokasi file input tectonic & volcanic,
   - lokasi output (parquet, model, geojson),
   - parameter ACO, GA, training, dll.

2. Membuat instance `` dengan konfigurasi tersebut.

3. Memanggil `` untuk menjalankan seluruh rangkaian langkah:

   - ingest → aco → ga → train → eval → predict.

`result` adalah dict ringkasan hasil (path model, metrik NB, ringkasan prediksi ROI).

---

## 3. Utilitas: `src/utils.py`

File ini menyediakan helper logging standar melalui fungsi:

```python
def setup_logger(name: Optional[str] = "app", level: str = "INFO") -> logging.Logger:
    ...
```

Hal yang dilakukan:

- Membuat **logger** dengan nama modul (misalnya `src.modules.train`).
- Men-set level (INFO, DEBUG, dll.).
- Menambahkan **handler console** (output ke terminal) dengan format waktu + level + nama logger + pesan.
- Menambahkan **RotatingFileHandler** yang menulis log ke berkas di folder `logs/`:
  - `logs/src.modules.train.log`, `logs/src.modules.aco.log`, dll.
  - File log memiliki batas ukuran (mis. 5 MB) dan rolling backup.
- `logger.propagate = False` supaya log tidak dobel ke root-logger.

Semua modul penting (`aco.py`, `ga.py`, `train.py`, dll.) memanggil ini untuk konsisten.

---

## 4. Konfigurasi: `src/config.py`

File ini adalah **jantung konfigurasi**. Di dalamnya ada:

### 4.1. Helper environment

- ``

  - Baca environment variable bernama `key`.
  - Jika tidak ada → pakai `default`.
  - Mendukung beberapa tipe:
    - jika default list → bisa parse string JSON `["a", "b"]` atau string koma `a,b`.
    - jika default bool / int / float → dicasting sesuai tipe.

- ``

  - Kalau ada file `.env` di root, dibaca dan di-`os.environ[...]`-kan.
  - Formatnya kunci=nilai per baris.

Dengan ini, kamu bisa mengontrol pipeline via `.env` tanpa ubah kode.

### 4.2. Dataclass Konfigurasi Per Bagian

Beberapa dataclass penting (nama bisa sedikit berbeda, tapi konsepnya seperti ini):

- ``

  - `tectonic_path: Path` – path CSV/parquet data gempa tektonik.
  - `volcanic_path: Path` – path CSV/parquet data aktivitas vulkanik.
  - `output_path: Path` – path output gabungan (`outputs/events.parquet`).

- ``

  - Parameter ACO:
    - `n_ants`, `n_iters` – jumlah semut dan iterasi.
    - `neighbor_k` – jumlah tetangga spatio-temporal per event.
    - `alpha`, `beta` – bobot pheromone vs heuristik edge.
    - `rho` – laju evaporasi pheromone.
    - `walk_length` – panjang langkah semut.
    - `seed` – random seed.
  - Output:
    - `output_path` – parquet hasil ACO.
    - `geojson_path` – GeoJSON titik ACO.

- ``

  - Parameter GA untuk *orientasi klaster*:
    - `freq` – frekuensi grouping (mis: "Y" per tahun, "M" per bulan).
    - `min_events` – minimum event per grup.
    - `pop_size`, `n_generations`, `elite_frac`, `mutation_std_deg` – param GA populasi & mutasi.
    - `seed` – random seed.
    - `geojson_path` – GeoJSON vektor GA.

- ``

  - `events_path: Path` – biasanya `outputs/events_aco_ga.parquet`.
  - `model_dir: Path` – folder `models/`.
  - `seq_len: int` – panjang sequence deret waktu.
  - Hyperparameter training:
    - `batch_size`, `learning_rate`, `epochs_cnn`, `epochs_lstm`.
    - `test_size`, `random_state`.

- ``

  - `events_path: Path` – sama seperti Train.
  - `metrics_path: Path` – path `outputs/metrics_nb.json`.
  - `test_size`, `random_state`, `stratify`.

- ``

  - `events_path: Path` – lagi-lagi hasil ACO+GA.
  - `model_dir: Path` – ambil LSTM dan scaler.
  - `seq_len` – harus cocok dengan training.
  - `roi_path`, `roi_geojson_path` – output ROI (CSV opsional, GeoJSON utama).

- ``

  - Berisi semua path I/O standar:
    - `events_path`
    - `events_aco_path`, `events_aco_geojson_path`
    - `events_aco_ga_path`
    - `nb_metrics_path`
    - `roi_path`, `roi_geojson_path`
    - `models_dir`

- ``

  - `io: IOConfig`
  - `aco: ACOConfig`
  - `ga: GAConfig`
  - `train: TrainConfig | None`
  - `eval: EvalConfig | None`
  - `predict: PredictConfig | None`
  - `seq_len: int`

  Serta method helper:

  - `build_ingest_config()` → `IngestConfig`.
  - `build_aco_config()`   → `ACOConfig` (dengan output\_path diset benar).
  - `build_ga_config()`    → `GAConfig`.
  - `build_train_config()` → `TrainConfig`.
  - `build_eval_config()`  → `EvalConfig`.
  - `build_predict_config()` → `PredictConfig`.

### 4.3. Builder dari Environment

Ada beberapa fungsi semacam:

- `build_aco_from_env()`
- `build_ga_from_env()`
- `build_train_from_env()`
- ``

`build_pipeline_from_env()` akan:

1. Memanggil `_load_env_file()`.
2. Membaca semua env seperti `IO_EVENTS_PATH`, `ACO_N_ANTS`, dll.
3. Membentuk `IOConfig`, lalu `PipelineConfig` yang lengkap.

Inilah yang dipakai `main.py`.

---

## 5. Orkestrator: `src/pipeline.py`

File ini mendefinisikan kelas `` yang mengatur alur end-to-end.

### 5.1. Struktur kelas

```python
class EarthquakePipeline:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.events: Optional[pd.DataFrame] = None
        self.events_aco: Optional[pd.DataFrame] = None
        self.events_aco_ga: Optional[pd.DataFrame] = None
```

Ada beberapa method utama:

- `_save_df(df, path)`

  - Helper: simpan DataFrame ke Parquet (`fastparquet`), fallback ke CSV kalau gagal.

- ``

  - Bangun `IngestConfig` dari `self.config.build_ingest_config()`.
  - Buat `EventIngestor` dan panggil `.run()`.
  - Simpan hasil ke `self.events` dan ke disk (`outputs/events.parquet`).

- ``

  - Pastikan `self.events` sudah ada, kalau belum baca dari file.
  - Set `ACOConfig.output_path` dan `ACOConfig.geojson_path` dari `IOConfig`.
  - Buat `ACOStep` dan panggil `.run(events)`.
  - Simpan ke `self.events_aco` dan ke disk (`events_aco.parquet`).

- ``

  - Pastikan `self.events_aco` ada, kalau belum baca dari file.
  - Set `GAConfig.geojson_path` dari `IOConfig`.
  - Buat `GAStep` dan panggil `.run(events_aco)`.
  - Simpan ke `self.events_aco_ga` dan ke disk (`events_aco_ga.parquet`).

- ``

  - Bangun `TrainConfig` dari pipeline-config.
  - Buat `Trainer` dan panggil `.train()`.
  - Menghasilkan dict berisi path model CNN, LSTM, dan scaler.

- ``

  - Bangun `EvalConfig`.
  - Buat `DirectionEvaluator` dan jalankan `.run()`.
  - Menghasilkan dict metrik NB.

- ``

  - Bangun `PredictConfig`.
  - Buat `Predictor` dan panggil `.run()`.
  - Menghasilkan dict ringkasan ROI.

- ``

```python
    def run_all(self) -> dict:
        logger.info("=== PIPELINE: INGEST ===")
        self.run_ingest()
        logger.info("=== PIPELINE: ACO ===")
        self.run_aco()
        logger.info("=== PIPELINE: GA / movement ===")
        self.run_ga()
        logger.info("=== PIPELINE: TRAIN (CNN+LSTM) ===")
        train_info = self.run_train()
        logger.info("=== PIPELINE: EVAL (Naive Bayes) ===")
        eval_metrics = self.run_eval()
        logger.info("=== PIPELINE: PREDICT (ROI) ===")
        pred_info = self.run_predict()

        return {
            "train": train_info,
            "eval": eval_metrics,
            "predict": pred_info,
        }
```

Method ini yang dipanggil di `main.py`.

---

## 6. Modul-modul Pipeline: `src/modules/*.py`

### 6.1. `modules/ingest.py` – *EventIngestor*

Tujuan:

- Menggabungkan data **tectonic** dan **volcanic** menjadi satu DataFrame `events` yang bersih dan konsisten.

Kelas utama: ``

Method kunci (konsep):

- `_load_source(path, type_label)`

  - Mencoba membaca `path` sebagai Parquet; kalau gagal, fallback CSV.
  - Menambahkan kolom `type` dengan nilai `type_label` (`"tectonic"` / `"volcanic"`).
  - Mengatur format kolom-kolom penting (`datetime`, `latitude`, `longitude`, `magnitude`, `depth_km`).

- `_safe_datetime(df)`

  - Memastikan kolom `datetime` di-parse menjadi `datetime64[ns, UTC]`.

- ``

  - Memuat dua sumber: tectonic & volcanic.
  - `pd.concat([...])` → gabungkan dua DataFrame.
  - Sortir berdasarkan `datetime` dan reset index.
  - Bersihkan baris dengan `latitude`/`longitude` NaN jika perlu.
  - Simpan ke `cfg.output_path` (`events.parquet`, fallback CSV).
  - Return DataFrame `events`.

Output ini menjadi input untuk ACO.

---

### 6.2. `modules/aco.py` – *ACOStep*

Tujuan:

- Menghitung **skor bahaya** dan **radius dampak** per event dengan pendekatan Ant Colony Optimization.
- Menghasilkan GeoJSON titik bahaya untuk divisualisasikan.

Fungsi penting:

- ``
  - Input: satu baris event (`pd.Series`).
  - Untuk `type="volcanic"`:
    - Memakai `magnitude`, `frp_mw`, `depth_km`, `hotspot_distance_km`.
    - FRP tinggi, kedalaman dangkal, dan hotspot dekat → `eta` besar.
  - Untuk `type="tectonic"`:
    - Memakai `magnitude` dan `depth_km`.
    - Magnitudo besar, kedalaman dangkal → `eta` besar.
  - Output: `hazard_eta_raw` (skor float).

Kelas ``:

- `_prepare(df)`

  - Copy DataFrame.
  - Buat kolom `_dt` = `datetime` dalam UTC, disortir naik.

- `_build_neighbors(n, k)`

  - Untuk setiap index `i` (0..n-1), pilih index tetangga di sekitar `i` dalam rentang ±`k`.
  - Ini membentuk graf tetangga sederhana dalam domain waktu.

- **ACO loop (di **``**)**

  - Hitung `hazard_raw = compute_hazard_eta` untuk semua event.
  - Normalisasi menjadi `eta_norm` dalam [0, 1].
  - Inisialisasi pheromone `tau` sebagai ones.
  - Bangun cache **edge heuristics** `edge_eta(i,j)` yang memperhitungkan:
    - jarak waktu (dalam hari),
    - jarak spasial (dalam km),
    - semakin dekat dalam ruang-waktu → edge lebih menarik.
  - Untuk setiap iterasi (1..`n_iters`):
    - Untuk setiap semut:
      - Pilih titik awal dengan probabilitas ∝ `eta_norm`.
      - Berjalan sepanjang `walk_length` langkah, setiap langkah:
        - memilih tetangga `j` dengan skor ∝ (`tau[j]`^α \* `edge_eta(i,j)`^β).
        - menambah pheromone lokal di `delta_tau[j]` ∝ `eta_norm[j]`.
    - Update global:
      ```python
      tau = (1 - rho) * tau + delta_tau
      ```

- Setelah iterasi:

  - Normalisasi `tau` menjadi `tau_norm`.

- **Radius & area**

  - Untuk setiap event, radius dasar (`r0`) ditentukan dari magnitudo (dan FRP untuk volcanic).

  - Radius final `radius_km` dimodulasi oleh `tau_norm`:

    ```python
    scale = 0.75 + 0.5 * tau_norm[i]
    r = max(min_r, r0 * scale)
    ```

  - `area_km2 = π * radius_km²`.

- Set kolom tambahan:

  - `hazard_eta_raw`, `hazard_eta`, `tau`, `tau_norm`, `radius_km`, `area_km2`.
  - `center_lat`, `center_lon` diset sama dengan `latitude`, `longitude`.

- ``

  - Loop semua baris, buat `Feature` GeoJSON tipe `Point` dengan properti:
    - `datetime`, `type`, `magnitude`, `depth_km`,
    - `hazard_eta`, `tau_norm`, `radius_km`, `area_km2`.
  - Simpan sebagai `events_aco.geojson`.

---

### 6.3. `modules/ga.py` – *GAStep* (orientasi klaster)

Tujuan:

- Dari hasil ACO, **mencari orientasi klaster** per kelompok (berdasarkan `type` dan periode waktu).
- Menghasilkan vektor orientasi (LineString) untuk divisualisasikan.

Fungsi penting:

- ``
  - Mengubah sudut 0–360° menjadi label 8 arah kompas `E, NE, N, NW, W, SW, S, SE`.

Kelas ``:

- `run(events_aco: pd.DataFrame) -> pd.DataFrame`

  Garis besar:

  1. Tambahkan kolom `ga_period` dari `datetime` dengan `dt.to_period(freq)` → misalnya per tahun (`"Y"`).
  2. Inisialisasi kolom GA:
     - `ga_theta_deg`, `ga_direction`, `ga_center_lat`, `ga_center_lon`, `ga_group_radius_km`.
  3. Group berdasarkan `(type, ga_period)`.
  4. Untuk tiap grup:
     - Jika jumlah event `< min_events`, grup dilewati.

     - Bobot **w**:

       - jika ada `hazard_eta_raw` → pakai itu.
       - kalau tidak → fallback ke `magnitude`.

     - Hitung koordinat lokal `x, y` dalam km di sekitar rata-rata lat/lon.

     - Hitung pusat berbobot `(cx, cy)` dengan bobot w.

     - Shift ke koordinat pusat: `xs_c = xs - cx`, `ys_c = ys - cy`.

     - Definisikan **fungsi fitness**:

       ```python
       def fitness(theta_rad):
           ux, uy = cos(theta), sin(theta)
           proj = xs_c * ux + ys_c * uy
           mean_proj = sum(w * proj) / sum(w)
           var = sum(w * (proj - mean_proj)**2) / sum(w)
           return var
       ```

       Jadi fitness = variansi proyeksi ke arah θ.

       → mirip mencari principal axis (sumbu utama) klaster.

     - Jalankan GA:

       - Populasi awal `pop` = array sudut random [0, 2π).
       - Tiap generasi:
         - Hitung fitness tiap individu.
         - Sort descending, ambil beberapa **elite**.
         - Isi populasi baru dengan crossover + mutasi (std mutasi mengecil seiring generasi).
       - Pilih sudut terbaik ``.

     - Dari `(cx, cy)` dan sudut terbaik:

       - Konversi kembali ke lat/lon:
         - `ga_center_lat`, `ga_center_lon`.
       - Hitung jarak RMS `ga_group_radius_km` dari `(cx, cy)`.

     - Tulis nilai GA ke semua baris dalam grup.

- ``

  - Membuat **LineString** dari `ga_center` ke ujung vektor sepanjang `ga_group_radius_km` ke arah `ga_theta_deg`.
  - Satu feature per `(type, period)`.
  - Properti utama: `type`, `theta_deg`, `direction`, `period`, `mode="ga"`.

Output DataFrame (`events_aco_ga.parquet`) memiliki kolom GA yang sama untuk semua event dalam grup, sehingga kamu bisa gabungkan analisis cluster dan event individu.

---

### 6.4. `modules/train.py` – *Trainer* (CNN + LSTM)

Tujuan:

- Membangun **dataset deret waktu** dari hasil ACO+GA.
- Melatih dua model:
  - CNN untuk regresi sudut + klasifikasi arah.
  - LSTM untuk tugas yang sama.
- Menyimpan model dan scaler.

Fungsi dan konstanta penting:

- ``

  ```python
  FEATURE_COLS = [
      "center_lat",
      "center_lon",
      "radius_km",
      "area_km2",
      "magnitude",
      "depth_km",
      "hazard_eta",
      "tau_norm",
  ]
  ```

  Ditambah di kode runtime: `is_tectonic`, `is_volcanic` → total 10 fitur.

- `` dan `DIR_TO_IDX`

  - Mapping label arah kompas → indeks 0..7.

- ``

  - Baca Parquet (atau fallback CSV) dari path.

- ``

  1. Baca `events_aco_ga`.
  2. Pastikan ada kolom `ga_theta_deg` dan `ga_direction`.
  3. Urutkan berdasarkan `datetime` dan drop NA pada fitur penting.
  4. Buat fitur one-hot:
     - `is_tectonic = (type == "tectonic")`.
     - `is_volcanic = (type == "volcanic")`.
  5. Ambil fitur menjadi array `feats` (N, F).
  6. Ambil `angles = ga_theta_deg`, `dirs = mapped ga_direction`.
  7. Bangun **sequence sliding window** sepanjang `seq_len`:
     - Untuk setiap i dari `seq_len-1` .. N-1:
       - X[i] = feats[i-seq\_len+1 : i+1].
       - y\_angle[i] = angles[i].
       - y\_dir[i] = dirs[i].
  8. **Scaling**:
     - Fit `StandardScaler` pada `X_train_raw` (bukan seluruh data) di dalam `Trainer.train()`.
     - Transform X (train & val) → `X_scaled`.

  Return: `X_scaled, y_angle, y_dir, scaler`.

- ``

  - Bungkus numpy array ke `TensorDataset` dan `DataLoader` PyTorch.

- ``

  - Training loop satu epoch:
    - Forward → `angle_pred`, `logits`.
    - Hitung loss gabungan:
      - loss\_reg (circular / MSE, tergantung versi)
      - loss\_cls (CrossEntropy).
    - Backprop, optimizer.step.

- ``

  - Model eval() tanpa grad.
  - Hitung:
    - total loss,
    - **RMSE sudut** (circular),
    - akurasi klasifikasi arah (`acc_dir`).

Kelas ``:

- `train(self) -> dict`
  1. Membaca dataset sequence dari `TrainConfig.events_path`.

  2. Split train/val dengan `train_test_split(..., stratify=y_dir)`.

  3. Inisialisasi device CPU / GPU.

  4. Buat loader train dan val.

  5. **Train CNN**:

     - Buat `CNNAngleDir` dari `src.models.cnn`.
     - Optimizer Adam dengan `cfg.learning_rate`.
     - Loop `epochs_cnn` → panggil `_train_epoch` dan `_eval_epoch`.
     - Track **val loss terbaik**, simpan `state_dict` terbaik ke `models/cnn_angle_dir.pt`.

  6. **Train LSTM**:

     - Proses yang sama dengan model `LSTMAngleDir`.
     - Simpan ke `models/lstm_angle_dir.pt`.

  7. Simpan scaler sebagai `models/scaler_seq{seq_len}.pkl`.

  8. Return dict:

     ```python
     {
       "cnn_model_path": cnn_path,
       "lstm_model_path": lstm_path,
       "scaler_path": scaler_path,
     }
     ```

---

### 6.5. `modules/eval.py` – *DirectionEvaluator* (NB + CNN + LSTM)

Tujuan:

- Menyediakan evaluasi **tiga model** (Gaussian Naive Bayes, CNN, LSTM)
  pada **test set yang sama** agar hasilnya bisa dibandingkan secara adil.
- Menghasilkan **confusion matrix** 8×8 (raw dan normalized) untuk tiap model.

Fungsi penting:

- `build_sequence_dataset` (dari `train.py`)

  - Membangun sequence window `(seq_len, n_features)` beserta label arah & sudut.
  - `split_sequence_indices` memakai `test_size`, `random_state`, dan
    `stratify=y_dir` yang sama dengan `Trainer`, sehingga test set evaluasi
    identik dengan test set saat training.

- `_evaluate_nb`

  - Memakai fitur **event terakhir** dari sequence (`X_raw[:, -1, :]`).
  - Latih `GaussianNB()` pada data train, prediksi data test.

- `_evaluate_torch_model`

  - Load `cnn_angle_dir.pt` / `lstm_angle_dir.pt`, prediksi `logits` dan sudut.
  - Model yang filenya tidak ada akan di-skip (dicatat di `meta.skipped`).

- `_metrics_from_predictions`

  - Menghitung `confusion_matrix` **dengan `labels=range(8)`** sehingga selalu
    8×8 (sebelumnya bisa 7×7 jika ada kelas yang tidak muncul di test set).
  - Menghitung `confusion_matrix_normalized` (normalisasi per baris/kelas aktual).
  - Menghitung `accuracy`, `precision_macro`, `recall_macro`, `f1_macro`,
    `support`, `per_class`, dan `rmse_angle` (wrap-around).

Kelas `DirectionEvaluator`:

- `run(self) -> Dict`
  1. Bangun dataset sequence.
  2. Split indeks train/test (stratified, seed sama dengan training).
  3. Load scaler `scaler_seq{seq_len}.pkl` (fallback fit ulang).
  4. Evaluasi NB, CNN, LSTM pada target event yang sama.
  5. Simpan:
     - `outputs/metrics_models.json` (gabungan ketiga model),
     - `web/data/metrics_models.json` (dipakai panel web),
     - `outputs/metrics_nb.json` (metrik NB, kompatibilitas).

Struktur `metrics_models.json`:

```json
{
  "classes": ["E", "NE", "N", "NW", "W", "SW", "S", "SE"],
  "meta": { "seq_len": 10, "test_size": 0.3, "random_state": 42,
            "n_train": 0, "n_test": 0, "eval_scope": "shared-sequence-test-set",
            "skipped": {} },
  "models": {
    "NB":   { "accuracy": 0, "confusion_matrix": [[0]], "confusion_matrix_normalized": [[0]] },
    "CNN":  { "...": "..." },
    "LSTM": { "...": "..." }
  }
}
```

Panel **Evaluasi Model** di `web/index.html` membaca file ini dan menampilkan
tab model, kartu metrik, serta heatmap confusion matrix interaktif
(raw / normalized).

---

### 6.6. `modules/predict.py` – *Predictor* (ROI)

Tujuan:

- Mengambil **sequence terakhir** dari data (misal 10 event tectonic terakhir).
- Menggunakan model LSTM terlatih untuk memprediksi sudut orientasi berikutnya.
- Membentuk **ROI (Region of Interest)** berupa poligon di peta.

Fungsi penting:

- ``

  - Mencoba memuat scaler `scaler_seq{seq_len}.pkl` di `models/`.
  - Kalau tidak ada → return `None` dan nanti akan fit ulang dari sequence terakhir.

- ``

  1. Baca `events_aco_ga`.
  2. Urutkan berdasarkan `datetime`.
  3. Drop NA pada `center_lat`, `center_lon`, `radius_km`, `area_km2`.
  4. Jika jumlah event valid `< seq_len` → raise error.
  5. Tambah `is_tectonic` dan `is_volcanic`.
  6. Ambil **window terakhir** sepanjang `seq_len` → `df_seq`.
  7. Bentuk fitur `X_raw`.
  8. Jika `scaler is None` → fit scaler baru pada `X_raw`.
  9. Transform → `X_scaled` dan reshape menjadi `(1, seq_len, n_features)` agar siap masuk model.
  10. `meta_last = df_seq.iloc[-1]` → event terakhir di sequence.

  Return: `(X_seq, meta_last, df_seq, scaler)`.

- ``

  1. Menentukan **titik awal**:
     - Pakai `win_center_lat/lon` jika ada,
     - kalau tidak → `center_lat/lon` terakhir.
  2. Menentukan **step\_km** (jarak perpindahan pusat):
     - default: `ga_group_radius_km` terakhir jika ada,
     - kalau tidak → `radius_km` terakhir.
  3. Menentukan **radius ROI**:
     - default: `radius_km` terakhir.
  4. Menggeser titik awal sejauh `step_km` ke arah `pred_angle_deg` (0° = timur, 90° = utara, ccw).
  5. Membuat poligon lingkaran di sekitar pusat prediksi dengan `n_vertices` titik.

  Return dict `roi_info` berisi:

  - `pred_center_lat`, `pred_center_lon`
  - `roi_radius_km`
  - `step_km`
  - `polygon_coords` (list koordinat `[lon, lat]`).

Kelas ``:

- `run(self) -> dict`
  1. Load scaler dari `model_dir`.

  2. Bangun sequence terakhir dengan `build_latest_sequence`.

  3. Load model LSTM `lstm_angle_dir.pt` dan set eval.

  4. Forward `X_seq` → `pred_angle_raw`, `logits`.

  5. Normalisasi sudut:

     ```python
     pred_angle_deg = (pred_angle_raw % 360 + 360) % 360
     ```

  6. Ambil `pred_dir_label` dari argmax logits.

  7. Panggil `forecast_roi(meta_last, pred_angle_deg)`.

  8. Bentuk **Feature GeoJSON** tipe `Polygon` dengan properti:

     - info model & prediksi (`pred_theta_deg`, `pred_direction`, `pred_step_km`, `pred_roi_radius_km`, `pred_center_lat/lon`),
     - konteks event terakhir (`last_datetime`, `last_type`, `last_magnitude`, dsb.),
     - info sequence (`n_sequence_events`).

  9. Simpan sebagai `web/data/inference_roi.geojson`.

  10. Return dict ringkasan `{"pred_angle_deg", "pred_direction", "roi_info", "geojson_path"}`.

---

## 7. Model Deep Learning: `src/models/*.py`

### 7.1. `models/cnn.py` – *CNNAngleDir*

```python
class CNNAngleDir(nn.Module):
    def __init__(self, n_features: int, n_classes: int = 8):
        ...
        self.conv1 = nn.Conv1d(in_channels=n_features, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.act = nn.ReLU()
        self.pool = nn.AdaptiveMaxPool1d(output_size=1)
        self.fc = nn.Linear(64, 64)
        self.dropout = nn.Dropout(p=0.2)
        self.head_angle = nn.Linear(64, 1)
        self.head_dir = nn.Linear(64, n_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1)   # (B, T, F) -> (B, F, T)
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.pool(x).squeeze(-1)  # (B, 64, T) -> (B, 64)
        x = self.dropout(self.act(self.fc(x)))
        angle = self.head_angle(x).squeeze(-1)  # (B,)
        logits = self.head_dir(x)               # (B, 8)
        return angle, logits
```

Intuisi:

- **Conv1d** menangkap pola lokal di sepanjang waktu (window 3 step).
- **AdaptiveMaxPool1d** mengompresi dimensi waktu menjadi satu nilai per filter (semacam “seberapa kuat pola ini muncul di sequence ini”).
- Representasi 64-dim kemudian dipakai untuk
  - regresi sudut (`head_angle`),
  - klasifikasi arah (`head_dir`).

Model ini cocok untuk pola yang dominan berupa “spike” atau pattern lokal pada window.

### 7.2. `models/lstm.py` – *LSTMAngleDir* & *CNNLSTMAngleDir*

```python
class LSTMAngleDir(nn.Module):
    def __init__(self, n_features, hidden_size=64, n_layers=1, n_classes=8):
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, 64)
        self.dropout = nn.Dropout(p=0.2)
        self.head_angle = nn.Linear(64, 1)
        self.head_dir = nn.Linear(64, n_classes)

    def forward(self, x):
        out, (h_n, c_n) = self.lstm(x)   # x: (B, T, F)
        h_last = h_n[-1]                 # (B, hidden_size)
        x = self.dropout(torch.relu(self.fc(h_last)))
        angle = self.head_angle(x).squeeze(-1)
        logits = self.head_dir(x)
        return angle, logits
```

Intuisi:

- LSTM membaca sequence secara berurutan.
- `h_last` menjadi ringkasan dinamis dari seluruh sequence.
- Lagi-lagi, satu backbone, dua head (angle & direction).

Ada juga `` yang menggabungkan CNN dan LSTM, tetapi dalam implementasi sekarang:

- CNN + pooling sudah menghilangkan dimensi waktu → menghasilkan `(B, 64)`.
- LSTM kemudian menerima sequence panjang 1 `(B, 1, 64)`.

Jadi pada versi ini, LSTM lebih berperan sebagai MLP ber-lapis (bukan benar-benar memodelkan sequence panjang). Kalau ke depan ingin hybrid CNN-LSTM yang penuh, bisa diubah supaya LSTM menerima `T` langkah output CNN tanpa pooling dulu.

---

## 8. Frontend: `web/index.html`

File ini adalah **peta web interaktif** menggunakan Leaflet untuk menampilkan output pipeline.

### 8.1. Struktur dasar

- HTML `<head>` memuat:

  - CSS Leaflet dari CDN.
  - CSS kustom untuk layout (map fullscreen, panel legend, control).

- `<body>` berisi:

  - `<div id="map"></div>` → elemen peta.

- `<script>` bagian akhir memuat:

  - JS Leaflet dari CDN.
  - Script kustom untuk:
    - menginisialisasi peta,
    - mendefinisikan layer base (OpenStreetMap, Stamen, dll.),
    - mendefinisikan layer overlay (ACO, GA, ROI),
    - memuat GeoJSON dari `web/data/*.geojson`.

### 8.2. Konfigurasi Layer

Ada objek `LAYER_CONFIGS` berisi sesuatu seperti:

```js
const LAYER_CONFIGS = {
  acoEvents: {
    name: "ACO Events",
    url: "data/events_aco.geojson",
    color: "#1f77b4",
  },
  gaVectors: {
    name: "GA Vectors",
    url: "data/ga_vectors.geojson",
    color: "#2ca02c",
  },
  roi: {
    name: "Predicted ROI",
    url: "data/inference_roi.geojson",
    color: "#d62728",
  },
};
```

Ada juga support untuk layer `rawEvents` jika kamu menambahkannya.

### 8.3. Inisialisasi Peta

- Membuat map dengan center default (misal di Jawa Timur) dan zoom awal.
- Menambahkan base layer OSM.
- Membuat `overlays = {}` untuk menyimpan group layer.
- Helper `` untuk mengupdate `boundsAccumulator`.

### 8.4. Layer ACO Events

Fungsi `loadACOEvents()`:

1. Buat `layer = L.layerGroup()` dan register di `overlays`.

2. `fetch("data/events_aco.geojson")` → parse JSON.

3. Buat `L.geoJSON(geojson, { pointToLayer, onEachFeature })`:

   - `pointToLayer`:
     - Membuat **dua geometry**:
       - **Circle** radius `radius_km * 1000` (buffer area).
       - **CircleMarker** kecil di pusat.
   - `onEachFeature`:
     - Menambahkan tooltip / popup dengan info:
       - datetime, type, magnitude, depth,
       - hazard, tau\_norm, radius, area.

4. Tambahkan GeoJSON ke `layer`, lalu ke map.

5. Panggil `addBounds` untuk meng-update extents.

### 8.5. Layer GA Vectors

Fungsi `loadGAVectors()`:

1. Buat `L.layerGroup()`.
2. Fetch `ga_vectors.geojson`.
3. `L.geoJSON(geojson, { style, onEachFeature })`:
   - `style`:
     - Warna disesuaikan dengan `type` (tectonic/volcanic) atau menggunakan default.
   - Geometry `LineString`
     - Menunjukkan **vektor orientasi klaster** dari GA.
   - Popup bisa menampilkan:
     - `type`, `theta_deg`, `direction`, `period`.

### 8.6. Layer ROI

Fungsi `loadROI()`:

1. Fetch `inference_roi.geojson`.

2. Buat `L.geoJSON(geojson, { style, onEachFeature })`:

   - Geometry `Polygon` menunjukkan area ROI.
   - Style memakai warna unik (misal merah transparan).
   - Popup menampilkan detail prediksi:
     - `pred_direction`, `pred_theta_deg`, `pred_step_km`, `pred_roi_radius_km`,
     - info event terakhir: `last_type`, `last_magnitude`, dll.

3. Tambahkan ke overlay dan panggil `addBounds`.

### 8.7. Legend & Layer Control

- Fungsi `addLegend()` menambahkan kontrol di pojok kanan bawah:

  - Menjelaskan warna:
    - Events Raw
    - ACO Events
    - GA Vectors
    - Predicted ROI

- `L.control.layers(baseLayers, overlays, { collapsed: false })`

  - Menambahkan kontrol checklist layer di pojok kanan atas.

- `setTimeout(...fitBounds...)`

  - Setelah semua layer sempat ter-load, memanggil `map.fitBounds` pada gabungan semua bounds layer.
  - Ini membuat peta otomatis nge-zoom ke area data.

---

## 9. Ringkasan Alur Data & Visualisasi

1. **Ingest**

   - Input: `data/tectonic_events.csv`, `data/volcanic_events.csv`.
   - Output: `outputs/events.parquet`.

2. **ACO**

   - Input: `events.parquet`.
   - Output:
     - `outputs/events_aco.parquet`.
     - `web/data/events_aco.geojson` (titik & radius bahaya).

3. **GA**

   - Input: `events_aco.parquet`.
   - Output:
     - `outputs/events_aco_ga.parquet`.
     - `web/data/ga_vectors.geojson` (vektor orientasi klaster per type-per-period).

4. **Train (CNN + LSTM)**

   - Input: `events_aco_ga.parquet`.
   - Output: model di `models/` + `scaler_seq{seq_len}.pkl`.

5. **Eval (NB + CNN + LSTM)**

   - Input: `events_aco_ga.parquet`, model di `models/`, dan scaler.
   - Output:
     - `outputs/metrics_models.json` (confusion matrix & metrik ketiga model),
     - `web/data/metrics_models.json` (dibaca panel Evaluasi Model),
     - `outputs/metrics_nb.json` (metrik NB, kompatibilitas).

6. **Predict (ROI)**

   - Input: `events_aco_ga.parquet`, model LSTM, scaler.
   - Output: `web/data/inference_roi.geojson`.

7. **Web Frontend**

   - Membaca GeoJSON dari `web/data/` dan memvisualisasikan:
     - daerah bahaya historis (ACO),
     - orientasi klaster (GA),
     - dan ROI prediksi berikutnya.
   - Menampilkan panel **Evaluasi Model** dari `web/data/metrics_models.json`
     berisi tab NB/CNN/LSTM, kartu metrik, dan heatmap confusion matrix
     (raw / normalized).



