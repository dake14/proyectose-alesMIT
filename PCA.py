import wfdb
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, filtfilt
from scipy.interpolate import interp1d
from collections import Counter
from sklearn.model_selection import GroupShuffleSplit
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.decomposition import PCA

# -----------------------------------------------------------
#                      FUNCIONES
# -----------------------------------------------------------

def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    return butter(order, [low, high], btype='band')

def butter_highpass(cutoff, fs, order=4):
    nyq = 0.5 * fs
    high = cutoff / nyq
    return butter(order, high, btype='high')

def normalizar_latidos(X):
    return (X - X.mean(axis=1, keepdims=True)) / X.std(axis=1, keepdims=True)

def remuestrear_latidos(X, nuevo_tam=100):
    X_resampled = []
    for latido in X:
        x_orig = np.linspace(0, 1, len(latido))
        x_nuevo = np.linspace(0, 1, nuevo_tam)
        interpolador = interp1d(x_orig, latido, kind='linear')
        X_resampled.append(interpolador(x_nuevo))
    return np.array(X_resampled)

# -----------------------------------------------------------
#                  PIPELINE DE TU PROYECTO
# -----------------------------------------------------------

datapath = r"C:\Users\didie\Documents\proyectodeseñales\mitdb_data"
record_ids = [100, 106, 119, 207, 223, 230, 231]
clases_interes = ['N', 'V']

X_all, y_all, groups_all = [], [], []

for record_id in record_ids:
    print(f"\nProcesando registro {record_id}...")

    record_path = os.path.join(datapath, str(record_id))
    record = wfdb.rdrecord(record_path)
    ann = wfdb.rdann(record_path, 'atr')

    sig = record.p_signal[:, 0]
    fs = record.fs
    r_peaks = ann.sample
    symbols = ann.symbol

    # ----------------------------------------------
    #                 FILTRADO
    # ----------------------------------------------
    b_hp, a_hp = butter_highpass(0.5, fs)
    sig_hp = filtfilt(b_hp, a_hp, sig)

    b_bp, a_bp = butter_bandpass(5, 15, fs)
    sig_filt = filtfilt(b_bp, a_bp, sig_hp)

    # -----------------------------------------------------------
    #     NUEVA VISUALIZACIÓN – Buscar segmento con N y V
    # -----------------------------------------------------------
    if record_id == 106:  # registro que sí tiene N y V

        window_duration = 7  # segundos a graficar
        step = fs * 1        # mover ventana 1 segundo cada vez
        total_len = len(sig)

        found = False
        best_start = None

        # buscar intervalo donde existan N y V
        for start in range(0, total_len - int(window_duration*fs), int(step)):
            end = start + int(window_duration * fs)

            seg_symbols = []
            for r, s in zip(r_peaks, symbols):
                if start <= r < end and s in ['N', 'V']:
                    seg_symbols.append(s)

            if 'N' in seg_symbols and 'V' in seg_symbols:
                best_start = start
                found = True
                break

        if not found:
            print("⚠ No se encontró segmento con N y V juntos.")
        else:
            win_start = best_start
            win_end = best_start + int(window_duration * fs)

            t_win = np.linspace(0, (win_end - win_start)/fs, win_end - win_start)
            sig_win_orig = sig[win_start:win_end]
            sig_win_hp   = sig_hp[win_start:win_end]
            sig_win_bp   = sig_filt[win_start:win_end]

            # picos locales
            r_local = []
            sym_local = []
            for r, s in zip(r_peaks, symbols):
                if win_start <= r < win_end:
                    r_local.append(r - win_start)
                    sym_local.append(s)

            plt.figure(figsize=(12, 7))

            # ---------------- ORIGINAL ----------------
            ax1 = plt.subplot(3, 1, 1)
            plt.plot(t_win, sig_win_orig, color='black', linewidth=0.9)
            for rp, ss in zip(r_local, sym_local):
                color = 'green' if ss == 'N' else 'red'
                plt.scatter(rp / fs, sig_win_orig[rp], s=35, color=color)

            plt.title("ECG Original – Latidos Normales (verde) y Ventriculares (rojo)")
            plt.ylabel("Amplitud (mV)")

            # ---------------- PASAALTAS ----------------
            ax2 = plt.subplot(3, 1, 2, sharex=ax1)
            plt.plot(t_win, sig_win_hp, color='orange', linewidth=0.9)
            for rp, ss in zip(r_local, sym_local):
                color = 'green' if ss == 'N' else 'red'
                plt.scatter(rp / fs, sig_win_hp[rp], s=35, color=color)

            plt.title("ECG después del Filtro Pasaaltas (0.5 Hz)")
            plt.ylabel("Amplitud (mV)")

            # ---------------- PASABANDA ----------------
            ax3 = plt.subplot(3, 1, 3, sharex=ax1)
            plt.plot(t_win, sig_win_bp, color='blue', linewidth=0.9)
            for rp, ss in zip(r_local, sym_local):
                color = 'green' if ss == 'N' else 'red'
                plt.scatter(rp / fs, sig_win_bp[rp], s=35, color=color)

            plt.title("ECG después del Filtro Pasabanda (5–15 Hz)")
            plt.xlabel("Tiempo (s)")
            plt.ylabel("Amplitud (mV)")
            plt.tight_layout()

    # ----------------------------------------------
    #            SEGMENTACIÓN DE LATIDOS
    # ----------------------------------------------
    pre_R = int(0.150 * fs)
    post_R = int(0.200 * fs)

    X, y = [], []

    for r, s in zip(r_peaks, symbols):
        if s not in clases_interes:
            continue
        if r - pre_R < 0 or r + post_R > len(sig_filt):
            continue
        latido = sig_filt[r - pre_R : r + post_R]
        X.append(latido)
        y.append(0 if s == 'N' else 1)

    if not X:
        print(f"No se encontraron latidos válidos en {record_id}")
        continue

    X = np.array(X)
    y = np.array(y)

    X = normalizar_latidos(X)
    X = remuestrear_latidos(X, nuevo_tam=100)

    groups = np.array([record_id] * len(y))

    X_all.append(X)
    y_all.append(y)
    groups_all.append(groups)

# -----------------------------------------------------------
# RESTO DEL CÓDIGO (NO MODIFICADO)
# -----------------------------------------------------------

X_total = np.vstack(X_all)
y_total = np.hstack(y_all)
groups_total = np.hstack(groups_all)

conteo = Counter(y_total)
plt.figure(figsize=(5, 4))
plt.bar(['Normal (N)', 'Ventricular (V)'], conteo.values(),
        color=['green', 'red'])
plt.title("Distribución de clases")
plt.ylabel("Cantidad de latidos")
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()

gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_idx, test_idx in gss.split(X_total, y_total, groups_total):
    X_train, X_test = X_total[train_idx], X_total[test_idx]
    y_train, y_test = y_total[train_idx], y_total[test_idx]

pca = PCA(n_components=15)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

plt.figure(figsize=(6, 4))
plt.plot(np.cumsum(pca.explained_variance_ratio_) * 100, marker='o')
plt.title("Varianza explicada acumulada (PCA)")
plt.xlabel("Componentes")
plt.ylabel("Varianza (%)")
plt.grid(True)
plt.tight_layout()

plt.figure(figsize=(10, 6))
for i in range(15):
    plt.plot(pca.components_[i], label=f"C{i+1}")
plt.title("Primeros 15 Componentes Principales")
plt.xlabel("Muestras")
plt.ylabel("Peso")
plt.legend(fontsize=7)
plt.tight_layout()

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train_pca, y_train)

y_pred = knn.predict(X_test_pca)
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\nEvaluación del modelo KNN:")
print(f"Accuracy: {acc:.2%}")
print(f"F1 Score: {f1:.2f}")

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=["N", "V"])
disp.plot(cmap="Blues")
plt.title("Matriz de Confusión - PCA + KNN")
plt.tight_layout()
plt.show()
