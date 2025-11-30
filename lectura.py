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

# Funciones de filtrado
def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    return butter(order, [low, high], btype='band')

def butter_highpass(cutoff, fs, order=4):
    nyq = 0.5 * fs
    high = cutoff / nyq
    return butter(order, high, btype='high')

# Normalización z-score
def normalizar_latidos(X):
    return (X - X.mean(axis=1, keepdims=True)) / X.std(axis=1, keepdims=True)

# Remuestreo lineal a N muestras
def remuestrear_latidos(X, nuevo_tam=100):
    X_resampled = []
    for latido in X:
        x_orig = np.linspace(0, 1, len(latido))
        x_nuevo = np.linspace(0, 1, nuevo_tam)
        interpolador = interp1d(x_orig, latido, kind='linear')
        X_resampled.append(interpolador(x_nuevo))
    return np.array(X_resampled)

# === Pipeline general ===

datapath = r"C:\Users\didie\Documents\proyectodeseñales\mitdb_data"
record_ids = [100, 106, 119, 207, 223, 230, 231]  # Incluye registros con latidos V
clases_interes = ['N', 'V']

# Almacenamiento de datos combinados
X_all = []
y_all = []
groups_all = []

for record_id in record_ids:
    print(f"\nProcesando registro {record_id}...")

    record_path = os.path.join(datapath, str(record_id))

    # Cargar señal y anotaciones
    record = wfdb.rdrecord(record_path)
    ann = wfdb.rdann(record_path, 'atr')

    sig = record.p_signal[:, 0]  # Usamos canal 0
    fs = record.fs
    r_peaks = ann.sample
    symbols = ann.symbol

    # Filtros
    b_hp, a_hp = butter_highpass(0.5, fs)
    sig_hp = filtfilt(b_hp, a_hp, sig)
    b_bp, a_bp = butter_bandpass(5, 15, fs)
    sig_filt = filtfilt(b_bp, a_bp, sig_hp)

    # Segmentación
    pre_R = int(0.150 * fs)
    post_R = int(0.200 * fs)
    ventana = pre_R + post_R

    X = []
    y = []

    for r, s in zip(r_peaks, symbols):
        if s not in clases_interes:
            continue
        if r - pre_R < 0 or r + post_R > len(sig_filt):
            continue  # Evita bordes
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

    # Guardar
    X_all.append(X)
    y_all.append(y)
    groups_all.append(groups)

# Unir todos los registros
X_total = np.vstack(X_all)
y_total = np.hstack(y_all)
groups_total = np.hstack(groups_all)

print("\n✅ Datos combinados:")
print("X:", X_total.shape)
print("y:", y_total.shape)
print("groups:", groups_total.shape)

# Distribución de clases
conteo = Counter(y_total)
print("\nDistribución de clases:")
for clase, cant in conteo.items():
    print(f"{'N' if clase==0 else 'V'}: {cant}")



# Split 80/20 por grupo (registro)
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_idx, test_idx in gss.split(X_total, y_total, groups=groups_total):
    X_train, X_test = X_total[train_idx], X_total[test_idx]
    y_train, y_test = y_total[train_idx], y_total[test_idx]
    groups_train, groups_test = groups_total[train_idx], groups_total[test_idx]

# Verificación
print("\n✅ División completada")
print("Train:", X_train.shape, "Test:", X_test.shape)

# Comprobamos qué registros quedaron en cada grupo
print("Registros en train:", np.unique(groups_train))
print("Registros en test:", np.unique(groups_test))

# Ver distribución de clases
from collections import Counter

print("\nDistribución de clases en train:")
for clase, cant in Counter(y_train).items():
    print(f"{'N' if clase==0 else 'V'}: {cant}")

print("\nDistribución de clases en test:")
for clase, cant in Counter(y_test).items():
    print(f"{'N' if clase==0 else 'V'}: {cant}")


# Ajustamos PCA en los datos de entrenamiento
pca = PCA(n_components=15)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

print("\n✅ PCA aplicado")
print("Forma original:", X_train.shape)
print("Forma reducida:", X_train_pca.shape)

# Cuánta varianza se retiene
var_explicada = np.sum(pca.explained_variance_ratio_)
print(f"Varianza total explicada por 15 componentes: {var_explicada:.2%}")

# Entrenar KNN con k=3
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train_pca, y_train)

# Predicciones
y_pred = knn.predict(X_test_pca)

# Métricas
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='binary')

print("\n✅ Evaluación del modelo KNN (k=3):")
print(f"Accuracy: {acc:.2%}")
print(f"F1 Score: {f1:.2f}")

# Matriz de confusión
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=["N", "V"])
disp.plot(cmap="Blues")
plt.title("Matriz de Confusión - KNN")
plt.tight_layout()
plt.show()