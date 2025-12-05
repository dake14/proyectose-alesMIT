import wfdb
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import filtfilt
from collections import Counter
from sklearn.model_selection import GroupShuffleSplit
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.decomposition import PCA
from funciones.filtros import butter_bandpass, butter_highpass
from funciones.filtros import normalizar_latidos, remuestrear_latidos
from funciones.filtros import pca_manual_fit, pca_manual_transform, knn_predict_manual

# === Pipeline general ===

datapath = r"C:\Users\didie\Documents\proyectodeseñales\mitdb_data"
record_ids = [100, 101, 106, 108, 109,
111, 112, 113, 115, 116,
118, 119, 121, 122, 123,
124, 200, 201, 202, 203,
205, 207, 208, 209, 210,
212, 213, 214, 215, 219,
220, 221, 222, 223, 228,
230, 231, 233, 234]  # Incluye registros con latidos V
clases_interes = ['N', 'V']

# Almacenamiento de datos combinados
X_all = [] # almacena los latidos
y_all = [] # almacena el tipo de latido
groups_all = [] # agrupa en pacientes los latidos

for record_id in record_ids:
    print(f"\nProcesando registro {record_id}...")

    record_path = os.path.join(datapath, str(record_id))

    # Cargar señal y anotaciones
    record = wfdb.rdrecord(record_path)  #matriz con las señales
    ann = wfdb.rdann(record_path, 'atr') 

    sig = record.p_signal[:, 0]  # Estraemos la señal EGC
    fs = record.fs 
    r_peaks = ann.sample # lista de posiciones de las anotaciones o picos R
    symbols = ann.symbol # Extrae la clase o tipo de pulso

    # Filtros
    b_hp, a_hp = butter_highpass(0.5, fs)
    sig_hp = filtfilt(b_hp, a_hp, sig)  # para evitar corrimientos  
    b_bp, a_bp = butter_bandpass(5, 15, fs)
    sig_filt = filtfilt(b_bp, a_bp, sig_hp)  # para evitar corrimientos

    # Segmentación
    pre_R = int(0.150 * fs)  
    post_R = int(0.200 * fs)
    ventana = pre_R + post_R  # ventana del pulso

    X = []   # guarda los latido segmentados con su ventana especifica
    y = []    # guarda las etiquetas del latido

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
    X = normalizar_latidos(X)   #elimina la componente dc restando la media t dividiendo por la desviacion estandar
    X = remuestrear_latidos(X, nuevo_tam=100)  # sampleo nuevo a 100 muestras por ventana mediante interpolacion lineal

    groups = np.array([record_id] * len(y)) # agrupa por registro 

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


print("\nDistribución de clases en train:")
for clase, cant in Counter(y_train).items():
    print(f"{'N' if clase==0 else 'V'}: {cant}")

print("\nDistribución de clases en test:")
for clase, cant in Counter(y_test).items():
    print(f"{'N' if clase==0 else 'V'}: {cant}")


# Ajustamos PCA en los datos de entrenamiento
mu, W, eigvals_top, total_var = pca_manual_fit(X_train, n_components=15)
X_train_pca = pca_manual_transform(X_train, mu, W)
X_test_pca = pca_manual_transform(X_test, mu, W)


print("\n✅ PCA aplicado")
print("Forma original:", X_train.shape)
print("Forma reducida:", X_train_pca.shape)

# Cuánta varianza se retiene
var_explicada = np.sum(eigvals_top) / total_var
print(f"Varianza total explicada por 15 componentes: {var_explicada:.2%}")


# Entrenar KNN con k=3
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train_pca, y_train)

# Predicciones
y_pred = knn_predict_manual(X_train_pca, y_train, X_test_pca, k=10)
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

# ============================================================
#                 G R A F I C A S   F I N A L E S
# ============================================================

print("\nGenerando gráficas finales...")

# -------------------------------------------------------------------
# 1. Extraer primer latido NORMAL y VENTRICULAR desde X_total
# -------------------------------------------------------------------
idx_N = np.where(y_total == 0)[0][0]   # primer Normal
idx_V = np.where(y_total == 1)[0][0]   # primer Ventricular

latido_N = X_total[idx_N]
latido_V = X_total[idx_V]

# Tiempo para latidos remuestreados (100 muestras)
t_latido = np.linspace(0, 1, len(latido_N))

# -------------------------------------------------------------------
# 2. Graficar Normal y Ventricular YA PROCESADOS (normalizados+remuestreados)
# -------------------------------------------------------------------
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(t_latido, latido_N)
plt.title("Latido Normal (procesado)")
plt.grid()

plt.subplot(1,2,2)
plt.plot(t_latido, latido_V, color='r')
plt.title("Latido Ventricular (procesado)")
plt.grid()

plt.tight_layout()
plt.show()

# -------------------------------------------------------------------
# 3. Graficar estado del latido ANTES DE FILTROS del ÚLTIMO REGISTRO
# -------------------------------------------------------------------

# Ya existen: sig (original), sig_hp, sig_filt del último registro procesado
# También pre_R y post_R del loop final

# Buscar primer N y V en el último registro
found_N = None
found_V = None

for r, s in zip(r_peaks, symbols):
    if r - pre_R < 0 or r + post_R > len(sig): 
        continue
    if s == 'N' and found_N is None:
        found_N = r
    if s == 'V' and found_V is None:
        found_V = r
    if found_N is not None and found_V is not None:
        break

# Segmentos ANTES del procesamiento
lat_N_o = sig[found_N-pre_R : found_N+post_R]
lat_V_o = sig[found_V-pre_R : found_V+post_R]

# Segmentos después del filtro HP
lat_N_hp = sig_hp[found_N-pre_R : found_N+post_R]
lat_V_hp = sig_hp[found_V-pre_R : found_V+post_R]

# Segmentos después del filtro BP
lat_N_bp = sig_filt[found_N-pre_R : found_N+post_R]
lat_V_bp = sig_filt[found_V-pre_R : found_V+post_R]

t_raw = np.linspace(-pre_R/fs, post_R/fs, len(lat_N_o))

# ------------------------- Gráficas ---------------------------

# --- Imagen filtros 1: sin filtrar ---
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(t_raw, lat_N_o)
plt.title("Normal (sin filtrar)")
plt.grid()

plt.subplot(1,2,2)
plt.plot(t_raw, lat_V_o, color='r')
plt.title("Ventricular (sin filtrar)")
plt.grid()

plt.tight_layout()
plt.show()

# --- Filtro pasa-altas ---
plt.figure(figsize=(10,6))
plt.subplot(2,1,1)
plt.plot(t_raw, lat_N_hp)
plt.title("Normal - Pasa-altas 0.5 Hz")
plt.grid()

plt.subplot(2,1,2)
plt.plot(t_raw, lat_V_hp, color='r')
plt.title("Ventricular - Pasa-altas 0.5 Hz")
plt.grid()

plt.tight_layout()
plt.show()

# --- Filtro pasa-banda ---
plt.figure(figsize=(10,6))
plt.subplot(2,1,1)
plt.plot(t_raw, lat_N_bp)
plt.title("Normal - Pasa-banda 5–15 Hz")
plt.grid()

plt.subplot(2,1,2)
plt.plot(t_raw, lat_V_bp, color='r')
plt.title("Ventricular - Pasa-banda 5–15 Hz")
plt.grid()

plt.tight_layout()
plt.show()

# -------------------------------------------------------------------
# 4. Ilustración del corte de ventana en la señal sin filtrar
# -------------------------------------------------------------------
# ============================================================
# 4. Ilustración del corte de ventana centrado correctamente
# ============================================================

inicio = found_N - pre_R - 200      # 200 muestras antes
fin    = found_N + post_R + 200     # 200 muestras después

# Limitar dentro del rango real de la señal
inicio = max(0, inicio)
fin = min(len(sig), fin)

plt.figure(figsize=(12,4))
plt.plot(np.arange(inicio, fin), sig[inicio:fin], label="ECG (último registro)")

plt.axvline(found_N - pre_R, color='g', linestyle='--', label="Inicio ventana")
plt.axvline(found_N + post_R, color='r', linestyle='--', label="Fin ventana")

plt.title("Ejemplo de ventana de segmentación (centrada)")
plt.legend()
plt.grid()
plt.show()



# -------------------------------------------------------------------
# 6. Barras Train/Test
# -------------------------------------------------------------------
conteo_train = Counter(y_train)
conteo_test = Counter(y_test)

plt.figure(figsize=(7,5))
plt.bar(["Train N","Train V","Test N","Test V"],
        [conteo_train[0], conteo_train[1], conteo_test[0], conteo_test[1]],
        color=["blue","red","blue","red"])
plt.title("Distribución de clases en Train/Test")
plt.grid(axis='y')
plt.show()

# -------------------------------------------------------------------
# 8. Ilustración KNN
# -------------------------------------------------------------------
pca2 = PCA(n_components=2)
train_2D = pca2.fit_transform(X_train)

plt.figure(figsize=(6,6))
plt.scatter(train_2D[:,0], train_2D[:,1], c=y_train, cmap="bwr", alpha=0.6)
test_point = pca2.transform(X_test[:1])
plt.scatter(test_point[:,0], test_point[:,1], c="yellow", s=200, label="Punto a clasificar")
plt.title("Ilustración del funcionamiento de KNN")
plt.grid()
plt.legend()
plt.show()

print("\n🎉 Gráficas generadas correctamente.")


# ============================================
# Gráfica 2: Comparación Antes vs Después de PCA
# ============================================

plt.figure(figsize=(14,10))

# --------------------------------------
# (A) Señales originales superpuestas
# --------------------------------------
plt.subplot(2,1,1)
for i in range(10):   # graficamos solo 10 para que no se vea caótico
    plt.plot(X_train[i], alpha=0.7)

plt.title("Espacio original (100 muestras por latido)")
plt.xlabel("Índice")
plt.ylabel("Amplitud")
plt.grid()

# --------------------------------------
# (B) Representación en PCA (2D)
# --------------------------------------
plt.subplot(2,1,2)
plt.scatter(X_train_pca[y_train==0,0], X_train_pca[y_train==0,1],
            c='blue', s=15, label="Normal")
plt.scatter(X_train_pca[y_train==1,0], X_train_pca[y_train==1,1],
            c='red', s=15, label="Ventricular")

plt.title("Espacio PCA (2 componentes)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.grid()
plt.legend()

plt.tight_layout()
plt.show()



# ------------------------------------------------------------
# Función para distancia promedio entre todos los pares
# ------------------------------------------------------------

X_train_2d = X_train_pca[:, :2]

# Separamos por clase
N_2d = X_train_2d[y_train == 0]
V_2d = X_train_2d[y_train == 1]

N_15d = X_train_pca[y_train == 0]
V_15d = X_train_pca[y_train == 1]

def distancia_promedio(A, B):
    dist_total = 0
    count = 0
    for a in A:
        dists = np.linalg.norm(B - a, axis=1)
        dist_total += np.sum(dists)
        count += len(dists)
    return dist_total / count

# Distancias
dist_2d = distancia_promedio(N_2d, V_2d)
dist_15d = distancia_promedio(N_15d, V_15d)

print("\nDistancia promedio entre clases:")
print(f" - En PCA 2D:  {dist_2d:.3f}")
print(f" - En PCA 15D: {dist_15d:.3f}")

# ------------------------------------------------------------
# Gráfica comparativa
# ------------------------------------------------------------
plt.figure(figsize=(7,5))
plt.bar(["PCA 2D", "PCA 15D"], [dist_2d, dist_15d],
        color=["purple", "green"])
plt.ylabel("Distancia promedio entre clases")
plt.title("Separación entre clases: Comparación 2D vs 15D")
plt.grid(axis='y')
plt.show()

# ============================================================
# 10. Visualizar señal original vs componentes PCA
# ============================================================

# Elegimos un latido del conjunto de test
latido = X_test[0]             # señal original (100 muestras)
latido_pca = X_test_pca[0]     # sus 15 componentes

# Reconstrucción usando los 15 componentes
latido_rec = latido_pca @ W.T + mu

# Vamos a graficar solamente los primeros 5 componentes
num_comp = 5

plt.figure(figsize=(14,10))

# ---------------------------------------
# (A) Señal original vs reconstruida
# ---------------------------------------
plt.subplot(2,1,1)
plt.plot(latido, label="Original", linewidth=2)
plt.plot(latido_rec, label="Reconstruida PCA (15 comps)", linestyle="--")
plt.title("Señal original vs reconstrucción PCA")
plt.xlabel("Muestras")
plt.ylabel("Amplitud")
plt.grid()
plt.legend()

# ---------------------------------------
# (B) Primeros 5 componentes principales
# ---------------------------------------
plt.subplot(2,1,2)

for i in range(num_comp):
    componente = latido_pca[i] * W[:, i]      # reconstrucción parcial del componente i
    plt.plot(componente, label=f"Componente PCA {i+1}")

plt.title("Contribución de los primeros 5 componentes PCA")
plt.xlabel("Muestras")
plt.ylabel("Amplitud parcial")
plt.grid()
plt.legend()

plt.tight_layout()
plt.show()


# ============================================================
# Gráfica: vecinos más cercanos en PCA 2D
# ============================================================

from sklearn.neighbors import NearestNeighbors

# Tomamos 1 punto del test
punto = X_test_pca[0].reshape(1,-1)

# Calculamos vecinos en train
nbrs = NearestNeighbors(n_neighbors=5, metric='euclidean').fit(X_train_pca)
distancias, indices = nbrs.kneighbors(punto)

vecinos = X_train_pca[indices[0]]
clases_vecinos = y_train[indices[0]]

plt.figure(figsize=(8,6))

# Dibujar todos los puntos del train (azules y rojos)
plt.scatter(X_train_pca[y_train==0,0], X_train_pca[y_train==0,1],
            color="blue", alpha=0.3, label="Normal (Train)")
plt.scatter(X_train_pca[y_train==1,0], X_train_pca[y_train==1,1],
            color="red", alpha=0.3, label="Ventricular (Train)")

# Dibujar el punto a clasificar
plt.scatter(punto[0,0], punto[0,1], color="yellow", s=200,
            edgecolor="black", label="Punto a clasificar")

# Dibujar vecinos y líneas
for i, v in enumerate(vecinos):
    color = "blue" if clases_vecinos[i]==0 else "red"
    plt.scatter(v[0], v[1], color=color, s=150, edgecolor="black")
    plt.plot([punto[0,0], v[0]], [punto[0,1], v[1]], color="gray", linestyle="--")

plt.title("Funcionamiento de KNN: vecinos más cercanos (k=5)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend()
plt.grid()
plt.show()
