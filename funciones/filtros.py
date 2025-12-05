import numpy as np
from scipy.signal import butter
from scipy.interpolate import interp1d
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


# Normalizacion de muestras 
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



def pca_manual_fit(X, n_components=15):
    """
    X: matriz (n_muestras, n_features)
    Devuelve:
      - mu: media por columna
      - W: autovectores principales (n_features, n_components)
      - eigvals_top: autovalores principales
      - total_var: suma total de autovalores (para varianza explicada)
    """
    # 1) Centrar datos
    mu = X.mean(axis=0)
    X_centered = X - mu

    # 2) Matriz de covarianza (features en columnas)
    cov = np.cov(X_centered, rowvar=False)

    # 3) Autovalores y autovectores
    eigvals, eigvecs = np.linalg.eigh(cov)  # cov es simétrica

    # 4) Ordenar de mayor a menor
    idx = np.argsort(eigvals)[::-1]
    eigvals_sorted = eigvals[idx]
    eigvecs_sorted = eigvecs[:, idx]

    # 5) Tomar solo los primeros n_components
    W = eigvecs_sorted[:, :n_components]
    eigvals_top = eigvals_sorted[:n_components]
    total_var = eigvals_sorted.sum()

    return mu, W, eigvals_top, total_var


def pca_manual_transform(X, mu, W):
    """
    Proyecta X centrada sobre los autovectores W.
    """
    X_centered = X - mu
    return X_centered @ W  # (n_muestras, n_features) · (n_features, n_comp)



def knn_predict_manual(X_train, y_train, X_test, k=3):
    """
    Implementación simple de KNN:
      - Distancia euclidiana
      - Votación mayoritaria
    """
    preds = []

    for x in X_test:
        # 1) Distancias a todos los puntos de train
        dists = np.linalg.norm(X_train - x, axis=1)

        # 2) Índices de los k vecinos más cercanos
        idx_sorted = np.argsort(dists)[:k]

        # 3) Clases de esos vecinos
        vecinos = y_train[idx_sorted]

        # 4) Votación mayoritaria
        vals, counts = np.unique(vecinos, return_counts=True)
        pred = vals[np.argmax(counts)]

        preds.append(pred)

    return np.array(preds)