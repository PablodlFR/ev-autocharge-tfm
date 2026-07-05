# Autor: Pablo de la Fuente Rodríguez
# TFM: Identificación de vehículos eléctricos mediante señales de carga
# Descripción: Comparación de formas de onda mediante residuos de corriente
# Año: 2026

from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# 1. CONFIGURACIÓN

CARPETA_DATOS = Path("datos")
COLUMNA_CORRIENTE = "Current (A)"
MUESTRAS_POR_CICLO = 512

OBJETIVO_SMOTE = 9

# 2. LEER CORRIENTE

def leer_corriente(ruta_csv):
    with open(ruta_csv, "r", encoding="utf-8") as f:
        lineas = f.readlines()

    fila_datos = None

    for i, linea in enumerate(lineas):
        if linea.startswith("Time"):
            fila_datos = i
            break

    df = pd.read_csv(ruta_csv, skiprows=fila_datos)

    corriente = pd.to_numeric(
        df[COLUMNA_CORRIENTE],
        errors="coerce"
    ).dropna().to_numpy()

    return corriente

# 3. EXTRAER RESIDUO DE LA ONDA

def obtener_residuo(ruta_csv):
    corriente = leer_corriente(ruta_csv)

    # Usar el primer ciclo completo
    ciclo = corriente[:MUESTRAS_POR_CICLO]

    # Alinear el pico positivo
    pico = np.argmax(ciclo)
    objetivo = len(ciclo) // 4
    desplazamiento = objetivo - pico
    ciclo = np.roll(ciclo, desplazamiento)

    # Crear los ángulos de un ciclo completo: 0 -> 2pi
    n = len(ciclo)
    t = np.linspace(0, 2 * np.pi, n)

    # Crear seno y coseno
    seno = np.sin(t)
    coseno = np.cos(t)

    # Calcular los coeficientes de la sinusoide ideal
    a = 2 * np.mean(ciclo * seno)
    b = 2 * np.mean(ciclo * coseno)

    # Crear la sinusoide ideal
    sinusoide = a * seno + b * coseno

    # Residuo = diferencia entre la onda real y la sinusoide ideal
    residuo = ciclo - sinusoide

    return residuo

# 4. CARGAR TODAS LAS FORMAS DE ONDA

def cargar_muestras():
    muestras = []

    for carpeta_vehiculo in sorted(CARPETA_DATOS.iterdir()):
        if not carpeta_vehiculo.is_dir():
            continue

        carpeta_waveforms = carpeta_vehiculo / "Waveforms"

        if not carpeta_waveforms.exists():
            continue

        modelo = carpeta_vehiculo.name
        ficheros = sorted(carpeta_waveforms.glob("Waveform_*.csv"))

        ficheros_usados = ficheros

        for fichero in ficheros_usados:
            residuo = obtener_residuo(fichero)

            muestras.append({
                "modelo": modelo,
                "fichero": fichero.name,
                "residuo": residuo
            })

    return muestras

# 5. EVALUAR MODELOS

def evaluar_modelos_ml(muestras):

    # Crear X e y con todas las muestras reales
    X = np.array([
        muestra["residuo"]
        for muestra in muestras
    ])

    y = np.array([
        muestra["modelo"]
        for muestra in muestras
    ])

    # Aplicar SMOTE
    sampling_strategy = {
        modelo: OBJETIVO_SMOTE
        for modelo in np.unique(y)
    }

    smote = SMOTE(
        sampling_strategy=sampling_strategy,
        k_neighbors=1,
        random_state=57
    )

    X_smote, y_smote = smote.fit_resample(X, y)

    # 80 % train, 20 % test
    X_train, X_test, y_train, y_test = train_test_split(
        X_smote,
        y_smote,
        test_size=0.20,
        stratify=y_smote,
        random_state=57
    )

    print("DIVISIÓN ENTRENAMIENTO / PRUEBA:\n")
    print(f"Waveforms de entrenamiento: {len(X_train)}")
    print(f"Waveforms de prueba: {len(X_test)}")

    # Modelos a comparar
    modelos = {
        "k-NN": make_pipeline(
            StandardScaler(),
            KNeighborsClassifier(n_neighbors=3)
        ),

        "Regresión logística": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=3000)
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            random_state=42
        )
    }

    resumen = []
    resultados_detallados = []

    for nombre_modelo, modelo in modelos.items():

        # Entrenar modelo
        modelo.fit(X_train, y_train)

        # Predecir
        predicciones = modelo.predict(X_test)

        # Calcular resultados
        aciertos = (predicciones == y_test).sum()
        total = len(y_test)

        accuracy = accuracy_score(y_test, predicciones) * 100

        precision = precision_score(
            y_test,
            predicciones,
            average="macro",
            zero_division=0
        )

        recall = recall_score(
            y_test,
            predicciones,
            average="macro",
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            predicciones,
            average="macro",
            zero_division=0
        )

        print(f"\n{nombre_modelo}:\n")
        print(f"Aciertos: {aciertos}/{total}")
        print(f"Accuracy: {accuracy:.2f} %")
        print(f"F1-score: {f1:.2f}")
        print(f"Precision: {precision:.2f}")
        print(f"Recall: {recall:.2f}")

        resumen.append({
            "modelo": nombre_modelo,
            "train": len(X_train),
            "test": len(X_test),
            "aciertos": aciertos,
            "total": total,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        })

        for real, predicho in zip(y_test, predicciones):
            resultados_detallados.append({
                "modelo_ml": nombre_modelo,
                "modelo_real": real,
                "fichero": "smote",
                "modelo_predicho": predicho,
                "correcto": real == predicho
            })

    pd.DataFrame(resultados_detallados).to_csv(
        "resultados_modelos.csv",
        index=False
    )

# 6. MATRIZ DE CONFUSIÓN

def guardar_matriz_confusion(nombre_modelo="Random Forest"):
    df = pd.read_csv("resultados_modelos.csv")

    # Filtrar resultados del modelo elegido
    df_modelo = df[df["modelo_ml"] == nombre_modelo]

    clases = sorted(df_modelo["modelo_real"].unique())

    matriz = confusion_matrix(
        df_modelo["modelo_real"],
        df_modelo["modelo_predicho"],
        labels=clases,
        normalize="true"
    )

    fig, ax = plt.subplots(figsize=(9, 7))

    disp = ConfusionMatrixDisplay(
        confusion_matrix=matriz,
        display_labels=clases
    )

    disp.plot(
        ax=ax,
        cmap="Blues",
        values_format=".2f",
        colorbar=True
    )

    ax.set_title(f"Matriz de confusión - {nombre_modelo}")
    ax.set_xlabel("Modelo predicho")
    ax.set_ylabel("Modelo real")

    plt.xticks(rotation=90)
    plt.yticks(rotation=0)

    plt.tight_layout()

    nombre_archivo = nombre_modelo.lower().replace(" ", "_")

    plt.savefig(f"matriz_confusion_{nombre_archivo}.png", dpi=300)

    plt.close()

# 7. PROGRAMA PRINCIPAL

def main():
    muestras = cargar_muestras()


    evaluar_modelos_ml(muestras)

    guardar_matriz_confusion("k-NN")
    guardar_matriz_confusion("Regresión logística")
    guardar_matriz_confusion("Random Forest")


# Ejecutamos el programa principal
main()
