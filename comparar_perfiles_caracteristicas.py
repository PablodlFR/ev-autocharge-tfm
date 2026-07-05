# Autor: Pablo de la Fuente Rodríguez
# TFM: Identificación de vehículos eléctricos mediante señales de carga
# Descripción: Comparación de perfiles de carga mediante características estadísticas 
# Año: 2026

from pathlib import Path
import math
import pandas as pd
import matplotlib.pyplot as plt

# 1. CONFIGURACIÓN

CARPETA_DATOS = Path("datos_sinteticos_caracteristicas")

MAGNITUDES = {
    "V": "Voltage RMS Avg (V)",
    "I": "Current RMS Avg (A)",
    "P": "Real Power Avg (kW)",
    "Q": "Reactive Power Avg (kVAR)",
    "S": "Apparent Power Avg (kVA)",
    "F": "Frequency Avg (Hz)",
}


# 2. CARACTERÍSTICAS

def extraer_vector(ruta_csv):
    df = pd.read_csv(ruta_csv)
    vector = {}

    for prefijo, columna in MAGNITUDES.items():
        serie = pd.to_numeric(df[columna], errors="coerce").dropna()

        vector[f"{prefijo}_media"] = serie.mean()
        vector[f"{prefijo}_std"] = serie.std(ddof=0)
        vector[f"{prefijo}_iqr"] = serie.quantile(0.75) - serie.quantile(0.25)

    return vector


def cargar_muestras():
    filas = []

    for carpeta_modelo in sorted(CARPETA_DATOS.iterdir()):
        if not carpeta_modelo.is_dir():
            continue

        modelo = carpeta_modelo.name

        for archivo in sorted(carpeta_modelo.glob("Charging_Profile_*.csv")):
            tipo = "real" if "real" in archivo.name else "sintetico"

            fila = {
                "modelo": modelo,
                "tipo": tipo,
                "archivo": archivo.name,
            }

            fila.update(extraer_vector(archivo))
            filas.append(fila)
    

    return pd.DataFrame(filas)

# 3. NORMALIZACIÓN Y DISTANCIA

def normalizar(tabla):
    columnas_info = ["modelo", "tipo", "archivo"]
    columnas_features = [c for c in tabla.columns if c not in columnas_info]

    tabla_norm = tabla.copy()

    for col in columnas_features:
        minimo = tabla[col].min()
        maximo = tabla[col].max()

        if maximo == minimo:
            tabla_norm[col] = 0
        else:
            tabla_norm[col] = (tabla[col] - minimo) / (maximo - minimo)

    return tabla_norm, columnas_features


def distancia_euclidea(v1, v2):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(v1, v2)))


# 4. MATRIZ SINTÉTICOS VS REALES

def calcular_matriz(tabla_norm, columnas_features):
    reales = tabla_norm[tabla_norm["tipo"] == "real"].reset_index(drop=True)
    sinteticos = tabla_norm[tabla_norm["tipo"] == "sintetico"].reset_index(drop=True)

    filas = []

    for _, sint in sinteticos.iterrows():
        v_sint = sint[columnas_features].to_list()

        for _, real in reales.iterrows():
            v_real = real[columnas_features].to_list()

            filas.append({
                "modelo_sintetico": sint["modelo"],
                "modelo_real": real["modelo"],
                "distancia": distancia_euclidea(v_sint, v_real),
            })

    distancias = pd.DataFrame(filas)

    matriz = distancias.pivot_table(
        index="modelo_sintetico",
        columns="modelo_real",
        values="distancia",
        aggfunc="mean"
    )

    modelos = sorted(matriz.index)
    matriz = matriz.loc[modelos, modelos]

    return matriz


def obtener_umbral_diagonal(matriz):
    peor_mismo = -1
    modelo_peor = None

    for modelo in matriz.index:
        valor = matriz.loc[modelo, modelo]

        if valor > peor_mismo:
            peor_mismo = valor
            modelo_peor = modelo

    return peor_mismo, modelo_peor


# 5. COMPARACIÓN DE tODOS LOS FICHEROS

def comparar_todo_contra_todo(tabla_norm, columnas_features, umbral):
    total = 0
    aciertos = 0

    tp = 0
    tn = 0
    fp = 0
    fn = 0

    for i in range(len(tabla_norm)):
        for j in range(i + 1, len(tabla_norm)):
            fila1 = tabla_norm.iloc[i]
            fila2 = tabla_norm.iloc[j]

            modelo1 = fila1["modelo"]
            modelo2 = fila2["modelo"]

            v1 = fila1[columnas_features].to_list()
            v2 = fila2[columnas_features].to_list()

            distancia = distancia_euclidea(v1, v2)

            mismo_real = modelo1 == modelo2
            mismo_predicho = distancia <= umbral

            correcto = mismo_real == mismo_predicho

            if correcto:
                aciertos += 1

            if mismo_real and mismo_predicho:
                tp += 1
            elif mismo_real and not mismo_predicho:
                fn += 1
            elif not mismo_real and mismo_predicho:
                fp += 1
            else:
                tn += 1

            total += 1

    return total, aciertos, tp, tn, fp, fn


# 6. MAPA DE CALOR

def guardar_mapa_calor(matriz):
    plt.figure(figsize=(10, 8))

    plt.imshow(matriz.values, aspect="auto")
    plt.colorbar(label="Distancia media")

    plt.xticks(range(len(matriz.columns)), matriz.columns, rotation=90)
    plt.yticks(range(len(matriz.index)), matriz.index)

    plt.title("Distancia media entre sintéticos y perfiles reales")
    plt.xlabel("Modelo real comparado")
    plt.ylabel("Modelo de origen del sintético")

    plt.tight_layout()
    plt.savefig("mapa_calor.png", dpi=300)
    plt.close()

# 7. PROGRAMA PRINCIPAL

def main():
    tabla = cargar_muestras()
    tabla_norm, columnas_features = normalizar(tabla)

    matriz = calcular_matriz(tabla_norm, columnas_features)

    umbral, modelo_peor = obtener_umbral_diagonal(matriz)

    total, aciertos, tp, tn, fp, fn = comparar_todo_contra_todo(
        tabla_norm,
        columnas_features,
        umbral
    )

    total_mismo = tp + fn
    total_distinto = tn + fp

    accuracy_global = aciertos / total * 100
    accuracy_mismo = tp / total_mismo * 100
    accuracy_distinto = tn / total_distinto * 100

    matriz.to_csv("matriz_media.csv")
    guardar_mapa_calor(matriz)

    print("\n================ DATOS CARGADOS ================")
    print(f"Modelos: {tabla['modelo'].nunique()}")
    print(f"Perfiles totales: {len(tabla)}")
    print(f"Reales: {len(tabla[tabla['tipo'] == 'real'])}")
    print(f"Sintéticos: {len(tabla[tabla['tipo'] == 'sintetico'])}")

    print("\n================ UMBRAL ================")
    print(f"Umbral tomado de la diagonal principal: {umbral:.4f}")
    print(f"Peor caso mismo modelo: {modelo_peor} — {modelo_peor}")

    print("\n================ RESULTADOS ================")
    print(f"Aciertos totales: {aciertos}/{total}")
    print(f"Accuracy global: {accuracy_global:.2f}%")

    print("\n========= MISMO MODELO =========")
    print(f"Pares mismo modelo: {total_mismo}")
    print(f"Aciertos mismo modelo: {tp}/{total_mismo}")
    print(f"Accuracy mismo modelo: {accuracy_mismo:.2f}%")
    print(f"Falsos negativos: {fn}")

    print("\n========= MODELO DISTINTO =========")
    print(f"Pares modelo distinto: {total_distinto}")
    print(f"Aciertos modelo distinto: {tn}/{total_distinto}")
    print(f"Accuracy modelo distinto: {accuracy_distinto:.2f}%")
    print(f"Falsos positivos: {fp}")

    print("\nArchivos generados:")
    print("- matriz_media.csv")
    print("- mapa_calor.png")


# Ejecutamos el programa principal
main()