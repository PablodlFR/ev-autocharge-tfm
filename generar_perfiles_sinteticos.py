# Autor: Pablo de la Fuente Rodríguez
# TFM: Identificación de vehículos eléctricos mediante señales de carga
# Descripción: Generación de perfiles de carga sintéticos 
# Año: 2026

from pathlib import Path
import numpy as np
import pandas as pd

# 1. CONFIGURACIÓN

CARPETA_ENTRADA = Path("datos")
CARPETA_SALIDA = Path("datos_sinteticos_caracteristicas")

NOMBRE_CSV = "Charging_Profile.csv"

N_SINTETICOS = 30

# Ruido relativo
SIGMA = 0.15

# Escalado global por magnitud
ESCALADO_MIN = 0.85
ESCALADO_MAX = 1.15

SEMILLA = 43
np.random.seed(SEMILLA)

MAGNITUDES = [
    "Voltage RMS Avg (V)",
    "Current RMS Avg (A)",
    "Real Power Avg (kW)",
    "Reactive Power Avg (kVAR)",
    "Apparent Power Avg (kVA)"
]

# 2. FUNCIONES

def cargar_csv(ruta_csv):
    """
    Carga el Charging_Profile.csv original.
    """
    df = pd.read_csv(ruta_csv)

    for columna in MAGNITUDES:
        if columna not in df.columns:
            raise ValueError(f"No existe la columna requerida: {columna}")

    return df


def generar_sintetico(df):
    """
    Genera una versión sintética del Charging_Profile.

    Para cada magnitud eléctrica:
    1. Aplica ruido multiplicativo punto a punto.
    2. Aplica un escalado global.
    """

    df_sintetico = df.copy()

    for columna in MAGNITUDES:
        valores = pd.to_numeric(df_sintetico[columna], errors="coerce").to_numpy(dtype=float)

        # Ruido multiplicativo: x' = x * (1 + epsilon)
        epsilon = np.random.normal(loc=0, scale=SIGMA, size=len(valores))
        valores_ruido = valores * (1 + epsilon)

        # Escalado global: x'' = alpha * x'
        alpha = np.random.uniform(ESCALADO_MIN, ESCALADO_MAX)
        valores_finales = alpha * valores_ruido

        df_sintetico[columna] = valores_finales

    return df_sintetico


def procesar_modelo(carpeta_modelo):
    """
    Genera los CSV sintéticos para cada modelo.
    """
    modelo = carpeta_modelo.name
    ruta_csv = carpeta_modelo / NOMBRE_CSV

    if not ruta_csv.exists():
        print(f"No se encontró {NOMBRE_CSV} en {carpeta_modelo}")
        return

    df_original = cargar_csv(ruta_csv)

    carpeta_salida_modelo = CARPETA_SALIDA / modelo
    carpeta_salida_modelo.mkdir(parents=True, exist_ok=True)

    # Guardar copia del real
    df_original.to_csv(carpeta_salida_modelo / "Charging_Profile_real.csv", index=False)

    # Generar sintéticos
    for i in range(1, N_SINTETICOS + 1):
        df_sint = generar_sintetico(df_original)

        nombre_salida = f"Charging_Profile_sintetico_{i:02d}.csv"
        df_sint.to_csv(carpeta_salida_modelo / nombre_salida, index=False)

# 3. PROGRAMA PRINCIPAL

def main():
    CARPETA_SALIDA.mkdir(parents=True, exist_ok=True)

    carpetas_modelo = sorted([p for p in CARPETA_ENTRADA.iterdir() if p.is_dir()])

    if not carpetas_modelo:
        raise FileNotFoundError(f"No se encontraron modelos dentro de {CARPETA_ENTRADA}")

    for carpeta_modelo in carpetas_modelo:
        procesar_modelo(carpeta_modelo)

    print("Datos sintéticos generados correctamente.")
    print(f"Carpeta de salida: {CARPETA_SALIDA}")


# Ejecutamos el programa principal
main()