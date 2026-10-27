import pandas as pd

# Tarea 2: carga e inspección de datos
df = pd.read_csv("data/grupo_08.csv")

print("\n=== PRIMERAS 5 FILAS ===")
print(df.head())

print("\n=== DIMENSIONES DEL DATAFRAME ===")
print(df.shape)

print("\n=== TIPOS DE DATOS ===")
print(df.dtypes)

print("\n=== INFORMACIÓN GENERAL ===")
df.info()

print("\n=== VALORES NULOS POR COLUMNA ===")
print(df.isnull().sum())

print("\n=== FILAS DUPLICADAS ===")
print(df.duplicated().sum())

print("\n=== VALORES DE LAS COLUMNAS ===")
for columna in df.columns:
    print(f"\n{columna}:")
    print(df[columna].value_counts(dropna=False).head(15))

# ==========================================
# TAREA 3: LIMPIEZA DE DATOS
# ==========================================

print("\n=== LIMPIEZA DE DATOS ===")

filas_originales = len(df)
media_edad_antes = df["edad_anios"].mean()

# 1. Eliminar duplicados
duplicados = df.duplicated().sum()
df = df.drop_duplicates().copy()
print(f"Duplicados eliminados: {duplicados}")

# 2. Normalizar especialidades
df["especialidad"] = (
    df["especialidad"]
    .str.strip()
    .str.lower()
    .str.capitalize()
)
print("Especialidades normalizadas:")
print(df["especialidad"].value_counts())

# 3. Eliminar edades imposibles
edades_invalidas = ~df["edad_anios"].between(0, 120)
print("\nRegistros con edades imposibles:")
print(df.loc[edades_invalidas])

df = df.loc[~edades_invalidas].copy()

# 4. Completar valores faltantes con la mediana
mediana_dias = df["dias_anticipacion"].median()
nulos = df["dias_anticipacion"].isna().sum()

df["dias_anticipacion"] = (
    df["dias_anticipacion"].fillna(mediana_dias)
)

print(f"\nNulos corregidos: {nulos}")
print(f"Mediana utilizada: {mediana_dias}")

# 5. Guardar CSV limpio
df.to_csv("data/grupo_08_limpio.csv", index=False)

# Resultados finales
print("\n=== RESULTADOS DE LIMPIEZA ===")
print(f"Filas originales: {filas_originales}")
print(f"Filas limpias: {len(df)}")
print(f"Filas eliminadas: {filas_originales - len(df)}")

print("\nDistribución por clase:")
print(df["cita_perdida"].value_counts().sort_index())

print(f"\nMedia edad antes: {media_edad_antes:.2f}")
print(f"Media edad después: {df['edad_anios'].mean():.2f}")

print("\nNulos restantes:")
print(df.isnull().sum())

print("\nCSV limpio guardado correctamente.")

# ==========================================
# TAREA 4: VISUALIZACION CON SEABORN
# ==========================================

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

Path("graficos").mkdir(exist_ok=True)

sns.set_theme(style="whitegrid")

# Grafico 1: Edad vs dias de anticipacion
plt.figure(figsize=(9, 6))

sns.scatterplot(
    data=df,
    x="edad_anios",
    y="dias_anticipacion",
    hue="cita_perdida",
    palette={0: "green", 1: "red"},
    s=70
)

plt.title("Edad y anticipacion de citas medicas")
plt.xlabel("Edad del paciente (anios)")
plt.ylabel("Dias de anticipacion")
plt.legend(title="Cita perdida")
plt.tight_layout()
plt.savefig("graficos/dispersion.png", dpi=150)
plt.close()

# Grafico 2: Anticipacion por clase
plt.figure(figsize=(8, 6))

sns.boxplot(
    data=df,
    x="cita_perdida",
    y="dias_anticipacion",
    hue="cita_perdida",
    palette={0: "green", 1: "red"},
    legend=False
)

plt.title("Dias de anticipacion segun asistencia")
plt.xlabel("Cita perdida (0 = No, 1 = Si)")
plt.ylabel("Dias de anticipacion")
plt.tight_layout()
plt.savefig("graficos/comparacion_clases.png", dpi=150)
plt.close()

print("\n=== TAREA 4 COMPLETADA ===")
print("Grafico 1: graficos/dispersion.png")
print("Grafico 2: graficos/comparacion_clases.png")

print("\nPromedio de dias por clase:")
print(
    df.groupby("cita_perdida")["dias_anticipacion"]
    .agg(["count", "mean", "median"])
)

# ==========================================
# TAREA 5: PERCEPTRON DESDE CERO CON NUMPY
# ==========================================

import numpy as np

# Entradas y variable objetivo
columnas = ["edad_anios", "dias_anticipacion"]
X = df[columnas].to_numpy(dtype=float)
y = df["cita_perdida"].to_numpy(dtype=float)

# Estandarizacion
media = X.mean(axis=0)
desviacion = X.std(axis=0)
X_scaled = (X - media) / desviacion

print("\n=== ESTANDARIZACION ===")
print("Medias:", media)
print("Desviaciones:", desviacion)

# Funciones del perceptron
def sigmoide(z):
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))

def predecir_probabilidad(X, w, b):
    return sigmoide(X @ w + b)

def perdida_logaritmica(y, p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -np.mean(
        y * np.log(p) + (1 - y) * np.log(1 - p)
    )

def exactitud(y, p):
    predicciones = (p >= 0.5).astype(int)
    return np.mean(predicciones == y)

# Inicializacion
w = np.zeros(2)
b = 0.0

tasa_aprendizaje = 0.1
epocas = 1000
historial_perdida = []
historial_exactitud = []

print("\n=== PESOS INICIALES ===")
print(f"w1={w[0]:.4f}, w2={w[1]:.4f}, b={b:.4f}")

prob_inicial = predecir_probabilidad(X_scaled, w, b)
perdida_inicial = perdida_logaritmica(y, prob_inicial)

# Entrenamiento por descenso del gradiente
for epoca in range(epocas):
    probabilidades = predecir_probabilidad(X_scaled, w, b)

    error = probabilidades - y

    dw = (X_scaled.T @ error) / len(y)
    db = np.mean(error)

    w -= tasa_aprendizaje * dw
    b -= tasa_aprendizaje * db

    probabilidades = predecir_probabilidad(X_scaled, w, b)

    historial_perdida.append(
        perdida_logaritmica(y, probabilidades)
    )
    historial_exactitud.append(
        exactitud(y, probabilidades)
    )

    if (epoca + 1) % 100 == 0:
        print(
            f"Epoca {epoca + 1}: "
            f"perdida={historial_perdida[-1]:.4f}, "
            f"exactitud={historial_exactitud[-1]:.4f}"
        )

# Resultados
prob_final = predecir_probabilidad(X_scaled, w, b)
predicciones = (prob_final >= 0.5).astype(int)
errores = int(np.sum(predicciones != y.astype(int)))

print("\n=== RESULTADOS DEL PERCEPTRON ===")
print(f"w1 (edad): {w[0]:.6f}")
print(f"w2 (anticipacion): {w[1]:.6f}")
print(f"b (sesgo): {b:.6f}")
print(f"Perdida inicial: {perdida_inicial:.6f}")
print(f"Perdida final: {historial_perdida[-1]:.6f}")
print(f"Exactitud final: {historial_exactitud[-1]:.2%}")
print(f"Registros mal clasificados: {errores}")
print(f"Registros correctos: {len(y) - errores}")

# Grafico 3: curva de aprendizaje
plt.figure(figsize=(9, 6))
plt.plot(
    range(1, epocas + 1),
    historial_perdida,
    color="blue"
)
plt.title("Curva de aprendizaje del perceptron")
plt.xlabel("Epoca")
plt.ylabel("Perdida logaritmica")
plt.tight_layout()
plt.savefig("graficos/curva_aprendizaje.png", dpi=150)
plt.close()

# Grafico 4: frontera de decision
plt.figure(figsize=(9, 6))

sns.scatterplot(
    data=df,
    x="edad_anios",
    y="dias_anticipacion",
    hue="cita_perdida",
    palette={0: "green", 1: "red"},
    s=70
)

# Frontera en unidades originales
edades = np.linspace(X[:, 0].min(), X[:, 0].max(), 200)

if abs(w[1]) > 1e-10:
    edades_escaladas = (edades - media[0]) / desviacion[0]

    dias_escalados = (
        -b - w[0] * edades_escaladas
    ) / w[1]

    dias_originales = (
        dias_escalados * desviacion[1] + media[1]
    )

    plt.plot(
        edades,
        dias_originales,
        color="black",
        linewidth=2,
        label="Frontera de decision"
    )
elif abs(w[0]) > 1e-10:
    edad_frontera = media[0] - b * desviacion[0] / w[0]
    plt.axvline(
        edad_frontera,
        color="black",
        linewidth=2,
        label="Frontera de decision"
    )

plt.title("Frontera de decision: citas medicas")
plt.xlabel("Edad del paciente (anios)")
plt.ylabel("Dias de anticipacion")
plt.legend()
plt.tight_layout()
plt.savefig("graficos/frontera_decision.png", dpi=150)
plt.close()

# Dos pacientes nuevos, no utilizados en entrenamiento
nuevos = np.array([
    [25, 10],
    [75, 55]
], dtype=float)

esperados = [0, 1]
nuevos_scaled = (nuevos - media) / desviacion
nuevas_prob = predecir_probabilidad(nuevos_scaled, w, b)

print("\n=== PREDICCIONES NUEVAS ===")
for i, paciente in enumerate(nuevos):
    pred = int(nuevas_prob[i] >= 0.5)
    print(
        f"Paciente {i + 1}: edad={paciente[0]:.0f}, "
        f"anticipacion={paciente[1]:.0f}, "
        f"esperado={esperados[i]}, "
        f"probabilidad={nuevas_prob[i]:.4f}, "
        f"prediccion={pred}"
    )

print("\nGraficos del perceptron generados.")
