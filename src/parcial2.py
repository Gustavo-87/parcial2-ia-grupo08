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
