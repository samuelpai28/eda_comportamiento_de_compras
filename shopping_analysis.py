# Estadistica descriptiva: Comportamiento de compras

import pandas as pd
import numpy as np

# 1. Cargar datos
df = pd.read_csv('archive/shopping_behavior.csv')

# 2. Inspeccion rapida
print("Primeras filas:")
print(df.head(5))
print()

print("Ultimas filas:")
print(df.tail(3))
print()

print("Dimensiones (filas, columnas):", df.shape)
print("Columnas:", list(df.columns))
print()

print("Tipos de datos y nulos:")
print(df.info())
print()

# 3. Estadistica descriptiva (columnas numericas)
print("Resumen numerico (media, desv, min, max, cuartiles):")
print(df.describe())
print()

# 4. Valores unicos y conteos (variables categoricas)
print("Categorias de producto (valores unicos):")
print(df['Category'].unique())
print()

print("Conteo por categoria:")
print(df['Category'].value_counts())
print()

print("Conteo por metodo de pago:")
print(df['Payment Method'].value_counts())
print()

# 5. Limpieza de datos
print("Nulos por columna:")
print(df.isnull().sum())
print()

# En este dataset no hay nulos, se sigue con el original

# 6. Estadisticas por columna (Pandas)
print("Media de Purchase Amount (USD):", df['Purchase Amount (USD)'].mean())
print("Mediana de Purchase Amount (USD):", df['Purchase Amount (USD)'].median())
print("Desviacion estandar:", df['Purchase Amount (USD)'].std())
print("Minimo:", df['Purchase Amount (USD)'].min(), "Maximo:", df['Purchase Amount (USD)'].max())
print()

# 7. NumPy sobre una columna (como en Docs_Numpy)
montos = df['Purchase Amount (USD)'].values
print("Usando NumPy sobre montos:")
print("  Media:", np.mean(montos))
print("  Mediana:", np.median(montos))
print("  Varianza:", np.var(montos))
print("  Suma total:", montos.sum())
print()

# 8. Agrupacion y agregacion (groupby)
print("Gasto promedio por categoria:")
print(df.groupby('Category')['Purchase Amount (USD)'].mean())
print()

print("Gasto total por metodo de pago:")
print(df.groupby('Payment Method')['Purchase Amount (USD)'].sum())
print()

print("Varias estadisticas por temporada:")
print(df.groupby('Season')['Purchase Amount (USD)'].agg(['mean', 'min', 'max', 'count']))
print()

# 9. Filtrado (como en Docs_Pandas)
print("Compras mayores a 80 USD (primeras 5):")
print(df[df['Purchase Amount (USD)'] > 80].head())
print()

print("Compras en Winter con descuento (primeras 3):")
filtro = (df['Season'] == 'Winter') & (df['Discount Applied'] == 'Yes')
print(df[filtro][['Category', 'Purchase Amount (USD)', 'Location']].head(3))
print()

# 10. Seleccion por columnas e indices
print("Solo columnas numericas clave (iloc primeras 5 filas):")
print(df[['Age', 'Purchase Amount (USD)', 'Review Rating', 'Previous Purchases']].iloc[0:5])
print()

# 11. Resumen practico para reporte
resumen_categoria = df.groupby('Category')['Purchase Amount (USD)'].agg(['mean', 'sum', 'count'])
resumen_categoria = resumen_categoria.rename(columns={'mean': 'Promedio_USD', 'sum': 'Total_USD', 'count': 'Cantidad'})
print("Resumen por categoria (para reporte):")
print(resumen_categoria)
