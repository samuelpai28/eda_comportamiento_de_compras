"""
    pip install scikit-learn matplotlib pandas numpy
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
    accuracy_score,
    confusion_matrix,
)


# 1. Cargar datos
df = pd.read_csv("archive/shopping_behavior.csv")


# 2. Regresión lineal 
#    Variable dependiente (y): monto de la compra (Purchase Amount).
#    Variables independientes (X): edad, género, compras previas y calificación de reseña.

X_reg = df[["Age", "Gender", "Previous Purchases", "Review Rating"]]
y_reg = df["Purchase Amount (USD)"]

# Separar en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=40
)

# Pasos en código: instanciar, entrenar, predecir
modelo_linear = LinearRegression()
modelo_linear.fit(X_train, y_train)  # entrenar (fit)
y_pred = modelo_linear.predict(X_test)  # predecir (predict)

# Cálculo de residuos 
residuos = y_test - y_pred

print("=== REGRESIÓN LINEAL: Predicción de Purchase Amount (USD) ===")
print("Coeficientes (Age, Gender, Previous Purchases, Review Rating):", modelo_linear.coef_)
print("Intercepto:", modelo_linear.intercept_)
# Métricas de desempeño
y_train_pred = modelo_linear.predict(X_train)
r2_train = r2_score(y_train, y_train_pred)
r2_test = r2_score(y_test, y_pred)
mse_test = mean_squared_error(y_test, y_pred)
rmse_test = np.sqrt(mse_test)
mae_test = mean_absolute_error(y_test, y_pred)
print(f"R^2 entrenamiento: {r2_train:.3f}")
print(f"R^2 prueba:        {r2_test:.3f}")
print(f"RMSE prueba:       {rmse_test:.3f}")
print(f"MAE prueba:        {mae_test:.3f}")
print("Primeras 5 predicciones vs valores reales:")
for real, pred in list(zip(y_test.values[:5], y_pred[:5])):
    print(f"Real: {real:6.2f}  |  Predicho: {pred:6.2f}")
print()


# 3. Regresión logística 
#    Objetivo binario: ¿se aplicó descuento? (Discount Applied: Yes/No).
#    y_binario = 1 si hubo descuento, 0 en caso contrario.

df["discount_bin"] = (df["Discount Applied"] == "Yes").astype(int)
df["subscription_bin"] = (df["Subscription Status"] == "Yes").astype(int)

X_log = df[
    [
        "Age",
        "Gender",
        "Previous Purchases",
        "Review Rating",
        "Purchase Amount (USD)",
        "subscription_bin",
    ]
]
y_log = df["discount_bin"]

X_train_log, X_test_log, y_train_log, y_test_log = train_test_split(
    X_log, y_log, test_size=0.2, random_state=40
)

log_reg = LogisticRegression(max_iter=1000, class_weight="balanced")
log_reg.fit(X_train_log, y_train_log)  # entrenar 

probas = log_reg.predict_proba(X_test_log)[:, 1]  # probabilidades  
pred_bin = (probas >= 0.5).astype(int)  # umbral 0.5 

print("=== REGRESIÓN LOGÍSTICA: Probabilidad de descuento aplicado (Yes) ===")
print("Distribución real de la variable objetivo (0=No, 1=Sí):")
print(y_log.value_counts())
print()
print("Primeras 5 probabilidades y clases predichas vs reales:")
for p, pred, real in list(zip(probas[:5], pred_bin[:5], y_test_log.values[:5])):
    print(f"Probabilidad descuento: {p:5.3f}  |  Predicho: {pred}  |  Real: {real}")
print()
acc = accuracy_score(y_test_log, pred_bin)
cm = confusion_matrix(y_test_log, pred_bin)
print(f"Exactitud (accuracy) en prueba: {acc:.3f}")
print("Matriz de confusión (filas = reales, columnas = predichos):")
print(cm)
print()


# 4. Visualización tipo histograma (L59) para variables numéricas
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.hist(df["Purchase Amount (USD)"], bins=20, edgecolor="black")
plt.title("Histograma de Purchase Amount (USD)")
plt.xlabel("Monto de compra")
plt.ylabel("Frecuencia")

plt.subplot(1, 2, 2)
plt.hist(df["Age"], bins=20, edgecolor="black")
plt.title("Histograma de Age")
plt.xlabel("Edad")
plt.ylabel("Frecuencia")

plt.tight_layout()
plt.show()


# 5. Gráfica de residuos de la regresión lineal (L78)
plt.figure(figsize=(6, 4))
plt.scatter(y_pred, residuos, alpha=0.5)
plt.axhline(0, color="red", linestyle="--")
plt.title("Residuos vs Predicción (Regresión Lineal)")
plt.xlabel("Valor predicho (Purchase Amount)")
plt.ylabel("Residuo (real - predicho)")
plt.tight_layout()
plt.show()

