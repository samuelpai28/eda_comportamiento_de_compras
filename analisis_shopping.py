"""
Análisis Estadístico Descriptivo y Modelado con Python
Dataset: Shopping Behavior
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix, classification_report
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Configuración de estilo
try:
    plt.style.use('seaborn-v0_8-darkgrid')
except:
    try:
        plt.style.use('seaborn-darkgrid')
    except:
        plt.style.use('ggplot')
sns.set_palette("husl")

print("="*80)
print("ANÁLISIS ESTADÍSTICO DESCRIPTIVO Y MODELADO - SHOPPING BEHAVIOR")
print("="*80)

# ============================================================================
# 1. CARGA Y MANIPULACIÓN DE DATOS
# ============================================================================
print("\n" + "="*80)
print("1. CARGA Y MANIPULACIÓN DE DATOS")
print("="*80)

# Cargar datos
df = pd.read_csv('shopping_behavior_updated.csv')
print(f"\n[INFO] Dimensiones del dataset: {df.shape}")
print(f"   - Filas: {df.shape[0]}")
print(f"   - Columnas: {df.shape[1]}")

# Inspección inicial
print("\n[INFO] Primeras filas del dataset:")
print(df.head())

print("\n[INFO] Información del dataset:")
print(df.info())

print("\n[INFO] Columnas disponibles:")
print(df.columns.tolist())

# Estadística descriptiva global
print("\n" + "-"*80)
print("ESTADÍSTICA DESCRIPTIVA GLOBAL (df.describe())")
print("-"*80)
print(df.describe())

# Verificar valores nulos
print("\n" + "-"*80)
print("VALORES NULOS POR COLUMNA")
print("-"*80)
null_counts = df.isnull().sum()
print(null_counts[null_counts > 0] if null_counts.sum() > 0 else "[OK] No hay valores nulos")

# Limpieza de datos
print("\n[INFO] Limpiando datos...")
df_clean = df.copy()

# Rellenar valores nulos si existen (usando fillna como menciona la guía)
if df_clean.isnull().sum().sum() > 0:
    # Para columnas numéricas, usar la media
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    df_clean[numeric_cols] = df_clean[numeric_cols].fillna(df_clean[numeric_cols].mean())
    
    # Para columnas categóricas, usar la moda
    categorical_cols = df_clean.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0] if len(df_clean[col].mode()) > 0 else 'Unknown')
    print("[OK] Valores nulos rellenados")
else:
    print("[OK] No se encontraron valores nulos")

# ============================================================================
# USO DE PANDAS: df.groupby() - Segmentar estadísticas por categorías
# ============================================================================
# Estadísticas descriptivas por categorías (usando groupby)
print("\n" + "-"*80)
print("ESTADÍSTICAS POR CATEGORÍA (df.groupby())")
print("-"*80)
if 'Category' in df_clean.columns:
    # ⭐ PANDAS: df.groupby() - Agrupa datos por categoría para análisis segmentado
    stats_by_category = df_clean.groupby('Category')['Purchase Amount (USD)'].agg(['mean', 'std', 'count'])
    print(stats_by_category)

# Estadísticas por género
if 'Gender' in df_clean.columns:
    print("\n" + "-"*80)
    print("ESTADÍSTICAS POR GÉNERO")
    print("-"*80)
    # ⭐ PANDAS: df.groupby() - Agrupa datos por género para análisis segmentado
    stats_by_gender = df_clean.groupby('Gender')['Purchase Amount (USD)'].agg(['mean', 'std', 'count'])
    print(stats_by_gender)

# ============================================================================
# 2. ANÁLISIS ESTADÍSTICO DESCRIPTIVO CON NUMPY
# ============================================================================
print("\n" + "="*80)
print("2. ANÁLISIS ESTADÍSTICO DESCRIPTIVO (NUMPY)")
print("="*80)

# Convertir columnas numéricas a arrays de NumPy
purchase_amount = df_clean['Purchase Amount (USD)'].values
age = df_clean['Age'].values
review_rating = df_clean['Review Rating'].values
previous_purchases = df_clean['Previous Purchases'].values

# ============================================================================
# USO DE NUMPY: np.mean() y np.std()
# ============================================================================
print("\n[STATS] Estadísticas de Purchase Amount (USD):")
# ⭐ NUMPY: np.mean() - Calcula el centro de los datos (media aritmética)
print(f"   Media (np.mean): {np.mean(purchase_amount):.2f}")
print(f"   Mediana: {np.median(purchase_amount):.2f}")
# ⭐ NUMPY: np.std() - Mide la dispersión o "ruido" en los datos (desviación estándar)
print(f"   Desviación estándar (np.std): {np.std(purchase_amount):.2f}")
print(f"   Mínimo: {np.min(purchase_amount):.2f}")
print(f"   Máximo: {np.max(purchase_amount):.2f}")
print(f"   Rango: {np.max(purchase_amount) - np.min(purchase_amount):.2f}")

print("\n[STATS] Estadísticas de Age:")
# ⭐ NUMPY: np.mean() - Calcula el centro de los datos (edad promedio)
print(f"   Media (np.mean): {np.mean(age):.2f}")
# ⭐ NUMPY: np.std() - Mide la dispersión en las edades
print(f"   Desviación estándar (np.std): {np.std(age):.2f}")

print("\n[STATS] Estadísticas de Review Rating:")
# ⭐ NUMPY: np.mean() - Calcula el centro de los datos (calificación promedio)
print(f"   Media (np.mean): {np.mean(review_rating):.2f}")
# ⭐ NUMPY: np.std() - Mide la dispersión en las calificaciones
print(f"   Desviación estándar (np.std): {np.std(review_rating):.2f}")

# ============================================================================
# 3. REGRESIÓN LINEAL: Predicción de Purchase Amount
# ============================================================================
print("\n" + "="*80)
print("3. REGRESIÓN LINEAL: Predicción de Purchase Amount (USD)")
print("="*80)

# Preparar variables independientes (X) y dependiente (y)
# Variable dependiente (y): Purchase Amount
# Variables independientes (X): Age, Review Rating, Previous Purchases

# Preparar datos para regresión lineal
X_features = ['Age', 'Review Rating', 'Previous Purchases']
X = df_clean[X_features].values
y = df_clean['Purchase Amount (USD)'].values

print(f"\n[MODEL] Variables Independientes (X): {X_features}")
print(f"[MODEL] Variable Dependiente (y): Purchase Amount (USD)")
print(f"[MODEL] Forma de X: {X.shape}")
print(f"[MODEL] Forma de y: {y.shape}")

# ============================================================================
# USO DE SCIKIT-LEARN: Regresión Lineal
# ============================================================================
# Dividir datos en entrenamiento y prueba
# ⭐ SCIKIT-LEARN: train_test_split() - Divide datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\n[INFO] División de datos:")
print(f"   Entrenamiento: {X_train.shape[0]} muestras")
print(f"   Prueba: {X_test.shape[0]} muestras")

# Instanciar y entrenar el modelo de regresión lineal
print("\n[MODEL] Instanciando modelo: LinearRegression()")
# ⭐ SCIKIT-LEARN: LinearRegression() - Crea instancia del modelo de regresión lineal
modelo_lr = LinearRegression()

print("[MODEL] Entrenando modelo: modelo.fit(X_train, y_train)")
# ⭐ SCIKIT-LEARN: fit() - Entrena el modelo con los datos de entrenamiento
modelo_lr.fit(X_train, y_train)

# Predecir
print("[MODEL] Prediciendo: modelo.predict(X_test)")
# ⭐ SCIKIT-LEARN: predict() - Genera predicciones con el modelo entrenado
y_pred = modelo_lr.predict(X_test)

# Métricas
# ⭐ SCIKIT-LEARN: mean_squared_error() - Calcula el error cuadrático medio
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
# ⭐ SCIKIT-LEARN: r2_score() - Calcula el coeficiente de determinación R²
r2 = r2_score(y_test, y_pred)

print("\n[METRICS] Métricas del Modelo:")
print(f"   MSE (Mean Squared Error): {mse:.2f}")
print(f"   RMSE (Root Mean Squared Error): {rmse:.2f}")
print(f"   R² (Coeficiente de determinación): {r2:.4f}")

print("\n[MODEL] Coeficientes del modelo:")
print(f"   Intercepto (b): {modelo_lr.intercept_:.2f}")
for i, feature in enumerate(X_features):
    print(f"   {feature}: {modelo_lr.coef_[i]:.4f}")

# Ecuación del modelo
print("\n[MODEL] Ecuación del modelo: y = b + m1*x1 + m2*x2 + m3*x3")
print(f"   Purchase Amount = {modelo_lr.intercept_:.2f} + "
      f"{modelo_lr.coef_[0]:.4f}*Age + "
      f"{modelo_lr.coef_[1]:.4f}*Review_Rating + "
      f"{modelo_lr.coef_[2]:.4f}*Previous_Purchases")

# ============================================================================
# 4. REGRESIÓN LOGÍSTICA: Clasificación Binaria
# ============================================================================
print("\n" + "="*80)
print("4. REGRESIÓN LOGÍSTICA: Clasificación Binaria")
print("="*80)

# Clasificar Subscription Status (Yes/No)
if 'Subscription Status' in df_clean.columns:
    print("\n[MODEL] Objetivo: Predecir Subscription Status (Yes/No)")
    
    # Preparar datos
    # Variables independientes: Age, Purchase Amount, Review Rating, Previous Purchases
    X_log = df_clean[['Age', 'Purchase Amount (USD)', 'Review Rating', 'Previous Purchases']].values
    
    # ========================================================================
    # USO DE SCIKIT-LEARN: Regresión Logística y Preprocesamiento
    # ========================================================================
    # Variable dependiente binaria: Subscription Status
    # ⭐ SCIKIT-LEARN: LabelEncoder() - Convierte etiquetas categóricas a numéricas
    le = LabelEncoder()
    # ⭐ SCIKIT-LEARN: fit_transform() - Ajusta y transforma en un solo paso
    y_log = le.fit_transform(df_clean['Subscription Status'])
    # 1 = Yes, 0 = No
    
    print(f"   Distribución de clases:")
    print(f"   No (0): {np.sum(y_log == 0)}")
    print(f"   Yes (1): {np.sum(y_log == 1)}")
    
    # Dividir datos
    # ⭐ SCIKIT-LEARN: train_test_split() - Divide datos con estratificación para balancear clases
    X_log_train, X_log_test, y_log_train, y_log_test = train_test_split(
        X_log, y_log, test_size=0.2, random_state=42, stratify=y_log
    )
    
    # Estandarizar datos (importante para regresión logística)
    # ⭐ SCIKIT-LEARN: StandardScaler() - Estandariza características (media=0, std=1)
    scaler = StandardScaler()
    # ⭐ SCIKIT-LEARN: fit_transform() - Ajusta el escalador y transforma datos de entrenamiento
    X_log_train_scaled = scaler.fit_transform(X_log_train)
    # ⭐ SCIKIT-LEARN: transform() - Transforma datos de prueba usando parámetros del entrenamiento
    X_log_test_scaled = scaler.transform(X_log_test)
    
    # Instanciar y entrenar modelo
    print("\n[MODEL] Instanciando modelo: LogisticRegression()")
    # ⭐ SCIKIT-LEARN: LogisticRegression() - Crea instancia del modelo de clasificación binaria
    log_reg = LogisticRegression(random_state=42, max_iter=1000)
    
    print("[MODEL] Entrenando modelo: log_reg.fit(X_train, y_train)")
    # ⭐ SCIKIT-LEARN: fit() - Entrena el modelo de regresión logística
    log_reg.fit(X_log_train_scaled, y_log_train)
    
    # Predecir
    print("[MODEL] Prediciendo: log_reg.predict(X_test)")
    # ⭐ SCIKIT-LEARN: predict() - Genera predicciones de clase (0 o 1)
    y_log_pred = log_reg.predict(X_log_test_scaled)
    
    # Probabilidades (función sigmoide)
    print("[MODEL] Obteniendo probabilidades: log_reg.predict_proba(X_test)")
    # ⭐ SCIKIT-LEARN: predict_proba() - Obtiene probabilidades de cada clase (función sigmoide)
    y_log_proba = log_reg.predict_proba(X_log_test_scaled)
    
    print(f"\n   Ejemplo de probabilidades (primeras 5 muestras):")
    for i in range(min(5, len(y_log_proba))):
        print(f"   Muestra {i+1}: P(No)={y_log_proba[i][0]:.4f}, P(Yes)={y_log_proba[i][1]:.4f} -> Predicción: {y_log_pred[i]}")
    
    # Umbral por defecto es 0.5
    print(f"\n   Umbral (Threshold): 0.5 (por defecto)")
    print(f"   Si probabilidad > 0.5 -> Clase 1 (Yes), sino -> Clase 0 (No)")
    
    # Métricas
    # ⭐ SCIKIT-LEARN: accuracy_score() - Calcula la precisión del modelo
    accuracy = accuracy_score(y_log_test, y_log_pred)
    print(f"\n[METRICS] Métricas del Modelo:")
    print(f"   Accuracy: {accuracy:.4f}")
    
    print("\n[METRICS] Matriz de Confusión:")
    # ⭐ SCIKIT-LEARN: confusion_matrix() - Genera matriz de confusión (TP, TN, FP, FN)
    cm = confusion_matrix(y_log_test, y_log_pred)
    print(cm)
    
    print("\n[METRICS] Reporte de Clasificación:")
    # ⭐ SCIKIT-LEARN: classification_report() - Genera reporte completo con precision, recall, f1-score
    print(classification_report(y_log_test, y_log_pred, target_names=['No', 'Yes']))

# ============================================================================
# 5. VISUALIZACIONES SEPARADAS POR PROCESO
# ============================================================================
print("\n" + "="*80)
print("5. GENERANDO VISUALIZACIONES")
print("="*80)

# Ya importado arriba

# ============================================================================
# 5.1. VISUALIZACIONES: ESTADÍSTICA DESCRIPTIVA
# ============================================================================
print("\n[INFO] Generando visualizaciones de Estadística Descriptiva...")

fig1 = plt.figure(figsize=(18, 12))
fig1.suptitle('ANÁLISIS ESTADÍSTICO DESCRIPTIVO - Shopping Behavior Dataset', 
              fontsize=16, fontweight='bold', y=0.995)

# 5.1.1. Histograma de Purchase Amount con estadísticas
ax1 = plt.subplot(2, 3, 1)
n, bins, patches = plt.hist(purchase_amount, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
plt.xlabel('Purchase Amount (USD)', fontsize=11, fontweight='bold')
plt.ylabel('Frecuencia', fontsize=11, fontweight='bold')
plt.title('Distribución de Montos de Compra\n(Histograma de Frecuencias)', 
          fontsize=12, fontweight='bold')
# ⭐ NUMPY: np.mean() - Calcula el centro de los datos (media del monto de compra)
mean_pa = np.mean(purchase_amount)
median_pa = np.median(purchase_amount)
# ⭐ NUMPY: np.std() - Mide la dispersión o "ruido" en los datos (desviación estándar)
std_pa = np.std(purchase_amount)
plt.axvline(mean_pa, color='red', linestyle='--', linewidth=2, label=f'Media: ${mean_pa:.2f}')
plt.axvline(median_pa, color='green', linestyle='--', linewidth=2, label=f'Mediana: ${median_pa:.2f}')
plt.axvline(mean_pa + std_pa, color='orange', linestyle=':', alpha=0.7, label=f'±1σ: ${std_pa:.2f}')
plt.axvline(mean_pa - std_pa, color='orange', linestyle=':', alpha=0.7)
plt.legend(fontsize=9)
plt.grid(True, alpha=0.3, axis='y')
# Agregar texto con estadísticas
textstr = f'Min: ${np.min(purchase_amount):.2f}\nMax: ${np.max(purchase_amount):.2f}\nRango: ${np.max(purchase_amount)-np.min(purchase_amount):.2f}'
plt.text(0.02, 0.98, textstr, transform=ax1.transAxes, fontsize=9,
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# 5.1.2. Histograma de Age con estadísticas
ax2 = plt.subplot(2, 3, 2)
plt.hist(age, bins=30, edgecolor='black', alpha=0.7, color='coral')
plt.xlabel('Edad (Años)', fontsize=11, fontweight='bold')
plt.ylabel('Frecuencia', fontsize=11, fontweight='bold')
plt.title('Distribución de Edades de Clientes\n(Análisis Demográfico)', 
          fontsize=12, fontweight='bold')
# ⭐ NUMPY: np.mean() - Calcula el centro de los datos (edad promedio)
mean_age = np.mean(age)
# ⭐ NUMPY: np.std() - Mide la dispersión en las edades
std_age = np.std(age)
plt.axvline(mean_age, color='red', linestyle='--', linewidth=2, label=f'Media: {mean_age:.1f} años')
plt.axvline(mean_age + std_age, color='orange', linestyle=':', alpha=0.7, label=f'±1σ: {std_age:.1f}')
plt.axvline(mean_age - std_age, color='orange', linestyle=':', alpha=0.7)
plt.legend(fontsize=9)
plt.grid(True, alpha=0.3, axis='y')
textstr = f'Rango: {np.min(age)}-{np.max(age)} años\nModa: {stats.mode(age, keepdims=True)[0][0]} años'
plt.text(0.02, 0.98, textstr, transform=ax2.transAxes, fontsize=9,
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

# 5.1.3. Boxplot de Purchase Amount por Category (mejorado)
ax3 = plt.subplot(2, 3, 3)
if 'Category' in df_clean.columns:
    categories = df_clean['Category'].unique()
    data_to_plot = [df_clean[df_clean['Category'] == cat]['Purchase Amount (USD)'].values 
                    for cat in categories]
    bp = plt.boxplot(data_to_plot, labels=categories, patch_artist=True)
    for patch, color in zip(bp['boxes'], ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow']):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    plt.xlabel('Categoría de Producto', fontsize=11, fontweight='bold')
    plt.ylabel('Purchase Amount (USD)', fontsize=11, fontweight='bold')
    plt.title('Comparación de Precios por Categoría\n(Boxplot con Mediana y Cuartiles)', 
              fontsize=12, fontweight='bold')
    plt.xticks(rotation=15, ha='right')
    plt.grid(True, alpha=0.3, axis='y')
    # Agregar medias
    for i, cat in enumerate(categories):
        mean_val = df_clean[df_clean['Category'] == cat]['Purchase Amount (USD)'].mean()
        plt.plot(i+1, mean_val, 'rD', markersize=8, label='Media' if i == 0 else '')
    if len(categories) > 0:
        plt.legend(['Media'], fontsize=9)

# 5.1.4. Scatter Plot: Age vs Purchase Amount con línea de tendencia
ax4 = plt.subplot(2, 3, 4)
plt.scatter(age, purchase_amount, alpha=0.4, s=15, c='purple', edgecolors='black', linewidth=0.5)
plt.xlabel('Edad del Cliente (Años)', fontsize=11, fontweight='bold')
plt.ylabel('Purchase Amount (USD)', fontsize=11, fontweight='bold')
plt.title('Relación entre Edad y Monto de Compra\n(Análisis de Correlación)', 
          fontsize=12, fontweight='bold')
# Calcular correlación
correlation = np.corrcoef(age, purchase_amount)[0, 1]
# Línea de tendencia
z = np.polyfit(age, purchase_amount, 1)
p = np.poly1d(z)
plt.plot(age, p(age), "r--", alpha=0.8, linewidth=2, label=f'Tendencia (r={correlation:.3f})')
plt.legend(fontsize=9)
plt.grid(True, alpha=0.3)
textstr = f'Correlación: {correlation:.3f}\n{"Positiva débil" if 0 < correlation < 0.3 else "Negativa débil" if -0.3 < correlation < 0 else "Muy débil"}'
plt.text(0.02, 0.98, textstr, transform=ax4.transAxes, fontsize=9,
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

# 5.1.5. Distribución de Review Rating
ax5 = plt.subplot(2, 3, 5)
plt.hist(review_rating, bins=30, edgecolor='black', alpha=0.7, color='mediumseagreen')
plt.xlabel('Review Rating (1-5)', fontsize=11, fontweight='bold')
plt.ylabel('Frecuencia', fontsize=11, fontweight='bold')
plt.title('Distribución de Calificaciones\n(Satisfacción del Cliente)', 
          fontsize=12, fontweight='bold')
# ⭐ NUMPY: np.mean() - Calcula el centro de los datos (calificación promedio)
mean_rating = np.mean(review_rating)
plt.axvline(mean_rating, color='red', linestyle='--', linewidth=2, label=f'Media: {mean_rating:.2f}')
plt.axvline(5, color='gold', linestyle=':', alpha=0.5, label='Máximo: 5.0')
plt.axvline(1, color='gray', linestyle=':', alpha=0.5, label='Mínimo: 1.0')
plt.legend(fontsize=9)
plt.grid(True, alpha=0.3, axis='y')
# ⭐ NUMPY: np.std() - Mide la dispersión en las calificaciones
textstr = f'Promedio: {mean_rating:.2f}/5.0\nDesv. Est: {np.std(review_rating):.2f}'
plt.text(0.02, 0.98, textstr, transform=ax5.transAxes, fontsize=9,
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

# 5.1.6. Comparación de Purchase Amount por Gender
ax6 = plt.subplot(2, 3, 6)
if 'Gender' in df_clean.columns:
    genders = df_clean['Gender'].unique()
    gender_data = [df_clean[df_clean['Gender'] == g]['Purchase Amount (USD)'].values 
                   for g in genders]
    bp = plt.boxplot(gender_data, labels=genders, patch_artist=True)
    colors = ['pink', 'lightblue']
    for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    plt.xlabel('Género', fontsize=11, fontweight='bold')
    plt.ylabel('Purchase Amount (USD)', fontsize=11, fontweight='bold')
    plt.title('Comparación de Gastos por Género\n(Análisis de Segmentación)', 
              fontsize=12, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    # Agregar medias
    for i, gen in enumerate(genders):
        mean_val = df_clean[df_clean['Gender'] == gen]['Purchase Amount (USD)'].mean()
        plt.plot(i+1, mean_val, 'rD', markersize=8)
    # Estadísticas
    stats_text = '\n'.join([f'{g}: μ=${df_clean[df_clean["Gender"]==g]["Purchase Amount (USD)"].mean():.2f}' 
                            for g in genders])
    plt.text(0.02, 0.98, stats_text, transform=ax6.transAxes, fontsize=9,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout(rect=[0, 0, 1, 0.99])
plt.savefig('01_estadistica_descriptiva.png', dpi=300, bbox_inches='tight')
print("[OK] Visualización 1 guardada: 01_estadistica_descriptiva.png")
plt.close()

# ============================================================================
# 5.2. VISUALIZACIONES: REGRESIÓN LINEAL
# ============================================================================
print("[INFO] Generando visualizaciones de Regresión Lineal...")

fig2 = plt.figure(figsize=(18, 12))
fig2.suptitle('REGRESIÓN LINEAL: Predicción de Purchase Amount (USD)', 
              fontsize=16, fontweight='bold', y=0.995)

# Calcular residuos
residuos = y_test - y_pred

# 5.2.1. Predicciones vs Valores Reales (mejorado)
ax1 = plt.subplot(2, 3, 1)
plt.scatter(y_test, y_pred, alpha=0.5, s=30, c='steelblue', edgecolors='darkblue', linewidth=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=3, 
         label='Línea de Predicción Perfecta')
plt.xlabel('Valores Reales (USD)', fontsize=11, fontweight='bold')
plt.ylabel('Valores Predichos (USD)', fontsize=11, fontweight='bold')
plt.title(f'Evaluación del Modelo: Predicciones vs Reales\nR² = {r2:.4f}', 
          fontsize=12, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
# Agregar texto con métricas
textstr = f'MSE: {mse:.2f}\nRMSE: ${rmse:.2f}\nR²: {r2:.4f}\n{"Modelo útil" if r2 > 0 else "Modelo no útil"}'
plt.text(0.05, 0.95, textstr, transform=ax1.transAxes, fontsize=10,
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

# 5.2.2. GRÁFICO DE RESIDUOS (mejorado - importante según la guía)
ax2 = plt.subplot(2, 3, 2)
plt.scatter(y_pred, residuos, alpha=0.5, s=30, c='purple', edgecolors='darkviolet', linewidth=0.5)
plt.axhline(y=0, color='red', linestyle='--', linewidth=2, label='Residuo = 0')
plt.axhline(y=np.mean(residuos) + 2*np.std(residuos), color='orange', linestyle=':', 
            alpha=0.7, label='±2σ')
plt.axhline(y=np.mean(residuos) - 2*np.std(residuos), color='orange', linestyle=':', alpha=0.7)
plt.xlabel('Valores Predichos (USD)', fontsize=11, fontweight='bold')
plt.ylabel('Residuos (Real - Predicción)', fontsize=11, fontweight='bold')
plt.title('Análisis de Residuos\n(Validación de Supuestos del Modelo)', 
          fontsize=12, fontweight='bold')
plt.legend(fontsize=9)
plt.grid(True, alpha=0.3)
# Análisis de residuos
# ⭐ NUMPY: np.mean() - Calcula el centro de los residuos (debe ser cercano a 0)
mean_res = np.mean(residuos)
# ⭐ NUMPY: np.std() - Mide la dispersión de los residuos
std_res = np.std(residuos)
textstr = f'Media residuos: {mean_res:.2f}\nDesv. Est: {std_res:.2f}\n{"✓ Patrón aleatorio" if abs(mean_res) < 1 else "✗ Sesgo detectado"}'
plt.text(0.05, 0.95, textstr, transform=ax2.transAxes, fontsize=9,
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))

# 5.2.3. Histograma de Residuos (mejorado)
ax3 = plt.subplot(2, 3, 3)
n, bins, patches = plt.hist(residuos, bins=50, edgecolor='black', alpha=0.7, color='purple')
plt.xlabel('Residuos (Error del Modelo)', fontsize=11, fontweight='bold')
plt.ylabel('Frecuencia', fontsize=11, fontweight='bold')
plt.title('Distribución de Errores del Modelo\n(Análisis de Normalidad)', 
          fontsize=12, fontweight='bold')
plt.axvline(mean_res, color='red', linestyle='--', linewidth=2, label=f'Media: {mean_res:.2f}')
plt.axvline(0, color='green', linestyle='-', linewidth=1, alpha=0.5, label='Residuo = 0')
plt.legend(fontsize=9)
plt.grid(True, alpha=0.3, axis='y')
# Test de normalidad aproximado
if abs(mean_res) < 0.1 and abs(np.median(residuos)) < 0.1:
    normalidad = "✓ Aprox. Normal"
else:
    normalidad = "✗ No normal"
textstr = f'{normalidad}\nMedia: {mean_res:.3f}\nMediana: {np.median(residuos):.3f}'
plt.text(0.05, 0.95, textstr, transform=ax3.transAxes, fontsize=9,
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lavender', alpha=0.7))

# 5.2.4. Relación Age vs Purchase Amount con línea de regresión
ax4 = plt.subplot(2, 3, 4)
plt.scatter(age, purchase_amount, alpha=0.3, s=10, c='steelblue')
plt.xlabel('Edad (Años)', fontsize=11, fontweight='bold')
plt.ylabel('Purchase Amount (USD)', fontsize=11, fontweight='bold')
plt.title('Variable Independiente: Edad\n(Relación con Variable Dependiente)', 
          fontsize=12, fontweight='bold')
# Línea de regresión
z_age = np.polyfit(age, purchase_amount, 1)
p_age = np.poly1d(z_age)
plt.plot(age, p_age(age), "r--", alpha=0.8, linewidth=2, 
         label=f'y = {z_age[0]:.2f}x + {z_age[1]:.2f}')
plt.legend(fontsize=9)
plt.grid(True, alpha=0.3)

# 5.2.5. Relación Review Rating vs Purchase Amount
ax5 = plt.subplot(2, 3, 5)
plt.scatter(review_rating, purchase_amount, alpha=0.3, s=10, c='green')
plt.xlabel('Review Rating', fontsize=11, fontweight='bold')
plt.ylabel('Purchase Amount (USD)', fontsize=11, fontweight='bold')
plt.title('Variable Independiente: Review Rating\n(Relación con Variable Dependiente)', 
          fontsize=12, fontweight='bold')
# Línea de regresión
z_rating = np.polyfit(review_rating, purchase_amount, 1)
p_rating = np.poly1d(z_rating)
plt.plot(review_rating, p_rating(review_rating), "r--", alpha=0.8, linewidth=2,
         label=f'y = {z_rating[0]:.2f}x + {z_rating[1]:.2f}')
plt.legend(fontsize=9)
plt.grid(True, alpha=0.3)

# 5.2.6. Coeficientes del Modelo (Bar Plot)
ax6 = plt.subplot(2, 3, 6)
coef_names = ['Intercepto'] + X_features
coef_values = [modelo_lr.intercept_] + list(modelo_lr.coef_)
colors_bar = ['red' if 'Intercepto' in name else 'steelblue' for name in coef_names]
bars = plt.barh(coef_names, coef_values, color=colors_bar, alpha=0.7, edgecolor='black')
plt.xlabel('Valor del Coeficiente', fontsize=11, fontweight='bold')
plt.title('Coeficientes del Modelo de Regresión\n(Ecuación: y = b + m₁x₁ + m₂x₂ + m₃x₃)', 
          fontsize=12, fontweight='bold')
plt.grid(True, alpha=0.3, axis='x')
# Agregar valores en las barras
for i, (bar, val) in enumerate(zip(bars, coef_values)):
    plt.text(val, i, f' {val:.4f}', va='center', fontsize=9, fontweight='bold')
# Ecuación completa
eq_text = f'y = {modelo_lr.intercept_:.2f} + {modelo_lr.coef_[0]:.4f}*Age + {modelo_lr.coef_[1]:.4f}*Rating + {modelo_lr.coef_[2]:.4f}*PrevPurch'
plt.text(0.05, 0.02, eq_text, transform=ax6.transAxes, fontsize=8,
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

plt.tight_layout(rect=[0, 0, 1, 0.99])
plt.savefig('02_regresion_lineal.png', dpi=300, bbox_inches='tight')
print("[OK] Visualización 2 guardada: 02_regresion_lineal.png")
plt.close()

# ============================================================================
# 5.3. VISUALIZACIONES: REGRESIÓN LOGÍSTICA
# ============================================================================
print("[INFO] Generando visualizaciones de Regresión Logística...")

if 'Subscription Status' in df_clean.columns:
    fig3 = plt.figure(figsize=(18, 12))
    fig3.suptitle('REGRESIÓN LOGÍSTICA: Clasificación de Subscription Status', 
                  fontsize=16, fontweight='bold', y=0.995)
    
    prob_yes = y_log_proba[:, 1]
    prob_no = y_log_proba[:, 0]
    
    # 5.3.1. Distribución de Probabilidades (Función Sigmoide)
    ax1 = plt.subplot(2, 3, 1)
    plt.hist(prob_yes, bins=50, edgecolor='black', alpha=0.7, color='green', label='P(Yes)')
    plt.hist(prob_no, bins=50, edgecolor='black', alpha=0.5, color='red', label='P(No)')
    plt.axvline(x=0.5, color='blue', linestyle='--', linewidth=2, label='Umbral de Decisión (0.5)')
    plt.xlabel('Probabilidad', fontsize=11, fontweight='bold')
    plt.ylabel('Frecuencia', fontsize=11, fontweight='bold')
    plt.title('Distribución de Probabilidades (Función Sigmoide)\nP(Yes) y P(No) para cada Cliente', 
              fontsize=12, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3, axis='y')
    # Estadísticas
    # ⭐ NUMPY: np.mean() - Calcula el centro de las probabilidades
    mean_prob_yes = np.mean(prob_yes)
    textstr = f'Media P(Yes): {mean_prob_yes:.3f}\nPredicciones Yes: {np.sum(y_log_pred == 1)}\nPredicciones No: {np.sum(y_log_pred == 0)}'
    plt.text(0.05, 0.95, textstr, transform=ax1.transAxes, fontsize=9,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    
    # 5.3.2. Matriz de Confusión (mejorada)
    ax2 = plt.subplot(2, 3, 2)
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax2, 
                cbar_kws={'label': 'Cantidad de Casos'}, linewidths=1, linecolor='black')
    ax2.set_xlabel('Predicción del Modelo', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Valor Real', fontsize=11, fontweight='bold')
    ax2.set_title('Matriz de Confusión\n(TP, TN, FP, FN)', fontsize=12, fontweight='bold')
    ax2.set_xticklabels(['No (0)', 'Yes (1)'])
    ax2.set_yticklabels(['No (0)', 'Yes (1)'])
    # Agregar anotaciones
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm[0,0], cm[0,1], cm[1,0], cm[1,1]
        textstr = f'TP (Verdaderos Pos): {tp}\nTN (Verdaderos Neg): {tn}\nFP (Falsos Pos): {fp}\nFN (Falsos Neg): {fn}'
        plt.text(1.3, 0.5, textstr, transform=ax2.transAxes, fontsize=9,
                 verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    # 5.3.3. Curva ROC simplificada (Probabilidades ordenadas)
    ax3 = plt.subplot(2, 3, 3)
    sorted_probs = np.sort(prob_yes)[::-1]
    sorted_labels = y_log_test[np.argsort(prob_yes)[::-1]]
    cumulative_tp = np.cumsum(sorted_labels)
    cumulative_fp = np.cumsum(1 - sorted_labels)
    plt.plot(cumulative_fp / len(y_log_test), cumulative_tp / np.sum(y_log_test), 
             'b-', linewidth=2, label='Curva de Probabilidades')
    plt.plot([0, 1], [0, 1], 'r--', linewidth=1, label='Línea Base (Random)')
    plt.xlabel('Tasa de Falsos Positivos', fontsize=11, fontweight='bold')
    plt.ylabel('Tasa de Verdaderos Positivos', fontsize=11, fontweight='bold')
    plt.title('Distribución de Probabilidades Ordenadas\n(Análisis de Clasificación)', 
              fontsize=12, fontweight='bold')
    plt.legend(fontsize=9)
    plt.grid(True, alpha=0.3)
    
    # 5.3.4. Comparación de características por clase
    ax4 = plt.subplot(2, 3, 4)
    if len(y_log) > 0:
        # Comparar Age entre clases
        age_yes = df_clean.iloc[np.where(y_log == 1)[0]]['Age'].values
        age_no = df_clean.iloc[np.where(y_log == 0)[0]]['Age'].values
        data_age = [age_no, age_yes]
        bp = plt.boxplot(data_age, labels=['Sin Suscripción', 'Con Suscripción'], patch_artist=True)
        bp['boxes'][0].set_facecolor('lightcoral')
        bp['boxes'][1].set_facecolor('lightgreen')
        for patch in bp['boxes']:
            patch.set_alpha(0.7)
        plt.ylabel('Edad (Años)', fontsize=11, fontweight='bold')
        plt.title('Comparación de Edades\n(Suscripción vs No Suscripción)', 
                  fontsize=12, fontweight='bold')
        plt.grid(True, alpha=0.3, axis='y')
    
    # 5.3.5. Comparación de Purchase Amount por clase
    ax5 = plt.subplot(2, 3, 5)
    if len(y_log) > 0:
        pa_yes = df_clean.iloc[np.where(y_log == 1)[0]]['Purchase Amount (USD)'].values
        pa_no = df_clean.iloc[np.where(y_log == 0)[0]]['Purchase Amount (USD)'].values
        data_pa = [pa_no, pa_yes]
        bp = plt.boxplot(data_pa, labels=['Sin Suscripción', 'Con Suscripción'], patch_artist=True)
        bp['boxes'][0].set_facecolor('lightcoral')
        bp['boxes'][1].set_facecolor('lightgreen')
        for patch in bp['boxes']:
            patch.set_alpha(0.7)
        plt.ylabel('Purchase Amount (USD)', fontsize=11, fontweight='bold')
        plt.title('Comparación de Gastos\n(Suscripción vs No Suscripción)', 
                  fontsize=12, fontweight='bold')
        plt.grid(True, alpha=0.3, axis='y')
        # Estadísticas
        # ⭐ NUMPY: np.mean() - Calcula el centro de los datos (media de gastos)
        mean_yes = np.mean(pa_yes) if len(pa_yes) > 0 else 0
        mean_no = np.mean(pa_no) if len(pa_no) > 0 else 0
        textstr = f'Media con Suscripción:\n${mean_yes:.2f}\nMedia sin Suscripción:\n${mean_no:.2f}'
        plt.text(0.02, 0.98, textstr, transform=ax5.transAxes, fontsize=9,
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    # 5.3.6. Métricas del Modelo
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    # Calcular métricas adicionales
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm[0,0], cm[0,1], cm[1,0], cm[1,1]
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        metrics_text = f"""
        MÉTRICAS DEL MODELO
        
        Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)
        
        Precision: {precision:.4f}
        (De los predichos como Yes, 
        cuántos realmente eran Yes)
        
        Recall (Sensitivity): {recall:.4f}
        (De los realmente Yes,
        cuántos predijo correctamente)
        
        Specificity: {specificity:.4f}
        (De los realmente No,
        cuántos predijo correctamente)
        
        F1-Score: {f1:.4f}
        (Media armónica de
        Precision y Recall)
        
        Umbral de Decisión: 0.5
        (Probabilidad > 0.5 → Yes)
        """
        ax6.text(0.1, 0.5, metrics_text, transform=ax6.transAxes, fontsize=11,
                verticalalignment='center', family='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.savefig('03_regresion_logistica.png', dpi=300, bbox_inches='tight')
    print("[OK] Visualización 3 guardada: 03_regresion_logistica.png")
    plt.close()

print("\n[OK] Todas las visualizaciones generadas exitosamente:")
print("   - 01_estadistica_descriptiva.png")
print("   - 02_regresion_lineal.png")
print("   - 03_regresion_logistica.png")

# ============================================================================
# 6. ANÁLISIS ADICIONAL: Estadísticas por grupos
# ============================================================================
print("\n" + "="*80)
print("6. ANÁLISIS ADICIONAL: Estadísticas por Grupos")
print("="*80)

# Estadísticas por Season
if 'Season' in df_clean.columns:
    print("\n[STATS] Estadísticas de Purchase Amount por Season:")
    # ⭐ PANDAS: df.groupby() - Segmenta estadísticas por temporada (Season)
    stats_season = df_clean.groupby('Season')['Purchase Amount (USD)'].agg(['mean', 'std', 'count'])
    print(stats_season)

# Estadísticas por Payment Method
if 'Payment Method' in df_clean.columns:
    print("\n[STATS] Estadísticas de Purchase Amount por Payment Method:")
    # ⭐ PANDAS: df.groupby() - Segmenta estadísticas por método de pago
    stats_payment = df_clean.groupby('Payment Method')['Purchase Amount (USD)'].agg(['mean', 'std', 'count'])
    print(stats_payment)

# ============================================================================
# RESUMEN FINAL
# ============================================================================
print("\n" + "="*80)
print("RESUMEN DEL ANÁLISIS")
print("="*80)
print("\n[OK] Análisis estadístico descriptivo completado")
print("[OK] Regresión lineal implementada y evaluada")
print("[OK] Regresión logística implementada y evaluada")
print("[OK] Visualizaciones generadas (incluyendo gráfico de residuos)")
print("[OK] Uso de scikit-learn demostrado")
print("\n[INFO] Archivos generados:")
print("   - 01_estadistica_descriptiva.png (6 gráficos de análisis descriptivo)")
print("   - 02_regresion_lineal.png (6 gráficos de regresión lineal)")
print("   - 03_regresion_logistica.png (6 gráficos de regresión logística)")

print("\n" + "="*80)
print("ANÁLISIS COMPLETADO")
print("="*80)
