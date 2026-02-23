# 📋 Documentación: Uso de Herramientas en el Análisis

Este documento señala específicamente dónde se utilizan cada una de las herramientas mencionadas en la guía.

---

## 🔢 NUMPY: `np.mean()` - Calcular el centro de los datos

**Uso Principal:** Calcular el centro de los datos (o brillo de imagen).

### Ubicaciones en el código:

1. **Línea 116:** Estadísticas de Purchase Amount
   ```python
   print(f"   Media (np.mean): {np.mean(purchase_amount):.2f}")
   ```

2. **Línea 124:** Estadísticas de Age
   ```python
   print(f"   Media (np.mean): {np.mean(age):.2f}")
   ```

3. **Línea 128:** Estadísticas de Review Rating
   ```python
   print(f"   Media (np.mean): {np.mean(review_rating):.2f}")
   ```

4. **Línea 285:** Visualización - Media de Purchase Amount
   ```python
   mean_pa = np.mean(purchase_amount)
   ```

5. **Línea 306:** Visualización - Media de Age
   ```python
   mean_age = np.mean(age)
   ```

6. **Línea 366:** Visualización - Media de Review Rating
   ```python
   mean_rating = np.mean(review_rating)
   ```

7. **Línea 449:** Análisis de Residuos - Media de residuos
   ```python
   mean_res = np.mean(residuos)
   ```

8. **Línea 553:** Regresión Logística - Media de probabilidades
   ```python
   mean_prob_yes = np.mean(prob_yes)
   ```

9. **Línea 624:** Comparación de gastos - Media con suscripción
   ```python
   mean_yes = np.mean(pa_yes) if len(pa_yes) > 0 else 0
   ```

10. **Línea 625:** Comparación de gastos - Media sin suscripción
    ```python
    mean_no = np.mean(pa_no) if len(pa_no) > 0 else 0
    ```

**Total de usos:** 10 instancias

---

## 📊 NUMPY: `np.std()` - Medir la dispersión o "ruido" en los datos

**Uso Principal:** Medir la dispersión o "ruido" en los datos.

### Ubicaciones en el código:

1. **Línea 118:** Estadísticas de Purchase Amount
   ```python
   print(f"   Desviación estándar (np.std): {np.std(purchase_amount):.2f}")
   ```

2. **Línea 125:** Estadísticas de Age
   ```python
   print(f"   Desviación estándar (np.std): {np.std(age):.2f}")
   ```

3. **Línea 129:** Estadísticas de Review Rating
   ```python
   print(f"   Desviación estándar (np.std): {np.std(review_rating):.2f}")
   ```

4. **Línea 287:** Visualización - Desviación estándar de Purchase Amount
   ```python
   std_pa = np.std(purchase_amount)
   ```

5. **Línea 307:** Visualización - Desviación estándar de Age
   ```python
   std_age = np.std(age)
   ```

6. **Línea 372:** Visualización - Desviación estándar de Review Rating
   ```python
   textstr = f'Promedio: {mean_rating:.2f}/5.0\nDesv. Est: {np.std(review_rating):.2f}'
   ```

7. **Línea 439-440:** Análisis de Residuos - Límites de ±2σ
   ```python
   plt.axhline(y=np.mean(residuos) + 2*np.std(residuos), ...)
   plt.axhline(y=np.mean(residuos) - 2*np.std(residuos), ...)
   ```

8. **Línea 450:** Análisis de Residuos - Desviación estándar de residuos
   ```python
   std_res = np.std(residuos)
   ```

**Total de usos:** 8 instancias

---

## 🗂️ PANDAS: `df.groupby()` - Segmentar estadísticas por categorías

**Uso Principal:** Segmentar estadísticas por categorías.

### Ubicaciones en el código:

1. **Línea 91:** Estadísticas por Category
   ```python
   stats_by_category = df_clean.groupby('Category')['Purchase Amount (USD)'].agg(['mean', 'std', 'count'])
   ```
   - **Propósito:** Agrupa los datos por categoría de producto y calcula estadísticas (media, desviación estándar, conteo) del monto de compra para cada categoría.

2. **Línea 99:** Estadísticas por Gender
   ```python
   stats_by_gender = df_clean.groupby('Gender')['Purchase Amount (USD)'].agg(['mean', 'std', 'count'])
   ```
   - **Propósito:** Agrupa los datos por género y calcula estadísticas del monto de compra para cada género.

3. **Línea 689:** Estadísticas por Season
   ```python
   stats_season = df_clean.groupby('Season')['Purchase Amount (USD)'].agg(['mean', 'std', 'count'])
   ```
   - **Propósito:** Agrupa los datos por temporada (Season) y calcula estadísticas del monto de compra para cada temporada.

4. **Línea 695:** Estadísticas por Payment Method
   ```python
   stats_payment = df_clean.groupby('Payment Method')['Purchase Amount (USD)'].agg(['mean', 'std', 'count'])
   ```
   - **Propósito:** Agrupa los datos por método de pago y calcula estadísticas del monto de compra para cada método.

**Total de usos:** 4 instancias

**Ejemplo de salida:**
```
                  mean        std  count
Category                                
Accessories  59.838710  23.301230   1240
Clothing     60.025331  23.792460   1737
Footwear     60.255426  23.638439    599
Outerwear    57.172840  24.590033    324
```

---

## 🤖 SCIKIT-LEARN: Librería de Machine Learning

**Uso Principal:** Es imperativo que utilicen esta librería - se demuestra conocimiento completo de scikit-learn.

### Módulos y Métodos Utilizados:

#### 1. **`sklearn.model_selection.train_test_split()`**
   - **Línea 153:** División de datos para regresión lineal
     ```python
     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
     ```
   - **Línea 216:** División de datos para regresión logística (con estratificación)
     ```python
     X_log_train, X_log_test, y_log_train, y_log_test = train_test_split(
         X_log, y_log, test_size=0.2, random_state=42, stratify=y_log
     )
     ```
   - **Propósito:** Divide los datos en conjuntos de entrenamiento y prueba para validar el modelo.

#### 2. **`sklearn.linear_model.LinearRegression()`**
   - **Línea 160:** Instanciación del modelo
     ```python
     modelo_lr = LinearRegression()
     ```
   - **Línea 163:** Entrenamiento del modelo
     ```python
     modelo_lr.fit(X_train, y_train)
     ```
   - **Línea 167:** Predicción
     ```python
     y_pred = modelo_lr.predict(X_test)
     ```
   - **Propósito:** Modelo de regresión lineal para predecir valores continuos (Purchase Amount).

#### 3. **`sklearn.linear_model.LogisticRegression()`**
   - **Línea 227:** Instanciación del modelo
     ```python
     log_reg = LogisticRegression(random_state=42, max_iter=1000)
     ```
   - **Línea 230:** Entrenamiento del modelo
     ```python
     log_reg.fit(X_log_train_scaled, y_log_train)
     ```
   - **Línea 234:** Predicción de clases
     ```python
     y_log_pred = log_reg.predict(X_log_test_scaled)
     ```
   - **Línea 238:** Obtención de probabilidades (función sigmoide)
     ```python
     y_log_proba = log_reg.predict_proba(X_log_test_scaled)
     ```
   - **Propósito:** Modelo de clasificación binaria para predecir Subscription Status (Yes/No).

#### 4. **`sklearn.preprocessing.StandardScaler()`**
   - **Línea 221:** Instanciación del escalador
     ```python
     scaler = StandardScaler()
     ```
   - **Línea 222:** Ajuste y transformación de datos de entrenamiento
     ```python
     X_log_train_scaled = scaler.fit_transform(X_log_train)
     ```
   - **Línea 223:** Transformación de datos de prueba
     ```python
     X_log_test_scaled = scaler.transform(X_log_test)
     ```
   - **Propósito:** Estandariza las características (media=0, desviación estándar=1) para mejorar el rendimiento de la regresión logística.

#### 5. **`sklearn.preprocessing.LabelEncoder()`**
   - **Línea 207:** Instanciación del codificador
     ```python
     le = LabelEncoder()
     ```
   - **Línea 208:** Codificación de etiquetas categóricas
     ```python
     y_log = le.fit_transform(df_clean['Subscription Status'])
     ```
   - **Propósito:** Convierte etiquetas categóricas (Yes/No) a valores numéricos (1/0).

#### 6. **`sklearn.metrics.mean_squared_error()`**
   - **Línea 170:** Cálculo del error cuadrático medio
     ```python
     mse = mean_squared_error(y_test, y_pred)
     ```
   - **Propósito:** Evalúa el error del modelo de regresión lineal.

#### 7. **`sklearn.metrics.r2_score()`**
   - **Línea 172:** Cálculo del coeficiente de determinación
     ```python
     r2 = r2_score(y_test, y_pred)
     ```
   - **Propósito:** Mide qué tan bien el modelo explica la varianza de los datos.

#### 8. **`sklearn.metrics.accuracy_score()`**
   - **Línea 249:** Cálculo de la precisión del modelo
     ```python
     accuracy = accuracy_score(y_log_test, y_log_pred)
     ```
   - **Propósito:** Evalúa la precisión del modelo de clasificación.

#### 9. **`sklearn.metrics.confusion_matrix()`**
   - **Línea 254:** Generación de matriz de confusión
     ```python
     cm = confusion_matrix(y_log_test, y_log_pred)
     ```
   - **Propósito:** Muestra TP, TN, FP, FN para evaluar el rendimiento del clasificador.

#### 10. **`sklearn.metrics.classification_report()`**
   - **Línea 258:** Generación de reporte completo
     ```python
     print(classification_report(y_log_test, y_log_pred, target_names=['No', 'Yes']))
     ```
   - **Propósito:** Proporciona un reporte detallado con precision, recall, f1-score para cada clase.

---

## 📊 Resumen de Uso de Herramientas

| Herramienta | Método | Total de Usos | Ubicaciones Principales |
|------------|--------|---------------|------------------------|
| **NumPy** | `np.mean()` | 10 | Líneas 116, 124, 128, 285, 306, 366, 449, 553, 624, 625 |
| **NumPy** | `np.std()` | 8 | Líneas 118, 125, 129, 287, 307, 372, 439-440, 450 |
| **Pandas** | `df.groupby()` | 4 | Líneas 91, 99, 689, 695 |
| **Scikit-Learn** | Múltiples métodos | 15+ | Líneas 153, 160, 163, 167, 170, 172, 207, 208, 216, 221-223, 227, 230, 234, 238, 249, 254, 258 |

---

## 🎯 Conocimiento Demostrado de Scikit-Learn

El código demuestra conocimiento completo de scikit-learn en las siguientes áreas:

### ✅ **Preprocesamiento de Datos:**
- `StandardScaler()` - Estandarización de características
- `LabelEncoder()` - Codificación de etiquetas categóricas

### ✅ **División de Datos:**
- `train_test_split()` - Con y sin estratificación

### ✅ **Modelos de Machine Learning:**
- `LinearRegression()` - Regresión lineal
- `LogisticRegression()` - Clasificación binaria

### ✅ **Métodos de Modelos:**
- `fit()` - Entrenamiento
- `predict()` - Predicción
- `predict_proba()` - Probabilidades (función sigmoide)

### ✅ **Métricas de Evaluación:**
- `mean_squared_error()` - Error cuadrático medio
- `r2_score()` - Coeficiente de determinación
- `accuracy_score()` - Precisión
- `confusion_matrix()` - Matriz de confusión
- `classification_report()` - Reporte completo

---

## 📝 Notas Importantes

1. **Todos los métodos están claramente marcados** en el código con comentarios `⭐` que indican:
   - La herramienta utilizada
   - El propósito específico de cada uso

2. **El código sigue las mejores prácticas** de scikit-learn:
   - Separación de datos de entrenamiento y prueba
   - Estandarización de características cuando es necesario
   - Uso apropiado de métricas de evaluación

3. **Se demuestra conocimiento avanzado** al usar:
   - Estratificación en la división de datos
   - Escalado de características para regresión logística
   - Múltiples métricas de evaluación

---

**Fecha de creación:** Análisis de Shopping Behavior  
**Versión del código:** analisis_shopping.py
