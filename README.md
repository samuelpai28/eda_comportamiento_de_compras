# 📊 Guía de Interpretación de Visualizaciones - Análisis de Shopping Behavior

Este documento explica cómo leer e interpretar cada gráfico generado en el análisis estadístico descriptivo y modelado del dataset de comportamiento de compras.

## 📁 Archivos de Visualizaciones

Las visualizaciones están **separadas por proceso** en 3 archivos:

1. **`01_estadistica_descriptiva.png`** - 6 gráficos de análisis estadístico descriptivo
2. **`02_regresion_lineal.png`** - 6 gráficos de regresión lineal
3. **`03_regresion_logistica.png`** - 6 gráficos de regresión logística

---

## 📈 ARCHIVO 1: Estadística Descriptiva (`01_estadistica_descriptiva.png`)

### **Gráfico 1: Distribución de Montos de Compra (Histograma)**
**Ubicación:** Fila 1, Columna 1

**¿Qué muestra?**
- Distribución completa de frecuencias del monto de compra en dólares
- Muestra cuántas compras se realizaron en cada rango de precio

**Cómo leerlo:**
- **Eje X (horizontal):** Monto de compra en USD (de menor a mayor)
- **Eje Y (vertical):** Frecuencia (número de compras)
- **Línea roja punteada:** Media aritmética (promedio) de todos los montos
- **Línea verde punteada:** Mediana (valor que divide los datos en dos mitades iguales)
- **Líneas naranjas punteadas:** ±1 desviación estándar (rango de variabilidad)
- **Caja de texto (amarilla):** Muestra valores mínimo, máximo y rango

**Interpretación:**
- **Forma de campana (normal):** ✅ Datos bien distribuidos, comportamiento predecible
- **Sesgado a la izquierda:** Más compras de bajo precio, pocos clientes de alto valor
- **Sesgado a la derecha:** Más compras de alto precio, clientes premium
- **Diferencia media-mediana:** Indica presencia de valores extremos (outliers)
- **Desviación estándar grande:** Alta variabilidad en los precios

**Comportamiento de los datos:**
- Si la media ≈ mediana: distribución simétrica
- Si la media > mediana: cola hacia valores altos
- Si la media < mediana: cola hacia valores bajos

---

### **Gráfico 2: Distribución de Edades de Clientes (Análisis Demográfico)**
**Ubicación:** Fila 1, Columna 2

**¿Qué muestra?**
- Distribución de edades de los clientes en el dataset
- Permite identificar qué grupos de edad son más representados

**Cómo leerlo:**
- **Eje X (horizontal):** Edad de los clientes en años
- **Eje Y (vertical):** Frecuencia (número de clientes)
- **Línea roja punteada:** Edad promedio
- **Líneas naranjas punteadas:** ±1 desviación estándar
- **Caja de texto (azul):** Muestra rango de edades y moda

**Interpretación:**
- **Picos altos:** Grupos de edad más representados (público objetivo principal)
- **Distribución uniforme:** Cliente diverso en edades
- **Concentración en un rango:** Estrategia de marketing dirigida a ese segmento
- **Edad promedio:** Indica el perfil demográfico típico del cliente

**Comportamiento de los datos:**
- Identifica si hay una edad dominante o distribución equilibrada
- Útil para segmentación de mercado y estrategias de marketing dirigidas

---

### **Gráfico 3: Comparación de Precios por Categoría (Boxplot)**
**Ubicación:** Fila 1, Columna 3

**¿Qué muestra?**
- Comparación del monto de compra entre diferentes categorías de productos
- Muestra la distribución, mediana, cuartiles y valores atípicos por categoría

**Cómo leerlo:**
- **Eje X (horizontal):** Categorías (Accessories, Clothing, Footwear, Outerwear)
- **Eje Y (vertical):** Monto de compra en USD
- **Caja (box):** Contiene el 50% central de los datos (entre percentil 25 y 75)
- **Línea dentro de la caja:** Mediana (percentil 50)
- **Bigotes (whiskers):** Rango de datos normales (1.5 veces el rango intercuartílico)
- **Puntos fuera de los bigotes:** Valores atípicos (outliers)
- **Diamante rojo:** Media aritmética de cada categoría

**Interpretación:**
- **Cajas más altas:** Mayor variabilidad en precios dentro de esa categoría
- **Mediana más alta:** Categoría generalmente más cara
- **Cajas estrechas:** Precios más consistentes y predecibles
- **Muchos outliers:** Categoría con productos de precio muy variable
- **Posición de la mediana en la caja:** Indica asimetría (si está centrada = simétrica)

**Comportamiento de los datos:**
- Compara qué categorías tienen precios más estables vs. más variables
- Identifica categorías premium vs. económicas
- Útil para estrategias de pricing por categoría

---

### **Gráfico 4: Relación entre Edad y Monto de Compra (Análisis de Correlación)**
**Ubicación:** Fila 2, Columna 1

**¿Qué muestra?**
- Relación entre la edad del cliente y el monto que gasta
- Permite identificar si existe correlación entre estas variables

**Cómo leerlo:**
- **Eje X (horizontal):** Edad del cliente en años
- **Eje Y (vertical):** Monto de compra en USD
- **Cada punto:** Representa una compra individual
- **Línea roja discontinua:** Línea de tendencia (regresión lineal simple)
- **Caja de texto (amarilla):** Muestra el coeficiente de correlación (r)

**Interpretación:**
- **Línea ascendente (pendiente positiva):** Los clientes mayores gastan más (correlación positiva)
- **Línea descendente (pendiente negativa):** Los clientes mayores gastan menos (correlación negativa)
- **Línea horizontal:** No hay relación entre edad y gasto
- **Puntos muy dispersos:** La edad no es un buen predictor del monto de compra
- **Puntos agrupados cerca de la línea:** Relación más fuerte

**Coeficiente de Correlación (r):**
- **r > 0.7:** Correlación fuerte positiva
- **0.3 < r < 0.7:** Correlación moderada positiva
- **0 < r < 0.3:** Correlación débil positiva
- **r ≈ 0:** Sin correlación
- **r < 0:** Correlación negativa

**Comportamiento de los datos:**
- Si hay correlación: la edad puede usarse para predecir gasto
- Si no hay correlación: la edad no es útil para predecir gasto

---

### **Gráfico 5: Distribución de Calificaciones (Satisfacción del Cliente)**
**Ubicación:** Fila 2, Columna 2

**¿Qué muestra?**
- Distribución de las calificaciones que los clientes dan a los productos (1-5 estrellas)
- Indica el nivel de satisfacción general

**Cómo leerlo:**
- **Eje X (horizontal):** Review Rating (1 a 5)
- **Eje Y (vertical):** Frecuencia (número de calificaciones)
- **Línea roja punteada:** Calificación promedio
- **Líneas gris y dorada:** Mínimo (1.0) y máximo (5.0)
- **Caja de texto (verde):** Muestra promedio y desviación estándar

**Interpretación:**
- **Pico cerca de 5:** Clientes muy satisfechos
- **Pico cerca de 1:** Clientes insatisfechos
- **Distribución centrada en 3-4:** Satisfacción moderada
- **Promedio alto (>4):** Buena calidad percibida
- **Promedio bajo (<3):** Necesita mejora en calidad

**Comportamiento de los datos:**
- Indica la calidad percibida de los productos
- Útil para identificar áreas de mejora
- Relacionado con retención de clientes

---

### **Gráfico 6: Comparación de Gastos por Género (Análisis de Segmentación)**
**Ubicación:** Fila 2, Columna 3

**¿Qué muestra?**
- Comparación del monto de compra entre géneros
- Permite identificar diferencias en comportamiento de compra

**Cómo leerlo:**
- **Eje X (horizontal):** Género (Female, Male)
- **Eje Y (vertical):** Monto de compra en USD
- **Cajas de colores:** Distribución de gastos por género
- **Diamante rojo:** Media aritmética de cada género
- **Caja de texto (amarilla):** Muestra la media de gasto por género

**Interpretación:**
- **Cajas en posiciones similares:** Gasto similar entre géneros
- **Caja más alta:** Mayor variabilidad en gastos
- **Mediana más alta:** Ese género gasta más en promedio
- **Diferencias significativas:** Necesitan estrategias de marketing diferenciadas

**Comportamiento de los datos:**
- Identifica si hay diferencias significativas en comportamiento de compra
- Útil para segmentación y personalización de ofertas
- Puede indicar preferencias de productos por género

---

## 📈 ARCHIVO 2: Regresión Lineal (`02_regresion_lineal.png`)

### **Gráfico 1: Evaluación del Modelo - Predicciones vs Reales**
**Ubicación:** Fila 1, Columna 1

**¿Qué muestra?**
- Evalúa qué tan bien el modelo de regresión lineal predice los montos de compra
- Compara los valores reales con las predicciones del modelo

**Cómo leerlo:**
- **Eje X (horizontal):** Valores reales de Purchase Amount
- **Eje Y (vertical):** Valores predichos por el modelo
- **Línea roja diagonal:** Línea de predicción perfecta (donde deberían estar todos los puntos)
- **R²:** Coeficiente de determinación (mostrado en el título y caja de texto)
- **Caja de texto (amarilla):** Muestra MSE, RMSE, R² y evaluación del modelo

**Interpretación:**
- **Puntos cerca de la línea roja:** ✅ El modelo predice bien
- **Puntos dispersos lejos de la línea:** ❌ El modelo tiene errores grandes
- **R² cercano a 1:** ✅ Modelo muy bueno (explica mucha varianza)
- **R² cercano a 0 o negativo:** ❌ Modelo no es útil para predecir
- **MSE/RMSE bajo:** ✅ Errores pequeños en las predicciones

**Comportamiento de los datos:**
- Si todos los puntos forman una línea diagonal = predicción perfecta
- Dispersión indica incertidumbre en las predicciones
- R² negativo indica que el modelo es peor que simplemente predecir la media

---

### **Gráfico 2: Análisis de Residuos ⭐ (VALIDACIÓN DE SUPUESTOS)**
**Ubicación:** Fila 1, Columna 2

**¿Qué muestra?**
- Los residuos son la diferencia entre el valor real y la predicción del modelo
- **Este gráfico es FUNDAMENTAL** para validar la calidad del modelo de regresión lineal

**Cómo leerlo:**
- **Eje X (horizontal):** Valores predichos por el modelo
- **Eje Y (vertical):** Residuos (Real - Predicción)
- **Línea roja horizontal en y=0:** Línea de referencia (residuo cero = predicción perfecta)
- **Líneas naranjas punteadas:** ±2 desviaciones estándar (límites de valores normales)
- **Caja de texto (coral):** Muestra estadísticas de residuos y diagnóstico

**Interpretación:**
- **Patrón aleatorio alrededor de y=0:** ✅ Modelo correcto, no hay problemas
- **Patrón en forma de embudo:** ❌ Varianza no constante (heterocedasticidad)
- **Patrón curvo:** ❌ Relación no lineal (el modelo lineal no es apropiado)
- **Puntos concentrados cerca de y=0:** ✅ Buen modelo
- **Muchos puntos lejos de y=0:** ❌ Modelo con errores grandes
- **Media de residuos ≈ 0:** ✅ No hay sesgo sistemático

**Comportamiento de los datos:**
- Valida los supuestos de la regresión lineal:
  - **Linealidad:** Los residuos deben ser aleatorios
  - **Homocedasticidad:** Varianza constante
  - **Normalidad:** Residuos normalmente distribuidos
- Identifica problemas en el modelo que no se ven en otras métricas

---

### **Gráfico 3: Distribución de Errores del Modelo (Análisis de Normalidad)**
**Ubicación:** Fila 1, Columna 3

**¿Qué muestra?**
- Distribución de los errores del modelo (residuos)
- Complementa el gráfico de residuos anterior

**Cómo leerlo:**
- **Eje X (horizontal):** Valor del residuo (error)
- **Eje Y (vertical):** Frecuencia (cuántas veces ocurre ese error)
- **Línea roja punteada:** Media de los residuos
- **Línea verde:** Residuo = 0 (referencia)
- **Caja de texto (lavanda):** Muestra diagnóstico de normalidad

**Interpretación:**
- **Forma de campana (normal) centrada en 0:** ✅ Distribución ideal de errores
- **Media cercana a 0:** ✅ El modelo no tiene sesgo sistemático
- **Sesgado a la izquierda o derecha:** ❌ El modelo subestima o sobreestima consistentemente
- **Muy disperso:** ❌ Errores grandes y variables
- **Concentrado cerca de 0:** ✅ Errores pequeños y consistentes

**Comportamiento de los datos:**
- Valida el supuesto de normalidad de los residuos
- Errores normalmente distribuidos = modelo estadísticamente válido
- Sesgos indican que el modelo necesita ajustes

---

### **Gráfico 4: Variable Independiente - Edad (Relación con Variable Dependiente)**
**Ubicación:** Fila 2, Columna 1

**¿Qué muestra?**
- Relación individual entre la edad y el monto de compra
- Muestra cómo esta variable contribuye al modelo

**Cómo leerlo:**
- **Eje X (horizontal):** Edad en años
- **Eje Y (vertical):** Monto de compra en USD
- **Cada punto:** Una compra individual
- **Línea roja discontinua:** Línea de regresión simple (y = mx + b)
- **Ecuación mostrada:** Pendiente e intercepto

**Interpretación:**
- **Pendiente positiva:** A mayor edad, mayor gasto
- **Pendiente negativa:** A mayor edad, menor gasto
- **Pendiente cercana a 0:** La edad no influye mucho
- **Dispersión:** Indica qué tan fuerte es la relación

**Comportamiento de los datos:**
- Muestra la contribución individual de la edad al modelo
- Útil para entender qué variables son más importantes

---

### **Gráfico 5: Variable Independiente - Review Rating**
**Ubicación:** Fila 2, Columna 2

**¿Qué muestra?**
- Relación entre la calificación del producto y el monto de compra
- Indica si productos mejor calificados son más caros

**Cómo leerlo:**
- **Eje X (horizontal):** Review Rating (1-5)
- **Eje Y (vertical):** Monto de compra en USD
- **Línea roja discontinua:** Línea de regresión simple
- **Ecuación mostrada:** Pendiente e intercepto

**Interpretación:**
- **Pendiente positiva:** Productos mejor calificados son más caros
- **Pendiente negativa:** Productos mejor calificados son más baratos (poco común)
- **Relación fuerte:** La calidad percibida influye en el precio

**Comportamiento de los datos:**
- Muestra si hay relación precio-calidad
- Útil para estrategias de pricing basadas en calidad

---

### **Gráfico 6: Coeficientes del Modelo de Regresión**
**Ubicación:** Fila 2, Columna 3

**¿Qué muestra?**
- Valores de los coeficientes del modelo (intercepto y pendientes)
- Muestra la ecuación completa del modelo

**Cómo leerlo:**
- **Eje Y (vertical):** Nombre de cada coeficiente
- **Eje X (horizontal):** Valor del coeficiente
- **Barras rojas:** Intercepto
- **Barras azules:** Coeficientes de variables independientes
- **Valores en las barras:** Valor numérico de cada coeficiente
- **Ecuación en la parte inferior:** Ecuación completa del modelo

**Interpretación:**
- **Coeficiente positivo:** Aumenta el valor predicho
- **Coeficiente negativo:** Disminuye el valor predicho
- **Valor absoluto grande:** Variable más influyente
- **Valor absoluto pequeño:** Variable menos influyente
- **Intercepto:** Valor base cuando todas las variables son 0

**Ecuación del modelo:**
```
y = Intercepto + Coef_Age × Age + Coef_Rating × Rating + Coef_PrevPurch × Previous_Purchases
```

**Comportamiento de los datos:**
- Muestra la importancia relativa de cada variable
- Permite interpretar cómo cada factor afecta el monto de compra

---

## 📈 ARCHIVO 3: Regresión Logística (`03_regresion_logistica.png`)

### **Gráfico 1: Distribución de Probabilidades (Función Sigmoide)**
**Ubicación:** Fila 1, Columna 1

**¿Qué muestra?**
- Distribución de las probabilidades generadas por la función sigmoide
- Muestra qué tan "seguro" está el modelo en sus predicciones de clasificación

**Cómo leerlo:**
- **Eje X (horizontal):** Probabilidad de que el cliente tenga suscripción activa (0 a 1)
- **Eje Y (vertical):** Frecuencia (número de predicciones)
- **Histograma verde:** Probabilidad de "Yes" (con suscripción)
- **Histograma rojo:** Probabilidad de "No" (sin suscripción)
- **Línea azul vertical en x=0.5:** Umbral de decisión (threshold)
- **Caja de texto (verde):** Muestra estadísticas de predicciones

**Interpretación:**
- **Picos cerca de 0 y 1:** ✅ Modelo muy seguro en sus predicciones
- **Distribución centrada en 0.5:** ❌ Modelo incierto, probabilidades ambiguas
- **Más puntos a la izquierda de 0.5:** Predice más "No" (sin suscripción)
- **Más puntos a la derecha de 0.5:** Predice más "Yes" (con suscripción)
- **Función Sigmoide:** Transforma valores reales a probabilidades entre 0 y 1

**Comportamiento de los datos:**
- **Umbral de decisión:** Si probabilidad > 0.5 → Clase 1 (Yes), sino → Clase 0 (No)
- Modelos seguros tienen probabilidades extremas (cerca de 0 o 1)
- Modelos inciertos tienen probabilidades intermedias

---

### **Gráfico 2: Matriz de Confusión (TP, TN, FP, FN)**
**Ubicación:** Fila 1, Columna 2

**¿Qué muestra?**
- Evalúa el rendimiento del modelo de clasificación
- Muestra cuántas predicciones fueron correctas e incorrectas

**Cómo leerlo:**
- **Eje Y (vertical):** Valores reales (No, Yes)
- **Eje X (horizontal):** Valores predichos (No, Yes)
- **Números en cada celda:** Cantidad de casos
- **Colores:** Intensidad indica cantidad (más oscuro = más casos)
- **Caja de texto (amarilla):** Explica TP, TN, FP, FN

**Estructura de la matriz:**
```
                Predicción
              No (0)    Yes (1)
Real  No (0)  [TN]      [FP]
      Yes (1) [FN]      [TP]
```

**Interpretación:**
- **Diagonal principal (arriba-izquierda a abajo-derecha):** ✅ Predicciones correctas
  - **TP (True Positives):** Predijo "Yes" y era "Yes" ✅
  - **TN (True Negatives):** Predijo "No" y era "No" ✅
- **Fuera de la diagonal:** ❌ Predicciones incorrectas
  - **FP (False Positives):** Predijo "Yes" pero era "No" ❌ (Falso positivo)
  - **FN (False Negatives):** Predijo "No" pero era "Yes" ❌ (Falso negativo)

**Comportamiento de los datos:**
- **Muchos valores en la diagonal:** ✅ Modelo bueno
- **Muchos valores fuera de la diagonal:** ❌ Modelo con muchos errores
- **TP alto:** Modelo detecta bien los casos positivos
- **TN alto:** Modelo detecta bien los casos negativos

---

### **Gráfico 3: Distribución de Probabilidades Ordenadas (Análisis de Clasificación)**
**Ubicación:** Fila 1, Columna 3

**¿Qué muestra?**
- Curva que muestra cómo se distribuyen las probabilidades ordenadas
- Similar a una curva ROC simplificada

**Cómo leerlo:**
- **Eje X (horizontal):** Tasa de Falsos Positivos
- **Eje Y (vertical):** Tasa de Verdaderos Positivos
- **Línea azul:** Curva de probabilidades del modelo
- **Línea roja discontinua:** Línea base (clasificador aleatorio)

**Interpretación:**
- **Curva por encima de la línea roja:** ✅ Modelo mejor que aleatorio
- **Curva cerca de la línea roja:** ❌ Modelo similar a aleatorio
- **Curva cerca de la esquina superior izquierda:** ✅ Modelo excelente
- **Área bajo la curva grande:** ✅ Buen modelo de clasificación

**Comportamiento de los datos:**
- Muestra la capacidad del modelo para distinguir entre clases
- Útil para comparar diferentes modelos

---

### **Gráfico 4: Comparación de Edades (Suscripción vs No Suscripción)**
**Ubicación:** Fila 2, Columna 1

**¿Qué muestra?**
- Comparación de la distribución de edades entre clientes con y sin suscripción
- Identifica si la edad influye en tener suscripción

**Cómo leerlo:**
- **Eje X (horizontal):** Grupos (Sin Suscripción, Con Suscripción)
- **Eje Y (vertical):** Edad en años
- **Cajas de colores:** Distribución de edades en cada grupo
- **Caja roja:** Clientes sin suscripción
- **Caja verde:** Clientes con suscripción

**Interpretación:**
- **Cajas en posiciones similares:** La edad no influye mucho
- **Caja verde más alta/centrada diferente:** Edad influye en tener suscripción
- **Mediana más alta en un grupo:** Ese grupo tiene clientes más mayores
- **Cajas estrechas:** Edades más homogéneas en ese grupo

**Comportamiento de los datos:**
- Identifica si hay diferencias demográficas entre grupos
- Útil para estrategias de marketing dirigidas

---

### **Gráfico 5: Comparación de Gastos (Suscripción vs No Suscripción)**
**Ubicación:** Fila 2, Columna 2

**¿Qué muestra?**
- Comparación del monto de compra entre clientes con y sin suscripción
- Identifica si el gasto influye en tener suscripción

**Cómo leerlo:**
- **Eje X (horizontal):** Grupos (Sin Suscripción, Con Suscripción)
- **Eje Y (vertical):** Purchase Amount en USD
- **Cajas de colores:** Distribución de gastos en cada grupo
- **Caja de texto (amarilla):** Muestra la media de gasto por grupo

**Interpretación:**
- **Caja verde más alta:** Clientes con suscripción gastan más
- **Mediana más alta:** Ese grupo tiene mayor gasto promedio
- **Diferencias significativas:** El gasto está relacionado con tener suscripción
- **Cajas similares:** El gasto no influye en tener suscripción

**Comportamiento de los datos:**
- Muestra si hay relación entre gasto y suscripción
- Clientes que gastan más pueden ser más propensos a suscribirse
- Útil para identificar clientes potenciales para suscripción

---

### **Gráfico 6: Métricas del Modelo**
**Ubicación:** Fila 2, Columna 3

**¿Qué muestra?**
- Resumen completo de todas las métricas de evaluación del modelo
- Proporciona una evaluación integral del rendimiento

**Cómo leerlo:**
- **Texto en caja azul:** Muestra todas las métricas calculadas
- **Cada métrica incluye:** Valor numérico y explicación

**Métricas mostradas:**

1. **Accuracy (Precisión Global):**
   - Porcentaje de predicciones correctas
   - Rango: 0 a 1 (o 0% a 100%)
   - **Alto (>0.8):** ✅ Modelo bueno
   - **Bajo (<0.6):** ❌ Modelo necesita mejora

2. **Precision (Precisión):**
   - De los predichos como "Yes", cuántos realmente eran "Yes"
   - **Alto:** Pocos falsos positivos
   - **Bajo:** Muchos falsos positivos

3. **Recall (Sensibilidad):**
   - De los realmente "Yes", cuántos predijo correctamente
   - **Alto:** Detecta bien los casos positivos
   - **Bajo:** Se pierde muchos casos positivos

4. **Specificity (Especificidad):**
   - De los realmente "No", cuántos predijo correctamente
   - **Alto:** Detecta bien los casos negativos
   - **Bajo:** Muchos falsos negativos

5. **F1-Score:**
   - Media armónica de Precision y Recall
   - Balance entre ambas métricas
   - **Alto:** Buen balance entre precisión y recall

6. **Umbral de Decisión:**
   - Valor de probabilidad que separa las clases (0.5 por defecto)
   - Probabilidad > 0.5 → Yes
   - Probabilidad ≤ 0.5 → No

**Comportamiento de los datos:**
- **Accuracy alto:** Modelo general bueno
- **Precision alto:** Pocos falsos positivos (importante si el costo de FP es alto)
- **Recall alto:** Pocos falsos negativos (importante si no queremos perder casos positivos)
- **F1-Score alto:** Balance óptimo entre precisión y recall

---

## 🎯 Resumen de Conceptos Clave

### **Estadística Descriptiva:**
- **Media (np.mean):** Centro de los datos, promedio aritmético
- **Mediana:** Valor que divide los datos en dos mitades iguales
- **Desviación estándar (np.std):** Medida de dispersión o variabilidad
- **Histogramas:** Distribución de frecuencias, muestra cómo se distribuyen los datos
- **Boxplots:** Muestran cuartiles, mediana y valores atípicos

### **Regresión Lineal:**
- **Variable Independiente (X):** Causa (Age, Review Rating, Previous Purchases)
- **Variable Dependiente (y):** Efecto (Purchase Amount)
- **Residuos:** Diferencia entre valor real y predicción (deben ser aleatorios)
- **R²:** Coeficiente de determinación, mide qué tan bien el modelo explica la varianza
- **MSE/RMSE:** Medidas de error del modelo

### **Regresión Logística:**
- **Función Sigmoide:** Transforma valores reales a probabilidades (0-1)
- **Umbral (Threshold):** 0.5 por defecto, separa las clases
- **Clasificación Binaria:** Sí/No, 1/0, Yes/No
- **TP, TN, FP, FN:** Verdaderos positivos/negativos, Falsos positivos/negativos
- **Accuracy, Precision, Recall, F1-Score:** Métricas de evaluación

### **Scikit-Learn:**
- **fit():** Entrena el modelo con los datos
- **predict():** Hace predicciones
- **predict_proba():** Obtiene probabilidades (solo regresión logística)
- **train_test_split():** Divide datos en entrenamiento y prueba

---

## 📚 Referencias de la Guía

Este análisis implementa los conceptos de la guía de **Estadística Descriptiva y Modelado con Python**:

1. ✅ Manipulación de datos con Pandas (`df.describe()`, `df.fillna()`, `df.groupby()`)
2. ✅ Estadística descriptiva con NumPy (`np.mean()`, `np.std()`)
3. ✅ Regresión Lineal con visualización de residuos (⭐ importante)
4. ✅ Regresión Logística con función sigmoide y umbral
5. ✅ Uso completo de scikit-learn

---

## 🚀 Ejecución del Análisis

Para generar las visualizaciones, ejecuta:

```bash
python analisis_shopping.py
```

Se generarán automáticamente 3 archivos:
- `01_estadistica_descriptiva.png` (6 gráficos)
- `02_regresion_lineal.png` (6 gráficos)
- `03_regresion_logistica.png` (6 gráficos)

---

## 💡 Mejoras en las Visualizaciones

Las nuevas visualizaciones incluyen:

1. **Más información estadística:** Cada gráfico muestra estadísticas relevantes
2. **Anotaciones y textos explicativos:** Cajas de texto con información clave
3. **Mejor uso de colores:** Colores diferenciados para mejor comprensión
4. **Líneas de tendencia:** Regresiones simples en scatter plots
5. **Diagnósticos automáticos:** Evaluaciones del comportamiento de los datos
6. **Separación por proceso:** Cada archivo se enfoca en un aspecto específico
7. **Títulos descriptivos:** Explican qué muestra cada gráfico
8. **Métricas integradas:** Valores importantes mostrados directamente en los gráficos

---

**Nota:** Este README está diseñado para ayudarte a interpretar cada gráfico de manera educativa, siguiendo los principios de estadística descriptiva y modelado con Python. Las visualizaciones mejoradas proporcionan más contexto y facilitan la comprensión del comportamiento de los datos.
