# **Contextualización**

## Clasificación de default en préstamos de Lending Club con Random Forest (scikit-learn y PySpark)

Este proyecto corresponde al curso de **Machine Learning** de la Universidad del Norte (Proyecto Integrador) y se centra en la construcción de un clasificador **supervisado** para predecir el **riesgo de incumplimiento (default)** en préstamos personales emitidos por la plataforma **Lending Club**. Se busca comparar la implementación en **scikit-learn** y en **PySpark**, además de aplicar técnicas de **interpretabilidad con LIME**.

### Objetivo general

1. **Entrenar un modelo de clasificación supervisada (Random Forest)** para predecir si un préstamo será **pagado completamente (0)** o terminará en **default (1)**.
2. **Comparar el desempeño** de los modelos en dos entornos: *scikit-learn* (local, CPU) y *PySpark* (paralelización y escalabilidad).
3. **Aplicar LIME** para explicar predicciones individuales y comprender el aporte de las variables al modelo.

### Contexto aplicado

En el ámbito de los **servicios financieros digitales**, los modelos de clasificación permiten estimar el riesgo de incumplimiento de crédito, lo cual es crucial para:

* **Mitigar pérdidas** por préstamos en default.
* **Ajustar políticas de otorgamiento de crédito** basadas en riesgo.
* **Generar confianza** en plataformas de financiamiento colectivo (fintech).

El dataset de **Lending Club** (2007–2020) contiene más de **1,3 millones de registros** con información socioeconómica, laboral, crediticia y de propósito del préstamo, lo que lo convierte en un banco de pruebas ideal para comparar enfoques de **aprendizaje automático en gran escala**.

### Enfoque metodológico

* **Datos y variables**

  * Dataset: *Lending Club Loan Data (2007–2020)*.
  * Variables clave: `loan_amnt`, `int_rate`, `fico_range_high`, `emp_length`, `annual_inc`, `purpose`, `home_ownership`, `dti`, `addr_state`.
  * Variable objetivo: **default** (1 = *Charged Off*, 0 = *Fully Paid*).

* **Preprocesamiento**

  * *scikit-learn*: OneHotEncoder / LabelEncoder, StandardScaler, `train_test_split` (estratificado).
  * *PySpark*: StringIndexer + OneHotEncoder, VectorAssembler, StandardScaler opcional, `randomSplit`.

* **Modelado**

  * `RandomForestClassifier` en *scikit-learn* y *PySpark*.
  * Comparación de hiperparámetros: `n_estimators = [10, 50, 100]`, `max_depth = [5, 10, 15]`.
  * Evaluación: **Accuracy, Precision, Recall, F1-score, ROC AUC, matriz de confusión**.
  * Medición de **tiempos de entrenamiento y predicción** en ambos entornos.

* **Interpretabilidad con LIME**

  * Selección de instancias mal clasificadas.
  * Análisis de variables más influyentes en la predicción.
  * Gráficas locales que muestren el peso de cada atributo en la decisión.

### Resultado esperado

Un **Jupyter Book bien documentado** que incluya:

* Exploración y visualización de los datos (EDA).
* Preprocesamiento en *scikit-learn* y *PySpark*.
* Modelado con Random Forest y ajuste de hiperparámetros.
* Evaluación comparativa de métricas y tiempos.
* Análisis interpretativo con LIME.
* **Reflexión crítica** sobre ventajas y limitaciones de ambos entornos (rapidez, escalabilidad, precisión, interpretabilidad).

### Restricciones didácticas

* Usar **Random Forest** en ambas implementaciones.
* No utilizar `Pipeline` en GridSearch de *scikit-learn*.
* Comparar métricas de forma estandarizada.
* Incluir **LIME obligatoriamente** para explicar predicciones.

### Herramientas sugeridas

`pandas`, `numpy`, `matplotlib`, `seaborn`; `scikit-learn`; `pyspark.ml`; `lime`; **Jupyter Notebook** / **Jupyter Book**.

### Diseño sugerido del notebook

1. **Carga y EDA** del dataset de Lending Club.
2. **Preprocesamiento** en *scikit-learn* y *PySpark*.
3. **Entrenamiento y evaluación** con Random Forest.
4. **Comparación de métricas y tiempos** entre entornos.
5. **Interpretabilidad con LIME** sobre predicciones seleccionadas.
6. **Reflexión crítica y conclusiones** sobre desempeño, escalabilidad y utilidad práctica.

### Resumen

* **Dataset:** Lending Club (2007–2020, >1,3M registros).
* **Tarea:** Clasificación binaria (default vs fully paid).
* **Modelo núcleo:** Random Forest (scikit-learn y PySpark).
* **Ventajas clave:** Comparación de entornos (local vs distribuido), análisis de desempeño y tiempo, explicabilidad con LIME en contexto financiero.