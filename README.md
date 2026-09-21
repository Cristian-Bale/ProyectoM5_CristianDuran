# 📊 Proyecto de Scoring Crediticio y MLOps: Predicción y Monitoreo de Riesgo

## 🚀 1. Caso de Negocio
Este proyecto se enfoca en el sector financiero para la **predicción de riesgo crediticio** (identificación de clientes con probabilidad de caer en mora o pago a tiempo). El objetivo principal es optimizar el otorgamiento de créditos mediante un modelo de Machine Learning robusto, capaz de anticiparse a los incumplimientos y, al mismo tiempo, garantizar la estabilidad del modelo a lo largo del tiempo a través de prácticas de **MLOps** (monitoreo continuo de *Data Drift*).

---

## 📂 2. Estructura del Repositorio y Versionamiento
El proyecto sigue un flujo de trabajo estructurado en ramas (`main`, `certification`, `developer`) para asegurar un control estricto de versiones y calidad de código:

```text
├── data/                      # Datasets limpios y de entrenamiento
├── src/                       # Scripts principales (monitoreo, utilidades)
├── notebooks/                 # Cuadernos Jupyter (EDA y Modelamiento)
├── app.py                     # Aplicación interactiva en Streamlit
├── model_monitoring.py        # Motor estadístico de detección de Data Drift
├── requirements.txt           # Dependencias del proyecto
└── README.md                  # Documentación oficial


3. Hallazgos Clave del EDA y Limpieza de Datos
Durante la fase de exploración (Comprensión_eda.ipynb):

Se unificaron criterios de valores nulos y se eliminaron variables irrelevantes o con alta redundancia.

Se identificó y corrigió un grave problema de Data Leakage (Fuga de datos) provocado por variables predictoras directas del target (como el puntaje interno), lo que inflaba artificialmente las métricas iniciales. Tras su eliminación, se logró un entorno de aprendizaje legítimo y realista.


 4. Modelamiento y Evaluación
Se evaluaron tres algoritmos diferentes aplicando técnicas de balanceo de clases (scale_pos_weight y class_weight='balanced') para atender la clase minoritaria de deudores:

Random Forest: Modelo con mejor desempeño general, alcanzando un F1-Score de 0.97 y un Accuracy de 0.94.

XGBoost: Excelente capacidad predictiva, situándose con un F1-Score de 0.94 tras resolver el desbalance.

Regresión Logística: Utilizada como modelo base de comparación.


5. Monitoreo de MLOps y Data Drift (model_monitoring.py)
Para evitar la degradación del modelo en producción, se implementó un sistema automatizado de detección de desviaciones poblacionales (Data Drift) utilizando métricas estadísticas avanzadas:

Test de Kolmogorov-Smirnov (KS): Evalúa cambios en distribuciones numéricas continuas.

Population Stability Index (PSI): Mide la estabilidad general de la población mediante binned scoring.

Divergencia de Jensen-Shannon: Cuantifica la distancia entre las probabilidades históricas y actuales.

Chi-cuadrado: Detecta cambios en las frecuencias de variables categóricas.


6. Dashboard Interactivo (Streamlit)
El proyecto cuenta con una interfaz visual desarrollada en Streamlit que permite:

Visualizar un panel de alertas en tiempo real (semáforos de riesgo).

Consultar tablas detalladas con métricas de drift por variable.

Comparar gráficamente las curvas de distribución histórica frente a la población actual de producción.




7. Despliegue del Modelo y Contenedorización (API con FastAPI y Docker)

En esta etapa se ha completado la puesta en producción del modelo de Machine Learning para la predicción de riesgo crediticio, garantizando su portabilidad y facilidad de consumo mediante una API robusta.

### Características Principales
* **API de Predicción (`FastAPI`)**: Se desarrolló el script `model_deploy.py` que expone una interfaz web interactiva y un endpoint POST `/predict` optimizado para recibir datos en formato JSON.
* **Soporte de Inferencia por Lotes**: La API permite procesar múltiples registros de clientes de forma simultánea en una sola solicitud.
* **Alineación de Esquema**: Se integró un adaptador lógico en el servicio para asegurar que los datos crudos de entrada coincidan de forma transparente con los requerimientos y transformaciones internas del pipeline entrenado (`modelo_entrenado.pkl`).
* **Contenedorización (`Docker`)**: Se empaquetó toda la aplicación utilizando un entorno aislado con Python 3.10-slim, asegurando que las dependencias (`requirements.txt`) y el modelo se ejecuten de manera idéntica en cualquier sistema operativo.

---

### Instrucciones para Ejecutar localmente con Docker

1. **Construir la imagen de Docker**:
   ```bash
   docker build -t api-riesgo-credito:v1.0 .
   docker build -t api-riesgo-credito:v1.0 .
