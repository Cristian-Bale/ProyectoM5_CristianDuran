import streamlit as st
import pandas as pd
import numpy as np
from model_monitoring import DriftDetector 

st.set_page_config(page_title="Monitor de Data Drift", layout="wide")
st.title("🚦 Panel de Monitoreo de Riesgo Crediticio")
st.markdown("Sistema de alerta temprana para desviación poblacional (Data Drift).")

try:
    # Carga de tu base de datos real
    df_completo = pd.read_excel("data/Base_de_datos.xlsx")
    
    # Aplicar la misma limpieza de columnas que en el EDA para que coincidan
    columnas_a_borrar = [
        'tipo_credito', 'huella_consulta', 'saldo_mora_codeudor', 
        'puntaje_datacredito', 'tendencia_ingresos', 'saldo_mora', 
        'saldo_total', 'saldo_principal', 'puntaje'
    ]
    df_completo = df_completo.drop(columns=columnas_a_borrar, errors='ignore')

    mitad = len(df_completo) // 2
    df_historico = df_completo.iloc[:mitad]
    df_actual = df_completo.iloc[mitad:]      

    
    vars_numericas = ["capital_prestado", "plazo_meses", "edad_cliente", "salario_cliente", "cuota_pactada"] 
    vars_categoricas = ["tipo_laboral"]          


    monitor = DriftDetector(df_historico, df_actual)
    df_resultados = monitor.evaluar_dataset(vars_numericas, vars_categoricas)
    datos_cargados = True

except FileNotFoundError:
    st.warning("⚠️ No se encontró el archivo de base de datos en 'data/Base_de_datos.xlsx'. Mostrando datos de simulación...")
    # Datos de respaldo para que la app no colapse si la ruta falla
    df_resultados = pd.DataFrame({
        "Variable": ["capital_prestado", "edad_cliente", "salario_cliente", "tipo_laboral"],
        "Metrica": ["PSI", "KS Test", "Jensen-Shannon", "Chi-Square"],
        "Valor": [0.25, 0.02, 0.08, 0.45],
        "Drift": [True, True, False, False]
    })
    datos_cargados = False

# --- 2. ALERTAS Y RECOMENDACIONES ---
st.header("1. Estado de Salud y Recomendaciones")
variables_drift = df_resultados[df_resultados['Drift'] == True]

if not variables_drift.empty:
    st.error(f"⚠️ ALERTA CRÍTICA: Se detectó desviación poblacional en {len(variables_drift)} prueba(s).")
    st.warning("""
    **💡 Acciones Sugeridas:**
    1. **Retraining:** Programar re-entrenamiento del modelo con la nueva distribución de datos.
    2. **Revisión de Ingesta:** Verificar el pipeline de captura para las variables afectadas buscando posibles errores de formato o valores atípicos recientes.
    """)
else:
    st.success("✅ Sistema estable. No se requiere intervención. La población de datos es consistente con el histórico.")

# --- 3. MÉTRICAS VISUALES ---
st.header("2. Visualización de Métricas de Drift")
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Tabla de Detección por Variable")
    # Aplicamos el semáforo usando map()
    st.dataframe(df_resultados.style.map(
        lambda x: 'background-color: #ffcccc' if x is True else ('background-color: #ccffcc' if x is False else ''),
        subset=['Drift']
    ), use_container_width=True)

with col2:
    st.subheader("Distribución General: Histórico vs Actual")
    # Usamos una variable numérica real de créditos, por ejemplo, "capital_prestado"
    columna_grafico = "capital_prestado"
    
    if datos_cargados and columna_grafico in df_historico.columns:
        chart_data = pd.DataFrame({
            'Histórico': df_historico[columna_grafico].dropna().sample(min(500, len(df_historico)), replace=True).values,
            'Actual': df_actual[columna_grafico].dropna().sample(min(500, len(df_actual)), replace=True).values
        })
    else:
        # Gráfico simulado si no cargan los datos
        chart_data = pd.DataFrame({
            'Histórico': np.random.normal(10000, 1500, 500),
            'Actual': np.random.normal(11000, 2000, 500)
        })
    st.area_chart(chart_data, color=["#1f77b4", "#ff7f0e"])

# --- 4. ANÁLISIS TEMPORAL ---
st.header("3. Evolución Temporal del Drift")
st.markdown("Monitor de tendencias a lo largo de los últimos 30 días (Simulación de métrica continua).")

fechas = pd.date_range(end=pd.Timestamp.today(), periods=30)
psi_trend = np.linspace(0.08, 0.23, 30) + np.random.normal(0, 0.02, 30)
df_trend = pd.DataFrame({'Fecha': fechas, 'Valor_PSI': psi_trend}).set_index('Fecha')

st.line_chart(df_trend)
st.caption("🚨 Nota: La línea ascendente muestra una degradación progresiva. Valores de PSI > 0.20 indican una desviación estructural fuerte.")