import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any

app = FastAPI(
    title="API de Predicción de Riesgo Crediticio",
    description="Servicio para disponibilizar el modelo de Machine Learning entrenado.",
    version="1.0.0"
)

# Cargar el modelo entrenado
try:
    model = joblib.load("modelo_entrenado.pkl")
except Exception as e:
    model = None

class InputData(BaseModel):
    features: List[Dict[str, Any]]

@app.get("/")
def home():
    return {"mensaje": "¡El servidor de Machine Learning está activo y funcionando!"}

@app.post("/predict")
def predict(data: InputData):
    if model is None:
        raise HTTPException(status_code=500, detail="El modelo no se encuentra cargado en el servidor.")
    
    try:
        # 1. Convertir los datos recibidos a DataFrame
        df_raw = pd.DataFrame(data.features)
        
        # 2. Reconstruir las columnas con los prefijos que el modelo espera (num__ y cat__)
        df_procesado = pd.DataFrame()
        
        # Mapeo de variables numéricas con el prefijo num__
        cols_numericas = ['capital_prestado', 'plazo_meses', 'edad_cliente', 'salario_cliente', 'cuota_pactada', 'total_otros_prestamos']
        for col in cols_numericas:
            if col in df_raw.columns:
                df_procesado[f"num__{col}"] = df_raw[col]
            else:
                df_procesado[f"num__{col}"] = 0.0 # Valor por defecto si falta
                
        # Mapeo de variables categóricas (One-Hot Encoding manual básico para tipo_laboral)
        if 'tipo_laboral' in df_raw.columns:
            tipo = df_raw['tipo_laboral'].iloc[0]
            df_procesado["cat__tipo_laboral_Empleado"] = 1 if tipo == "Empleado" else 0
            df_procesado["cat__tipo_laboral_Independiente"] = 1 if tipo == "Independiente" else 0
        else:
            df_procesado["cat__tipo_laboral_Empleado"] = 0
            df_procesado["cat__tipo_laboral_Independiente"] = 0

        # Si el modelo espera más columnas específicas del fit, aseguramos que coincidan con las que conoce
        if hasattr(model, "feature_names_in_"):
            for col in model.feature_names_in_:
                if col not in df_procesado.columns:
                    df_procesado[col] = 0.0 # Rellenar con ceros cualquier otra columna faltante
            # Reordenar exactamente en el orden que el modelo espera
            df_procesado = df_procesado[model.feature_names_in_]

        # 3. Realizar la predicción
        predicciones = model.predict(df_procesado)

        return {
            "estado": "exitoso",
            "cantidad_registros": len(df_raw),
            "predicciones": predicciones.tolist()
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error en la inferencia: {str(e)}")