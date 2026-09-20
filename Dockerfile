# Usar una imagen oficial de Python ligera
FROM python:3.10-slim

# Establecer el directorio de trabajo dentro del contenedor
WORKDIR /app

# Copiar e instalar las dependencias
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar todo el código fuente y el modelo entrenado al contenedor
COPY . .

# Exponer el puerto por el que correrá FastAPI
EXPOSE 8000

# Comando para iniciar el servidor Uvicorn
CMD ["uvicorn", "model_deploy:app", "--host", "0.0.0.0", "--port", "8000"]