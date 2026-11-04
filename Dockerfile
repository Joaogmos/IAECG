# Imagem da API + interface (CPU). O modelo não vai na imagem: monte a pasta dele.
#
#   docker build -t iaecg .
#   docker run -p 8000:8000 -v "%cd%/resultados/v3/cnn1d_s42:/modelo" iaecg
#   (opcional, exemplos do CODE-test)  -v "%cd%/dados/code_test:/app/dados/code_test"
FROM python:3.13-slim

WORKDIR /app
RUN pip install --no-cache-dir torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY comum_ecg.py cnn1d_ecg.py transformer_ecg.py inferencia_ecg.py api_ecg.py triagem_lote.py ./
COPY frontend/ frontend/

ENV IAECG_MODELO=/modelo/modelo_final.pt \
    IAECG_DISPOSITIVO=cpu \
    PYTHONUNBUFFERED=1
EXPOSE 8000
CMD ["python", "api_ecg.py", "--host", "0.0.0.0", "--porta", "8000"]
