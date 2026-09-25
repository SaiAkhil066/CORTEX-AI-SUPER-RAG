FROM python:3.11-slim

WORKDIR /usr/src/app

# CPU-only torch keeps the image ~2 GB smaller; drop the index-url line for CUDA.
COPY requirements.txt ./
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt

COPY . .

ENV INDEX_DIR=/data/indexes
VOLUME ["/data"]
EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
