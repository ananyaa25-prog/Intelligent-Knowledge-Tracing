# GatedKT web app in one container: FastAPI model service (internal, port 8000) + Node gateway (port 7860).
# Build: docker build -t gatedkt .    Run: docker run -p 7860:7860 gatedkt    Open: http://localhost:7860/
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends nodejs npm && rm -rf /var/lib/apt/lists/*
RUN useradd -m -u 1000 user
WORKDIR /app

COPY requirements-deploy.txt .
RUN pip install --no-cache-dir torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements-deploy.txt

COPY server/package.json server/package-lock.json server/
RUN cd server && npm ci --omit=dev

COPY --chown=user . .
RUN chown -R user /app/data /app/reports /app/server
USER user
ENV MPLCONFIGDIR=/tmp/mpl PYTHONUNBUFFERED=1
EXPOSE 7860
CMD ["python", "scripts/start_demo.py", "--py-port", "8000", "--node-port", "7860"]
