# Hugging Face Space: build the app straight from the GitHub repository.
FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends git nodejs npm && rm -rf /var/lib/apt/lists/*
RUN useradd -m -u 1000 user
ARG REPO=https://github.com/ananyaa25-prog/Intelligent-Knowledge-Tracing.git
RUN git clone --depth 1 "$REPO" /app && chown -R user /app
WORKDIR /app
RUN pip install --no-cache-dir torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements-deploy.txt
RUN cd server && npm ci --omit=dev && chown -R user /app/server
USER user
ENV MPLCONFIGDIR=/tmp/mpl PYTHONUNBUFFERED=1
EXPOSE 7860
CMD ["python", "scripts/start_demo.py", "--py-port", "8000", "--node-port", "7860"]
