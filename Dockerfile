FROM python:3.14-slim

WORKDIR /app

COPY pyproject.toml README.md ./
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.14.1+cpu

COPY src ./src

RUN pip install --no-cache-dir . \
    && python -m nltk.downloader punkt_tab \
    && python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-large-en-v1.5')"

ENV PORT=5000
EXPOSE 5000

CMD ["bruno"]
