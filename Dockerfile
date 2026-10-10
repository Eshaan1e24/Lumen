FROM python:3.13-slim
# One thread per numeric library: statsmodels' small linear algebra is much slower when BLAS spawns many threads,
# and a free-tier host has little CPU to share.
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PORT=8000
WORKDIR /srv
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN useradd --create-home --uid 10001 lumen && chown -R lumen /srv
USER lumen
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s CMD python -c "import os,urllib.request as u; u.urlopen('http://127.0.0.1:%s/api/health' % os.environ.get('PORT','8000'), timeout=4)" || exit 1
# A single worker on purpose: sessions live in this process's memory (nothing is written to disk).
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT} --workers 1"]
