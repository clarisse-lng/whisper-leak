FROM python:3.11-slim

# tcpdump is required for packet capture (the script calls it directly via subprocess)
RUN apt-get update && apt-get install -y --no-install-recommends \
        tcpdump \
        tshark \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy pre-downloaded wheels first (built on host via download_packages.sh)
# so the image build never hits PyPI.
COPY pip_packages/ ./pip_packages/
COPY requirements.txt .
RUN pip install --no-index --find-links=./pip_packages -r requirements.txt \
    && rm -rf pip_packages

# Copy application source
COPY . .

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
# Default prompts file — override per-service in docker-compose.yml
CMD ["-c", "GPT4oMiniObfuscation", "-p", "/app/prompts/standard/prompts.json"]

