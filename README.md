# HiRAG-MCP

## Quick Start

**Option 1: Docker Deployment**
Run the following command to start the container, before that you should install docker and docker-compose.
```bash
git clone https://github.com/Kasma-Inc/hirag-prod
cd hirag-prod
HIRAG_PROD_DIR="." docker compose -p $(whoami)_hirag-prod-compose up -d
```
Then use the following command to enter the container:
```bash
docker exec -it $(whoami)_hirag-prod /bin/bash
```
or use VSCode to connect to the container.

Then create the virtual environment and install dependencies using uv as below:
```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e .
```

Then create the `.env` file and replace the placeholders with your own values:
```bash
cp .env.example .env
source .env
```

Run the script:
```bash

python main.py
```

**Option 2: Local Deployment**
Make sure uv is installed, if not:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then create the virtual environment and install dependencies using uv as below:
```bash
uv venv
source .venv/bin/activate
uv pip install -e .
```

Then create the `.env` file and replace the placeholders with your own values:
```bash
cp .env.example .env
```

Run the script:
```bash
python main.py
```

## Using a Remote ColBERT Vector Service

HiRAG can delegate vector indexing and retrieval to a remote ColBERT service. This is
useful when the local Docker environment has limited GPU or CPU resources.

1. **Configure environment variables**
   - Set `VDB_TYPE=colbert_remote` in your `.env` file.
   - Provide the remote service details:
     ```ini
     COLBERT_BASE_URL="https://colbert.example.com"
     COLBERT_API_KEY="your-api-key"    # optional
     COLBERT_INDEX_PREFIX="hirag"
     COLBERT_DEFAULT_INDEX_NAME="demo-index"  # optional
     COLBERT_TIMEOUT=30
     COLBERT_MAX_RETRIES=3
     COLBERT_RETRY_BACKOFF_SECONDS=1.0
     ```

2. **Start the ColBERT backend** on a server with sufficient resources. The service
   should expose REST endpoints compatible with the new `ColbertRemoteVDB`, including
   `/vdb/init`, `/vdb/{table}/upsert`, `/vdb/{table}/query`, `/vdb/{table}/query_by_keys`,
   `/vdb/{table}/delete`, `/vdb/{table}/keys`, `/vdb/graph/pagerank`, and `/vdb/graph/node`.

3. **Run HiRAG locally** as usual (`python main.py`). The ingestion and query pipelines
   will automatically route vector operations to the remote ColBERT API while keeping
   metadata, graph operations, and reranking logic within the existing HiRAG services.

When running with `VDB_TYPE=colbert_remote`, the `ResourceManager` maintains a shared
`httpx.AsyncClient` configured with the ColBERT settings, and `ColbertRemoteVDB` converts
ingestion/query calls into the appropriate HTTP requests.
