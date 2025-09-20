# ColBERT 远端服务示例

本目录提供一个可独立部署的 FastAPI 服务，用于承接 HiRAG 在「`colbert_remote`」模式下发起的所有向量库与图谱操作。服务默认以内存为存储介质，方便在本地或临时环境中演示与联调，同时也预留了切换到真实 ColBERT 模型推理的接口。

## 功能概览

- **向量写入与查询**：兼容 `/vdb/{table}/upsert`、`/vdb/{table}/query`、`/vdb/{table}/query_by_keys` 等接口，可处理 `Chunks`、`Items`、`Triplets`、`Files` 等表结构。
- **图谱操作**：支持 `Graph` 表的关系写入、按条件清理，并提供 `pagerank` 与 `query_node` 接口供知识图谱增强使用。
- **认证与日志**：内置 Bearer Token 校验开关，以及可选的查询日志输出，便于排障与审计。
- **编码后端多样化**：默认使用轻量级哈希向量以避免额外依赖，可通过环境变量切换到 `sentence-transformer` 或真正的 `ColBERT` 推理后端。

## 环境依赖

- Python ≥ 3.10
- FastAPI、Uvicorn 等基础依赖，可通过 `pip install -r requirements.txt` 安装
- 如需使用 SentenceTransformer 或 ColBERT，请分别安装 `sentence-transformers`、`colbert-ai` 及对应的 GPU/CPU 依赖

示例 `requirements.txt`：

```text
fastapi
uvicorn[standard]
pydantic-settings
numpy
```

若需切换编码后端，可额外安装：

```text
sentence-transformers  # backend=sentence-transformer
colbert-ai             # backend=colbert
```

## 配置项

服务使用 `COLBERT_SERVER_` 前缀的环境变量进行配置：

| 变量名 | 默认值 | 说明 |
| --- | --- | --- |
| `COLBERT_SERVER_API_KEY` | 空 | 若设置，将启用 Bearer Token 鉴权 |
| `COLBERT_SERVER_BACKEND` | `hashing` | 向量编码后端，可选 `hashing` / `sentence-transformer` / `colbert` |
| `COLBERT_SERVER_MODEL_NAME` | `sentence-transformers/all-MiniLM-L6-v2` | 非 hashing 模式下使用的模型或 checkpoint |
| `COLBERT_SERVER_DEVICE` | 空 | 指定运行设备，如 `cuda:0`，留空则自动推断 |
| `COLBERT_SERVER_FALLBACK_DIMENSION` | `384` | 哈希编码的维度，同时作为兜底值 |
| `COLBERT_SERVER_DEFAULT_TOP_K` | `20` | 查询默认返回条目数 |
| `COLBERT_SERVER_LOG_QUERIES` | `false` | 是否在日志中打印查询请求 |
| `COLBERT_SERVER_ALLOW_CORS` | `false` | 是否对所有来源开放 CORS（开发环境建议开启） |
| `COLBERT_SERVER_MAX_BATCH_SIZE` | `128` | 单次 upsert 允许的最大记录数 |

## 运行方式

1. 准备虚拟环境并安装依赖：

   ```bash
   cd colbert_remote_service
   pip install -r requirements.txt
   ```

2. 配置 `.env` 或直接导出环境变量，例如：

   ```bash
   export COLBERT_SERVER_API_KEY=your-secret
   export COLBERT_SERVER_BACKEND=sentence-transformer
   export COLBERT_SERVER_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
   ```

3. 启动服务：

   ```bash
   uvicorn colbert_remote_service.main:app --host 0.0.0.0 --port 8001
   ```

4. 在本地 `.env` 中配置 HiRAG：

   ```env
   VDB_TYPE=colbert_remote
   COLBERT_BASE_URL=http://<server-ip>:8001
   COLBERT_API_KEY=your-secret
   ```

5. 启动本地 HiRAG（Docker 或裸机均可），即可将分块写入远端服务，并通过 `/healthz` 检查状态。

## 接口速览

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| `GET /healthz` | 健康检查 |
| `POST /vdb/init` | 初始化索引参数（嵌入维度等） |
| `POST /vdb/{table}/upsert` | 写入向量或文件元数据，`table` 取 `Chunks`/`Items`/`Files` 等 |
| `POST /vdb/{table}/query` | 执行相似度检索，结果中包含 `distance` 字段 |
| `POST /vdb/{table}/query_by_keys` | 通过 `documentKey` 批量读取记录 |
| `POST /vdb/{table}/keys` | 查询某 URI 已存在的切片键 |
| `POST /vdb/{table}/delete` | 按条件删除记录 |
| `POST /vdb/graph/pagerank` | 计算 PageRank，返回 `documentKey` 与分值 |
| `POST /vdb/graph/node` | 查询图谱节点的邻接信息及关联切片 |

所有写入接口均支持 `mode=overwrite`，便于增量更新。若开启 `COLBERT_SERVER_API_KEY`，请在请求头中附带 `Authorization: Bearer <token>`。

## 与真实 ColBERT 模型集成

- 将 `COLBERT_SERVER_BACKEND` 设置为 `colbert`，并确保已安装官方 `colbert-ai` 包及其依赖（Faiss、PyTorch 等）。
- 在 `COLBERT_SERVER_MODEL_NAME` 中填写 checkpoint 名称或本地路径，必要时通过 `COLBERT_SERVER_DEVICE` 指定 GPU。
- 若需持久化索引，可在 `store.py` 中替换内存实现，接入自建的向量存储或检索器。

## 开发提示

- 目前服务默认在内存中维护所有数据，适合演示与小规模联调，生产环境请替换为持久化实现。
- PageRank 与图节点查询基于简单的有向图建模，如需更复杂的知识图谱能力，可在 `GraphStore` 中接入专用图数据库。
- 若需扩展监控与日志，可在 FastAPI 中加入中间件或集成 Prometheus/OpenTelemetry。

欢迎根据业务需求自定义存储、向量模型或安全策略。若在接入过程中遇到问题，可结合 `log_queries` 与应用日志进行排查。
