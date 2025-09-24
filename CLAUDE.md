# ColBERT集成到HiRAG项目分析与实施计划

## 项目背景

本项目旨在将ColBERT检索系统集成到HiRAG（Hierarchical Retrieval-Augmented Generation）系统中，以提供基于token级embedding的高效检索能力。

## ColBERT存储机制深度理解

### ColBERT索引存储完整流程

**Indexer.index()方法分析**：
1. **参数输入**：
   - `name`: 索引名称
   - `collection`: Collection对象（从TSV文件：`pid\tpassage_text`格式加载）
   - `overwrite`: 覆盖策略

2. **索引路径确定**：
   - `self.config.index_path_` = `{index_root}/{index_name}/`
   - 所有索引文件存储在这个路径下

3. **CollectionIndexer核心流程**：
   - **setup()**: 创建`plan.json`，包含索引规划信息
   - **train()**: 训练centroids，保存`sample.{rank}.pt`文件
   - **index()**: 编码所有tokens，保存压缩embedding数据
   - **finalize()**: 构建元数据和倒排索引

**索引存储文件结构**：
```
{index_root}/{index_name}/
├── plan.json                    # 索引规划配置
├── metadata.json               # 整体索引元数据
├── {chunk_idx}.metadata.json   # 每个chunk的元数据
├── doclens.{chunk_idx}.json    # 文档长度信息
├── {chunk_idx}.codes.pt        # 压缩的embedding数据
├── {chunk_idx}.residuals.pt    # residual数据
├── centroids.pt                # 聚类中心
├── ivf.pid.pt                  # 倒排索引文件
└── avg_residual.pt            # 平均residual
```

### 关键配置参数理解

**ColBERTConfig中的关键设置**：
- `index_root`: 索引根目录路径
- `index_name`: 具体索引名称
- `collection`: Collection对象路径
- `checkpoint`: ColBERT模型checkpoint路径
- `doc_maxlen`: 文档最大token长度
- `nbits`: 量化位数
- `index_bsize`: 索引批处理大小

## 重新设计的集成方案

### 1. ColBERT API服务器扩展 (扩展server.py)

**完全复用ColBERT存储机制的API设计**：

#### 索引创建API
**POST /api/create-index**
```json
{
  "index_name": "workspace_kb_chunks",
  "collection_tsv_data": "doc1_chunk1\tpassage content 1\ndoc1_chunk2\tpassage content 2\n...",
  "hirag_metadata": {
    "doc1_chunk1": {"documentKey": "doc1_chunk1", "workspaceId": "ws1", ...},
    "doc1_chunk2": {"documentKey": "doc1_chunk2", "workspaceId": "ws1", ...}
  },
  "config": {
    "checkpoint": "/path/to/colbert/model",
    "doc_maxlen": 300,
    "nbits": 2
  }
}
```

**API实现流程**：
1. 接收TSV格式的collection数据和HiRAG元数据
2. 创建临时TSV文件供ColBERT使用
3. 单独保存HiRAG元数据到`{index_path}/hirag_metadata.json`
4. 使用ColBERT原生Indexer创建索引
5. 返回索引路径

#### 索引追加API
**POST /api/add-to-index**
```json
{
  "index_name": "workspace_kb_chunks",
  "collection_tsv_data": "...",
  "hirag_metadata": {...}
}
```

#### 检索API扩展
**GET /api/search**
```
参数: query, k, index_name
返回: {
  "query": "search text",
  "topk": [
    {
      "pid": "doc1_chunk1",
      "score": 0.95,
      "rank": 1,
      "text": "passage content",
      "hirag_metadata": {...}  // 从hirag_metadata.json恢复
    }
  ]
}
```

### 2. HiRAG ColBERTVDB实现 (src/hirag_prod/storage/colbert_vdb.py)

**完全作为ColBERT API客户端的设计**：

#### 核心方法实现逻辑

**upsert_texts()实现**：
```python
async def upsert_texts(self, texts_to_upsert, properties_list, table_name, ...):
    # 1. 构造索引名（多租户隔离）
    index_name = self._build_index_name(properties_list, table_name)

    # 2. 转换为ColBERT TSV格式
    tsv_data = ""
    hirag_metadata = {}
    for text, props in zip(texts_to_upsert, properties_list):
        pid = props.get("documentKey")
        tsv_data += f"{pid}\t{text}\n"
        hirag_metadata[pid] = props

    # 3. 调用ColBERT API，完全使用ColBERT存储
    if await self._index_exists(index_name):
        await self._add_to_index(index_name, tsv_data, hirag_metadata)
    else:
        await self._create_index(index_name, tsv_data, hirag_metadata)
```

**query()实现**：
```python
async def query(self, query, workspace_id, knowledge_base_id, table_name, topk, ...):
    # 1. 构造索引名
    index_name = f"{workspace_id}_{knowledge_base_id}_{table_name}"

    # 2. 调用ColBERT搜索API
    results = await self._search_api(query, index_name, topk)

    # 3. 结果已包含hirag_metadata，直接转换格式返回
    return self._convert_to_hirag_format(results)
```

### 3. ColBERT服务端实现要点

#### 服务器扩展实现 (server.py)
```python
from colbert import Indexer, Searcher
from colbert.infra import ColBERTConfig, RunConfig, Run
from colbert.data import Collection
import tempfile
import ujson
import os

# 全局配置
COLBERT_CHECKPOINT = os.getenv("COLBERT_CHECKPOINT")
INDEX_ROOT = os.getenv("INDEX_ROOT")

def create_index_endpoint():
    """完全使用ColBERT原生索引机制"""
    data = request.get_json()
    index_name = data['index_name']
    tsv_data = data['collection_tsv_data']
    hirag_metadata = data['hirag_metadata']
    config_params = data.get('config', {})

    # 1. 创建临时TSV文件
    with tempfile.NamedTemporaryFile(mode='w', suffix='.tsv', delete=False) as f:
        f.write(tsv_data)
        tsv_path = f.name

    try:
        # 2. 使用ColBERT原生配置和Indexer
        with Run().context(RunConfig(experiment='hirag')):
            config = ColBERTConfig(
                index_root=INDEX_ROOT,
                checkpoint=COLBERT_CHECKPOINT,
                **config_params
            )

            # 3. 创建Collection和Indexer
            collection = Collection(path=tsv_path)
            indexer = Indexer(checkpoint=COLBERT_CHECKPOINT, config=config)

            # 4. 执行索引创建 - 完全使用ColBERT机制
            index_path = indexer.index(name=index_name, collection=collection, overwrite=True)

            # 5. 保存HiRAG元数据到索引目录
            metadata_path = os.path.join(index_path, 'hirag_metadata.json')
            with open(metadata_path, 'w') as f:
                ujson.dump(hirag_metadata, f)

        return {"status": "success", "index_path": index_path}

    finally:
        os.unlink(tsv_path)

def search_endpoint():
    """扩展现有搜索，返回HiRAG元数据"""
    query = request.args.get('query')
    k = int(request.args.get('k', 10))
    index_name = request.args.get('index_name')

    # 1. 使用现有searcher逻辑
    with Run().context(RunConfig(experiment='hirag')):
        searcher = Searcher(index=index_name, index_root=INDEX_ROOT)
        pids, ranks, scores = searcher.search(query, k=k)

    # 2. 加载HiRAG元数据
    index_path = os.path.join(INDEX_ROOT, index_name)
    metadata_path = os.path.join(index_path, 'hirag_metadata.json')

    hirag_metadata = {}
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            hirag_metadata = ujson.load(f)

    # 3. 构造返回结果
    topk = []
    for pid, rank, score in zip(pids, ranks, scores):
        result = {
            'pid': str(pid),
            'rank': rank,
            'score': float(score),
            'text': searcher.collection[pid],
            'hirag_metadata': hirag_metadata.get(str(pid), {})
        }
        topk.append(result)

    return {"query": query, "topk": topk}
```

### 4. 索引命名和隔离策略

**命名规范**：`{workspace_id}_{knowledge_base_id}_{table_name}`
- 确保多租户完全隔离
- 每个索引在ColBERT端有独立的存储目录

**存储路径结构**：
```
{INDEX_ROOT}/
├── ws1_kb1_chunks/           # 租户1的chunks索引
│   ├── plan.json
│   ├── metadata.json
│   ├── hirag_metadata.json   # HiRAG元数据
│   └── ...                   # 其他ColBERT索引文件
├── ws1_kb1_items/            # 租户1的items索引
└── ws2_kb1_chunks/           # 租户2的chunks索引
```

### 5. 配置和部署

**ColBERT服务配置** (.env):
```
INDEX_ROOT=/data/colbert_indices
COLBERT_CHECKPOINT=/models/colbertv2.0
PORT=8893
```

**无需修改HiRAG配置**：
- HiRAG端只需知道ColBERT API地址
- 完全通过API交互，无需配置ColBERT相关参数

### 6. 关键优势

**完全复用ColBERT存储**：
- 使用ColBERT原生的Indexer和存储机制
- 保持ColBERT的所有优化特性（residual quantization、faiss索引等）
- 支持ColBERT的分布式索引能力

**最小侵入性**：
- HiRAG端只需实现ColBERTVDB作为API客户端
- 无需修改HiRAG的存储配置和架构
- ColBERT端只需扩展API，核心逻辑不变

**数据隔离和元数据保持**：
- 通过索引命名实现多租户隔离
- HiRAG元数据单独存储但与ColBERT索引关联
- 检索时完整恢复HiRAG所需的所有字段

这个方案完全遵循ColBERT的存储机制，同时实现了与HiRAG的无缝集成。