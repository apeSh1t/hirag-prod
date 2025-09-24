from flask import Flask, render_template, request, jsonify
from functools import lru_cache
import math
import os
import tempfile
import ujson
from dotenv import load_dotenv

from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert import Searcher, Indexer
from colbert.data import Collection

load_dotenv()

# 配置参数
INDEX_NAME = os.getenv("INDEX_NAME")
INDEX_ROOT = os.getenv("INDEX_ROOT")
COLBERT_CHECKPOINT = os.getenv("COLBERT_CHECKPOINT", "downloads/colbertv2.0")
app = Flask(__name__)

# 现有搜索器（如果INDEX_NAME已设置）
searcher = None
if INDEX_NAME:
    searcher = Searcher(index=INDEX_NAME, index_root=INDEX_ROOT)

counter = {"api" : 0}

# 存储活跃的搜索器实例，避免重复初始化
searchers_cache = {}

@lru_cache(maxsize=1000000)
def api_search_query(query, k, index_name):
    """执行ColBERT搜索并返回包含HiRAG元数据的结果"""
    print(f"Query={query}, Index={index_name}")
    if k == None: k = 10
    k = min(int(k), 100)

    try:
        # 获取或创建搜索器
        current_searcher = get_or_create_searcher(index_name)

        # 执行搜索
        pids, ranks, scores = current_searcher.search(query, k=100)
        pids, ranks, scores = pids[:k], ranks[:k], scores[:k]

        # 加载HiRAG元数据
        hirag_metadata = load_hirag_metadata(index_name)

        # 构造结果
        topk = []
        for pid, rank, score in zip(pids, ranks, scores):
            text = current_searcher.collection[pid]
            result = {
                'text': text,
                'pid': str(pid),
                'rank': rank,
                'score': float(score),
                'hirag_metadata': hirag_metadata.get(str(pid), {})
            }
            topk.append(result)

        topk = list(sorted(topk, key=lambda p: (-1 * p['score'], p['pid'])))
        return {"query" : query, "topk": topk}

    except Exception as e:
        print(f"Search error: {e}")
        return {"error": str(e)}, 500

@app.route("/api/search", methods=["GET"])
def api_search():
    if request.method == "GET":
        counter["api"] += 1
        print("API request count:", counter["api"])

        # 支持指定索引名称
        index_name = request.args.get("index_name", INDEX_NAME)
        if not index_name:
            return jsonify({"error": "index_name is required"}), 400

        return api_search_query(
            request.args.get("query"),
            request.args.get("k"),
            index_name
        )
    else:
        return ('', 405)

def get_or_create_searcher(index_name):
    """获取或创建搜索器实例"""
    if index_name not in searchers_cache:
        try:
            with Run().context(RunConfig(experiment='hirag')):
                searchers_cache[index_name] = Searcher(
                    index=index_name,
                    index_root=INDEX_ROOT
                )
        except Exception as e:
            raise Exception(f"Failed to create searcher for {index_name}: {e}")
    return searchers_cache[index_name]

def load_hirag_metadata(index_name):
    """加载指定索引的HiRAG元数据"""
    try:
        index_path = os.path.join(INDEX_ROOT, index_name)
        metadata_path = os.path.join(index_path, 'hirag_metadata.json')

        if os.path.exists(metadata_path):
            with open(metadata_path, 'r', encoding='utf-8') as f:
                return ujson.load(f)
        return {}
    except Exception as e:
        print(f"Warning: Failed to load HiRAG metadata for {index_name}: {e}")
        return {}

@app.route("/api/create-index", methods=["POST"])
def create_index():
    """创建新的ColBERT索引，完全使用ColBERT原生存储机制"""
    try:
        data = request.get_json()
        index_name = data['index_name']
        tsv_data = data['collection_tsv_data']
        hirag_metadata = data['hirag_metadata']
        config_params = data.get('config', {})

        print(f"Creating index: {index_name}")

        # 创建临时TSV文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.tsv', delete=False, encoding='utf-8') as f:
            f.write(tsv_data)
            tsv_path = f.name

        try:
            # 使用ColBERT原生配置和Indexer
            with Run().context(RunConfig(experiment='hirag')):
                config = ColBERTConfig(
                    index_root=INDEX_ROOT,
                    checkpoint=COLBERT_CHECKPOINT,
                    doc_maxlen=config_params.get('doc_maxlen', 300),
                    nbits=config_params.get('nbits', 2),
                    **{k: v for k, v in config_params.items() if k not in ['doc_maxlen', 'nbits']}
                )

                # 创建Collection和Indexer
                collection = Collection(path=tsv_path)
                indexer = Indexer(checkpoint=COLBERT_CHECKPOINT, config=config)

                # 执行索引创建 - 完全使用ColBERT机制
                index_path = indexer.index(name=index_name, collection=collection, overwrite=True)

                # 保存HiRAG元数据到索引目录
                metadata_path = os.path.join(index_path, 'hirag_metadata.json')
                with open(metadata_path, 'w', encoding='utf-8') as f:
                    ujson.dump(hirag_metadata, f, ensure_ascii=False, indent=2)

                # 清理搜索器缓存，确保下次使用新索引
                if index_name in searchers_cache:
                    del searchers_cache[index_name]

                print(f"Index created successfully: {index_path}")
                return jsonify({"status": "success", "index_path": index_path})

        finally:
            os.unlink(tsv_path)

    except Exception as e:
        print(f"Error creating index: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/add-to-index", methods=["POST"])
def add_to_index():
    """向现有索引添加数据（暂不实现，ColBERT原生不直接支持增量索引）"""
    return jsonify({"error": "Incremental indexing not yet implemented"}), 501

@app.route("/api/delete-index/<index_name>", methods=["DELETE"])
def delete_index(index_name):
    """删除指定索引"""
    try:
        index_path = os.path.join(INDEX_ROOT, index_name)
        if os.path.exists(index_path):
            import shutil
            shutil.rmtree(index_path)

            # 清理缓存
            if index_name in searchers_cache:
                del searchers_cache[index_name]

            return jsonify({"status": "deleted", "index_name": index_name})
        else:
            return jsonify({"error": "Index not found"}), 404

    except Exception as e:
        print(f"Error deleting index: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/list-indices", methods=["GET"])
def list_indices():
    """列出所有可用索引"""
    try:
        if not os.path.exists(INDEX_ROOT):
            return jsonify({"indices": []})

        indices = []
        for item in os.listdir(INDEX_ROOT):
            item_path = os.path.join(INDEX_ROOT, item)
            if os.path.isdir(item_path) and os.path.exists(os.path.join(item_path, 'metadata.json')):
                indices.append(item)

        return jsonify({"indices": indices})

    except Exception as e:
        print(f"Error listing indices: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/status", methods=["GET"])
def status():
    """服务健康检查"""
    return jsonify({
        "status": "healthy",
        "version": "2.0",
        "index_root": INDEX_ROOT,
        "checkpoint": COLBERT_CHECKPOINT
    })

if __name__ == "__main__":
    print(f"Starting ColBERT server with INDEX_ROOT={INDEX_ROOT}")
    print(f"COLBERT_CHECKPOINT={COLBERT_CHECKPOINT}")
    app.run("0.0.0.0", int(os.getenv("PORT", 8893)))

