"""
ColBERT向量数据库实现

作为ColBERT API的客户端，完全使用ColBERT的存储机制进行索引和检索。
索引数据存储在ColBERT端，通过API进行交互。
"""

import aiohttp
import asyncio
import logging
from typing import List, Dict, Any, Optional, Literal, Union
from hirag_prod.storage.base_vdb import BaseVDB

logger = logging.getLogger(__name__)


class ColBERTVDB(BaseVDB):
    """ColBERT向量数据库适配器，作为ColBERT API的客户端"""

    def __init__(self, colbert_host: str = "localhost", colbert_port: int = 8893, **config):
        """
        初始化ColBERT VDB客户端

        Args:
            colbert_host: ColBERT服务器地址
            colbert_port: ColBERT服务器端口
            **config: ColBERT配置参数（doc_maxlen, nbits等）
        """
        self.base_url = f"http://{colbert_host}:{colbert_port}"
        self.config = config
        self.embedding_func = None  # ColBERT不需要外部embedding函数

    async def _init_vdb(self, *args, **kwargs):
        """初始化VDB连接"""
        # 检查ColBERT服务是否可用
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/api/status") as response:
                    if response.status == 200:
                        logger.info("ColBERT service is healthy")
                    else:
                        raise Exception(f"ColBERT service returned status {response.status}")
        except Exception as e:
            logger.error(f"Failed to connect to ColBERT service: {e}")
            raise

    async def upsert_texts(
        self,
        texts_to_upsert: List[str],
        properties_list: List[dict],
        table_name: str,
        with_tokenization: bool = False,
        with_translation: bool = False,
        mode: Literal["append", "overwrite"] = "append",
    ):
        """
        将HiRAG chunks转换为ColBERT格式并建索引

        Args:
            texts_to_upsert: 要索引的文本列表
            properties_list: 对应的HiRAG元数据列表
            table_name: 表名（Chunks/Items/Triplets）
            mode: 索引模式（目前只支持overwrite，ColBERT原生不支持增量）
        """
        if len(texts_to_upsert) != len(properties_list):
            raise ValueError("texts_to_upsert and properties_list must have the same length")

        # 1. 构造索引名称（确保多租户隔离）
        index_name = self._build_index_name(properties_list, table_name)
        logger.info(f"Creating ColBERT index: {index_name}")

        # 2. 转换为ColBERT TSV格式
        tsv_data = ""
        hirag_metadata = {}

        for text, props in zip(texts_to_upsert, properties_list):
            pid = props.get("documentKey", f"chunk_{len(hirag_metadata)}")
            # 清理文本，确保TSV格式正确
            clean_text = text.replace('\t', ' ').replace('\n', ' ').replace('\r', ' ').strip()
            tsv_data += f"{pid}\t{clean_text}\n"
            hirag_metadata[pid] = props

        # 3. 调用ColBERT API创建索引
        payload = {
            "index_name": index_name,
            "collection_tsv_data": tsv_data,
            "hirag_metadata": hirag_metadata,
            "config": self.config
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/api/create-index",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=3600)  # 索引可能需要较长时间
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        logger.info(f"Index created successfully: {result.get('index_path')}")
                    else:
                        error_text = await response.text()
                        raise Exception(f"ColBERT API error ({response.status}): {error_text}")

        except Exception as e:
            logger.error(f"Failed to create ColBERT index: {e}")
            raise

    async def query(
        self,
        query: Union[str, List[str]],
        workspace_id: str,
        knowledge_base_id: str,
        table_name: str,
        topk: Optional[int] = None,
        uri_list: Optional[List[str]] = None,
        require_access: Optional[Literal["private", "public"]] = None,
        columns_to_select: Optional[List[str]] = None,
        distance_threshold: Optional[float] = None,
        topn: Optional[int] = None,
        rerank: bool = False,
    ) -> List[dict]:
        """
        调用ColBERT检索并转换结果格式

        Args:
            query: 查询文本
            workspace_id: 工作空间ID
            knowledge_base_id: 知识库ID
            table_name: 表名
            topk: 返回结果数量
            其他参数: 过滤条件

        Returns:
            List[dict]: HiRAG格式的检索结果
        """
        # 1. 构造索引名称
        index_name = f"{workspace_id}_{knowledge_base_id}_{table_name}"

        # 2. 处理查询（支持单个或多个查询）
        queries = [query] if isinstance(query, str) else query
        all_results = []

        for q in queries:
            try:
                # 3. 调用ColBERT搜索API
                params = {
                    "query": q,
                    "k": topk or 10,
                    "index_name": index_name
                }

                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        f"{self.base_url}/api/search",
                        params=params
                    ) as response:
                        if response.status == 200:
                            response_data = await response.json()
                        else:
                            error_text = await response.text()
                            logger.warning(f"Search failed for index {index_name}: {error_text}")
                            continue

                # 4. 转换结果格式：ColBERT -> HiRAG
                results = []
                for item in response_data.get("topk", []):
                    # 恢复原始HiRAG元数据
                    chunk_data = item.get("hirag_metadata", {}).copy()
                    chunk_data.update({
                        "relevance_score": float(item.get("score", 0.0)),
                        "text": item.get("text", ""),
                        "documentKey": item.get("pid", ""),
                    })

                    # 应用过滤条件
                    if self._should_include_result(chunk_data, uri_list, require_access, distance_threshold):
                        results.append(chunk_data)

                all_results.extend(results)

            except Exception as e:
                logger.error(f"Search error for query '{q}': {e}")
                continue

        return all_results

    async def query_by_keys(
        self,
        key_value: List[str],
        workspace_id: str,
        knowledge_base_id: str,
        table_name: str,
        key_column: str = "documentKey",
        columns_to_select: Optional[List[str]] = None,
        limit: Optional[int] = None,
    ) -> List[dict]:
        """
        根据文档key查询（通过特殊查询实现）

        注意：这是一个简化实现，实际可能需要更复杂的逻辑
        """
        # 构造一个包含所有key的查询
        query_text = " ".join(key_value)
        results = await self.query(
            query=query_text,
            workspace_id=workspace_id,
            knowledge_base_id=knowledge_base_id,
            table_name=table_name,
            topk=limit or len(key_value) * 2
        )

        # 过滤出匹配的key
        filtered_results = []
        for result in results:
            if result.get(key_column) in key_value:
                filtered_results.append(result)

        return filtered_results[:limit] if limit else filtered_results

    def _build_index_name(self, properties_list: List[dict], table_name: str) -> str:
        """构造索引名称，确保多租户隔离"""
        if not properties_list:
            raise ValueError("properties_list cannot be empty")

        first_prop = properties_list[0]
        workspace_id = first_prop.get("workspaceId", "default")
        knowledge_base_id = first_prop.get("knowledgeBaseId", "default")

        # 清理名称，确保文件系统安全
        workspace_id = workspace_id.replace("/", "_").replace("\\", "_")
        knowledge_base_id = knowledge_base_id.replace("/", "_").replace("\\", "_")
        table_name = table_name.replace("/", "_").replace("\\", "_")

        return f"{workspace_id}_{knowledge_base_id}_{table_name}"

    def _should_include_result(
        self,
        chunk_data: dict,
        uri_list: Optional[List[str]],
        require_access: Optional[Literal["private", "public"]],
        distance_threshold: Optional[float]
    ) -> bool:
        """应用过滤条件"""
        # URI过滤
        if uri_list and chunk_data.get("uri") not in uri_list:
            return False

        # 访问权限过滤
        if require_access:
            is_private = chunk_data.get("private", False)
            if require_access == "private" and not is_private:
                return False
            if require_access == "public" and is_private:
                return False

        # 相关性分数过滤
        if distance_threshold and chunk_data.get("relevance_score", 0.0) < distance_threshold:
            return False

        return True

    # 以下方法为BaseVDB接口要求，但ColBERT不需要实现
    async def get_existing_document_keys(self, *args, **kwargs) -> List[str]:
        """获取现有文档keys（ColBERT暂不支持）"""
        return []

    async def delete_by_document_keys(self, *args, **kwargs):
        """根据文档key删除（ColBERT暂不支持）"""
        pass

    async def get_table_info(self, *args, **kwargs) -> dict:
        """获取表信息（ColBERT暂不支持）"""
        return {}