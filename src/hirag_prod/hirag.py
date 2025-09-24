import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

import numpy as np
import pandas as pd
from docling_core.types.doc import DoclingDocument

from hirag_prod._utils import (
    compute_mdhash_id,
    log_error_info,
)
from hirag_prod.chunk import BaseChunk, FixTokenChunk
from hirag_prod.configs.cli_options import CliOptions
from hirag_prod.configs.functions import (
    get_config_manager,
    get_hi_rag_config,
    get_llm_config,
    initialize_config_manager,
)
from hirag_prod.entity import BaseKG, VanillaKG
from hirag_prod.exceptions import (
    DocumentProcessingError,
    HiRAGException,
    KGConstructionError,
)
from hirag_prod.loader import load_document
from hirag_prod.loader.chunk_split import (
    build_rich_toc,
    chunk_docling_document,
    chunk_dots_document,
    chunk_langchain_document,
    items_to_chunks_recursive,
    obtain_docling_md_bbox,
)
from hirag_prod.loader.utils import route_file_path, validate_document_path
from hirag_prod.metrics import MetricsCollector, ProcessingMetrics
from hirag_prod.parser import DictParser, ReferenceParser
from hirag_prod.prompt import PROMPTS
from hirag_prod.resources.functions import (
    get_chat_service,
    get_embedding_service,
    get_qwen_translator,
    get_translator,
    initialize_resource_manager,
)
from hirag_prod.resume_tracker import JobStatus, ResumeTracker
from hirag_prod.schema import Chunk, File, Item, LoaderType, file_to_item, item_to_chunk
from hirag_prod.storage import (
    BaseGDB,
    BaseVDB,
    LanceDB,
    NetworkXGDB,
    RetrievalStrategyProvider,
)
from hirag_prod.storage.pgvector import PGVector
from hirag_prod.storage.query_service import QueryService
from hirag_prod.storage.storage_manager import StorageManager

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("HiRAG")


# ============================================================================
# Document processor
# ============================================================================


class DocumentProcessor:
    """Document processor for handling document ingestion pipeline"""

    def __init__(
        self,
        storage: StorageManager,
        chunker: BaseChunk,
        kg_constructor: BaseKG,
        resume_tracker: Optional[ResumeTracker] = None,
        metrics: Optional[MetricsCollector] = None,
    ):
        self.storage = storage
        self.chunker = chunker
        self.kg_constructor = kg_constructor
        self.resume_tracker = resume_tracker
        self.metrics = metrics or MetricsCollector()

    async def clear_document(
        self,
        document_id: str,
        workspace_id: str,
        knowledge_base_id: str,
    ) -> ProcessingMetrics:

        async with self.metrics.track_operation("clear_document"):
            where_dict = {
                "documentId": document_id,
                "workspaceId": workspace_id,
                "knowledgeBaseId": knowledge_base_id,
            }
            await self.storage.clean_vdb_document(where=where_dict)

            where_dict = {
                "documentKey": document_id,
                "workspaceId": workspace_id,
                "knowledgeBaseId": knowledge_base_id,
            }
            await self.storage.clean_vdb_file(where=where_dict)

        return self.metrics.metrics

    async def process_document(
        self,
        document_path: str,
        content_type: str,
        workspace_id: str,
        knowledge_base_id: str,
        with_graph: bool = True,
        document_meta: Optional[Dict] = None,
        loader_configs: Optional[Dict] = None,
        job_id: Optional[str] = None,
        loader_type: Optional[LoaderType] = None,
    ) -> ProcessingMetrics:
        """Process a single document"""
        async with self.metrics.track_operation(f"process_document"):
            # Load and chunk document
            chunks, file, items = await self._load_and_chunk_document(
                document_path,
                content_type,
                document_meta,
                loader_configs,
                loader_type,
            )

            if not chunks:
                logger.warning("⚠️ No chunks created from document")
                # Mark job failed if tracking is enabled
                if self.resume_tracker and job_id:
                    try:
                        await self.resume_tracker.set_job_failed(
                            job_id, "No chunks created from document"
                        )
                    except Exception as e:
                        log_error_info(
                            logging.ERROR,
                            "Failed to saving job status (failed) to Postgres",
                            e,
                        )
                return self.metrics.metrics

            self.metrics.metrics.total_chunks = len(chunks)
            self.metrics.metrics.job_id = job_id or ""

            # Check if document was already completed in a previous session
            if self.resume_tracker:
                document_id = chunks[0].documentId
                # Update job -> processing with doc info as soon as we know
                if job_id:
                    try:
                        await self.resume_tracker.set_job_processing(
                            job_id=job_id,
                            document_id=document_id,
                            total_chunks=len(chunks),
                        )
                    except Exception as e:
                        log_error_info(
                            logging.ERROR,
                            "Failed to saving job status (processing) to Postgres",
                            e,
                        )

                document_uri = chunks[0].uri
                await self.resume_tracker.register_chunks(
                    chunks,
                    document_id,
                    document_uri,
                    workspace_id,
                    knowledge_base_id,
                )

            # Store file information after chunking but before processing chunks
            await self.storage.upsert_file_to_vdb(file)

            # Process chunks
            await self._process_chunks(chunks, items, workspace_id, knowledge_base_id)
            # Update job progress for processed chunks
            if self.resume_tracker and job_id:
                try:
                    await self.resume_tracker.set_job_progress(
                        job_id,
                        processed_chunks=self.metrics.metrics.processed_chunks,
                    )
                except Exception as e:
                    log_error_info(
                        logging.ERROR,
                        "Failed to saving job status (progress) to Postgres",
                        e,
                    )

            # Process graph data
            if with_graph:
                await self._construct_kg(chunks)
                # Update job progress for entity/relation totals
                if self.resume_tracker and job_id:
                    try:
                        await self.resume_tracker.set_job_progress(
                            job_id,
                            total_entities=self.metrics.metrics.total_entities,
                            total_relations=self.metrics.metrics.total_relations,
                        )
                    except Exception as e:
                        log_error_info(
                            logging.ERROR,
                            "Failed to saving job status (progress) to Postgres",
                            e,
                        )

            # Mark as complete
            if self.resume_tracker:
                await self.resume_tracker.mark_document_completed(
                    document_id=chunks[0].documentId,
                    workspace_id=workspace_id,
                    knowledge_base_id=knowledge_base_id,
                )
                if job_id:
                    try:
                        await self.resume_tracker.set_job_completed(job_id)
                    except Exception as e:
                        log_error_info(
                            logging.ERROR,
                            "Failed to saving job status (completed) to Postgres",
                            e,
                        )

            return self.metrics.metrics

    async def _load_and_chunk_document(
        self,
        document_path: str,
        content_type: str,
        document_meta: Optional[Dict],
        loader_configs: Optional[Dict],
        loader_type: Optional[LoaderType],
    ) -> (List[Chunk], File):  # type: ignore
        """Load and chunk document"""
        async with self.metrics.track_operation("load_and_chunk"):
            generated_md = None
            items = None
            try:
                if content_type == "text/plain":
                    _, generated_md = await asyncio.to_thread(
                        load_document,
                        document_path=document_path,
                        content_type=content_type,
                        document_meta=document_meta,
                        loader_configs=loader_configs,
                        loader_type="langchain",
                    )
                    items = chunk_langchain_document(generated_md)
                    chunks = [
                        item_to_chunk(item) for item in items
                    ]  # Convert items to chunks
                elif (
                    content_type
                    == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                ):
                    try:
                        local_path = route_file_path("excel_loader", document_path)
                    except Exception:
                        local_path = document_path
                    validate_document_path(local_path)

                    all_sheets: Dict[str, pd.DataFrame] = pd.read_excel(
                        local_path, sheet_name=None
                    )

                    def keep_sheet(name: str) -> bool:
                        s = (name or "").lower()
                        return ("cache" not in s) and ("detail" not in s)

                    filtered_sheets = [
                        (name, df)
                        for name, df in all_sheets.items()
                        if keep_sheet(name)
                    ]
                    document_id = document_meta.get("documentKey", "")
                    file_name = document_meta.get(
                        "fileName", os.path.basename(local_path)
                    )
                    generated_md = File(
                        documentKey=document_id,
                        text=file_name,
                        type=document_meta.get("type", "xlsx"),
                        pageNumber=len(filtered_sheets),
                        fileName=file_name,
                        uri=document_meta.get("uri", document_path),
                        private=bool(document_meta.get("private", False)),
                        uploadedAt=document_meta.get("uploadedAt", datetime.now()),
                        knowledgeBaseId=document_meta.get("knowledgeBaseId", ""),
                        workspaceId=document_meta.get("workspaceId", ""),
                    )

                    latex_list = []
                    sheet_names = []
                    for sheet_name, df in filtered_sheets:
                        try:
                            latex_list.append(df.to_latex(index=False))
                        except Exception:
                            latex_list.append(df.to_string(index=False))
                        sheet_names.append(sheet_name)

                    # summarize each sheet latex into a concise caption using LLM
                    async def summarize_excel_sheet(sheet_name: str, latex: str) -> str:
                        system_prompt = PROMPTS["summary_excel_en"].format(
                            sheet_name=sheet_name, latex=latex
                        )
                        try:
                            return await get_chat_service().complete(
                                prompt=system_prompt,
                                model=get_llm_config().model_name,
                            )
                        except Exception:
                            raise HiRAGException(
                                f"Failed to summarize excel sheet {sheet_name}"
                            )

                    captions = await asyncio.gather(
                        *[
                            summarize_excel_sheet(sheet_name, latex)
                            for sheet_name, latex in zip(sheet_names, latex_list)
                        ]
                    )
                    items = []
                    chunks = []

                    for idx, (name, latex, caption) in enumerate(
                        zip(sheet_names, latex_list, captions), start=1
                    ):
                        sheet_key = compute_mdhash_id(
                            f"{document_id}:{name}", prefix="chunk-"
                        )
                        item = file_to_item(
                            generated_md,
                            documentKey=sheet_key,
                            text=(latex or "").strip(),
                            documentId=document_id,
                            chunkIdx=idx,
                        )
                        item.caption = (caption or "None").strip()
                        item.chunkType = "excel_sheet"

                        chunk = item_to_chunk(item)
                        chunk.text = (latex or "None").strip()
                        chunk.caption = (caption or "None").strip()
                        chunk.chunkType = "excel_sheet"

                        items.append(item)
                        chunks.append(chunk)

                else:
                    if (
                        content_type in ["application/pdf", "multimodal/image"]
                        or loader_type == "dots_ocr"
                    ):
                        json_doc, generated_md = await asyncio.to_thread(
                            load_document,
                            document_path=document_path,
                            content_type=content_type,
                            document_meta=document_meta,
                            loader_configs=loader_configs,
                            loader_type="dots_ocr",
                        )

                    else:
                        json_doc, generated_md = await asyncio.to_thread(
                            load_document,
                            document_path=document_path,
                            content_type=content_type,
                            document_meta=document_meta,
                            loader_configs=loader_configs,
                            loader_type="docling",
                        )

                    # summarize each table into a concise caption using LLM
                    async def summarize_table(idx: int):
                        table_item = items[idx]
                        system_prompt = PROMPTS["summary_table_en"].format(
                            table_content=table_item.text
                        )
                        try:
                            caption = await get_chat_service().complete(
                                prompt=system_prompt,
                                model=get_llm_config().model_name,
                            )
                            items[idx].caption = caption
                        except Exception:
                            raise HiRAGException(
                                f"Failed to summarize table {table_item.documentKey}"
                            )

                    # Validate instance, as it may fall back to docling if cloud service unavailable
                    if isinstance(json_doc, list):
                        # Chunk the Dots OCR document
                        items, header_set, table_items_idx = chunk_dots_document(
                            json_doc=json_doc, md_doc=generated_md
                        )

                        await asyncio.gather(
                            *[summarize_table(i) for i in table_items_idx]
                        )

                    elif isinstance(json_doc, DoclingDocument):
                        # Chunk the Docling document
                        items, header_set, table_items_idx = chunk_docling_document(
                            json_doc, generated_md
                        )

                        await asyncio.gather(
                            *[summarize_table(i) for i in table_items_idx]
                        )

                        if content_type == "text/markdown":
                            raw_md = generated_md.text
                            items = obtain_docling_md_bbox(json_doc, raw_md, items)

                    else:
                        raise DocumentProcessingError(
                            "Invalid document format returned by loader"
                        )

                    # Unified chunking method :)
                    chunks = items_to_chunks_recursive(
                        items=items,
                        header_set=header_set,
                    )
                    if generated_md:
                        generated_md.tableOfContents = build_rich_toc(
                            items, generated_md
                        )

                logger.info(
                    f"📄 Created {len(chunks)} chunks from document {document_path}"
                )
                return chunks, generated_md, items

            except Exception as e:
                log_error_info(
                    logging.ERROR,
                    f"Failed to loading document {document_path}",
                    e,
                    raise_error=True,
                    new_error_class=DocumentProcessingError,
                )

    async def _process_chunks(
        self,
        chunks: List[Chunk],
        items: List[Item],
        workspace_id: str,
        knowledge_base_id: str,
    ) -> None:
        """Process chunks for vector storage"""
        async with self.metrics.track_operation("process_chunks"):
            # Get chunks that need processing
            pending_chunks = await self._get_pending_chunks(
                chunks, workspace_id, knowledge_base_id
            )

            if not pending_chunks:
                logger.info("⏭️ All chunks already processed")
                return

            logger.info(f"📤 Processing {len(pending_chunks)} pending chunks...")

            # Batch storage
            await self.storage.upsert_chunks_to_vdb(pending_chunks)
            self.metrics.metrics.processed_chunks += len(pending_chunks)
            await self.storage.upsert_items_to_vdb(items)

            logger.info(f"✅ Processed {len(pending_chunks)} chunks")
            if items:
                logger.info(f"✅ Processed {len(items)} items")
            else:
                logger.info("⚠️ No items to process")

    async def _get_pending_chunks(
        self,
        chunks: List[Chunk],
        workspace_id: str,
        knowledge_base_id: str,
    ) -> List[Chunk]:
        """Get chunks that need processing"""
        if not chunks:
            return []

        if self.resume_tracker:
            # Check for existing chunks in vector database
            uri = chunks[0].uri
            existing_chunk_ids = await self.storage.get_existing_chunks(
                uri, workspace_id, knowledge_base_id
            )
            return [
                chunk for chunk in chunks if chunk.documentKey not in existing_chunk_ids
            ]

        return chunks

    async def _construct_kg(self, chunks: List[Chunk]) -> None:
        """Construct knowledge graph from chunks"""
        logger.info(f"🔍 Constructing knowledge graph from {len(chunks)} chunks...")

        try:
            entities, relations = await self.kg_constructor.construct_kg(chunks)

            if entities:
                self.metrics.metrics.total_entities += len(entities)

            # Store relations to both graph database and vector database
            if relations:
                # use pgvector to mimic graphdb
                await self.storage.vdb.upsert_graph(relations)

                # Store to vector database for semantic search
                await self.storage.upsert_relations_to_vdb(relations)

                self.metrics.metrics.total_relations += len(relations)

            logger.info(
                f"✅ Extracted and stored {len(entities)} entities and {len(relations)} relations"
            )

        except Exception as e:
            log_error_info(
                logging.ERROR,
                "Failed to construct knowledge graph",
                e,
                raise_error=True,
                new_error_class=KGConstructionError,
            )


# ============================================================================
# Main HiRAG class
# ============================================================================


@dataclass
class HiRAG:
    """
    Hierarchical Retrieval-Augmented Generation (HiRAG) system

    Simplified main interface, coordinating the work of all components
    """

    # Components (lazy initialization)
    _storage: Optional[StorageManager] = field(default=None, init=False)
    _processor: Optional[DocumentProcessor] = field(default=None, init=False)
    _query_service: Optional[QueryService] = field(default=None, init=False)
    _metrics: Optional[MetricsCollector] = field(default=None, init=False)
    _kg_constructor: Optional[VanillaKG] = field(default=None, init=False)

    @classmethod
    async def create(
        cls,
        cli_options_dict: Optional[Dict] = None,
        config_dict: Optional[Dict] = None,
        resource_dict: Optional[Dict] = None,
        **kwargs,
    ) -> "HiRAG":
        """Create HiRAG instance"""
        if not cli_options_dict:
            cli_options_dict: Dict = CliOptions().to_dict()
        initialize_config_manager(cli_options_dict, config_dict)
        await initialize_resource_manager(resource_dict)
        instance = cls()
        await instance._initialize(**kwargs)
        return instance

    async def set_language(self, language: str) -> None:
        """Set the language for the HiRAG instance"""
        if language not in get_config_manager().supported_languages:
            raise ValueError(
                f"Unsupported language: {language}. Supported languages: {get_config_manager().supported_languages}"
            )

        get_config_manager().language = language
        self._kg_constructor.update_language_config()

        logger.info(f"Language set to {get_config_manager().language}")

    async def set_db_paths(self, vector_db_path: str, graph_db_path: str) -> None:
        """Set the database paths for the HiRAG instance"""
        get_hi_rag_config().vector_db_path = vector_db_path
        get_hi_rag_config().graph_db_path = graph_db_path

        # Reinitialize storage with new paths
        await self._reinitialize_storage()

        logger.info(
            f"Database paths updated - VDB: {get_hi_rag_config().vector_db_path}, GDB: {get_hi_rag_config().graph_db_path}"
        )

    async def _create_storage_manager(
        self, vdb: Optional[BaseVDB] = None, gdb: Optional[BaseGDB] = None
    ) -> None:
        # Build VDB by type
        if vdb is None:
            if get_hi_rag_config().vdb_type == "lancedb":
                vdb = await LanceDB.create(
                    embedding_func=get_embedding_service().create_embeddings,
                    db_url=get_hi_rag_config().vector_db_path,
                    strategy_provider=RetrievalStrategyProvider(),
                )
            elif get_hi_rag_config().vdb_type == "pgvector":
                vdb = PGVector.create(
                    embedding_func=get_embedding_service().create_embeddings,
                    strategy_provider=RetrievalStrategyProvider(),
                    vector_type="halfvec",
                )

        # Build GDB by type
        if gdb is None:
            if get_hi_rag_config().gdb_type == "networkx":
                gdb = NetworkXGDB.create(
                    path=get_hi_rag_config().graph_db_path,
                    llm_func=get_chat_service().complete,
                )
            elif get_hi_rag_config().gdb_type == "neo4j":
                # Placeholder for future Neo4j adapter
                raise HiRAGException("Neo4j GDB not implemented yet")

        self._storage = StorageManager(
            vdb,
            gdb,
        )
        await self._storage.initialize()

    async def _reinitialize_storage(self) -> None:
        """Reinitialize storage components with current configuration"""
        await self._create_storage_manager()

        # Update dependent components
        if self._processor:
            self._processor.storage = self._storage
        if self._query_service:
            self._query_service.storage = self._storage

    # TODO: Enable initializing all resources (embedding_service, chat_service, vdb, gdb, etc.)
    # outside of the HiRAG class for better management of resources
    async def _initialize(self, **kwargs) -> None:
        """Initialize all components"""
        await self._create_storage_manager(kwargs.get("vdb"), kwargs.get("gdb"))

        # Initialize other components
        chunker = FixTokenChunk(
            chunk_size=get_hi_rag_config().chunk_size,
            chunk_overlap=get_hi_rag_config().chunk_overlap,
        )

        self._kg_constructor = VanillaKG.create(
            extract_func=get_chat_service().complete,
            llm_model_name=get_llm_config().model_name,
        )

        # Initialize resume tracker
        resume_tracker = kwargs.get("resume_tracker")
        if resume_tracker is None:
            resume_tracker = ResumeTracker()
            logger.info("Using Redis-based resume tracker")

        # Initialize components
        self._metrics = MetricsCollector()
        self._processor = DocumentProcessor(
            storage=self._storage,
            chunker=chunker,
            kg_constructor=self._kg_constructor,
            resume_tracker=resume_tracker,
            metrics=self._metrics,
        )
        self._query_service = QueryService(self._storage)

    # ========================================================================
    # Chat service methods
    # ========================================================================

    # Helper function for similarity calcuation
    async def calculate_similarity(
        self, sentence_embedding: List[float], references: Dict[str, List[float]]
    ) -> List[Dict[str, float]]:
        """Calculate similarity between sentence embedding and reference embeddings"""
        from sklearn.metrics.pairwise import (
            cosine_similarity as sklearn_cosine_similarity,
        )

        similar_refs = []
        for entity_key, embedding in references.items():
            if embedding is not None:
                similarity = sklearn_cosine_similarity(
                    [sentence_embedding], [embedding]
                )[0][0]
                similar_refs.append(
                    {"documentKey": entity_key, "similarity": similarity}
                )
        return similar_refs

    async def chat_complete(self, prompt: str, **kwargs: Any) -> str:
        """Chat with the user"""
        try:
            response = await get_chat_service().complete(
                prompt=prompt,
                **kwargs,
            )
            return response
        except Exception as e:
            log_error_info(
                logging.ERROR,
                "Chat completion failed",
                e,
                raise_error=True,
                new_error_class=HiRAGException,
            )

    async def extract_references(
        self,
        summary: str,
        chunks: List[Dict[str, Any]],
        workspace_id: str,
        knowledge_base_id: str,
    ) -> List[str]:
        """Extract references from summary"""

        # for each sentence, do a query and find the best matching document key to find the referenced chunk
        reference_chunk_list = []

        placeholder = PROMPTS["REFERENCE_PLACEHOLDER"]
        ref_parser = ReferenceParser()
        ref_sentences = await ref_parser.parse_references(summary, placeholder)

        chunk_ids = [c["documentKey"] for c in chunks]

        # Generate embeddings for each reference sentence
        if not ref_sentences:
            logger.warning("No reference sentences found in summary")
            return []

        # Create mapping between non-empty sentences and their indices
        non_empty_sentences = []
        sentence_index_map = {}
        for i, sentence in enumerate(ref_sentences):
            if sentence.strip():
                sentence_index_map[i] = len(non_empty_sentences)
                non_empty_sentences.append(sentence)

        # Only embed non-empty sentences
        if non_empty_sentences:
            sentence_embeddings = await get_embedding_service().create_embeddings(
                texts=non_empty_sentences
            )
        else:
            sentence_embeddings = []

        chunk_embeddings = await self._query_service.query_chunk_embeddings(
            workspace_id=workspace_id,
            knowledge_base_id=knowledge_base_id,
            chunk_ids=chunk_ids,
        )

        for i, sentence in enumerate(ref_sentences):
            # If the sentence is empty, continue
            if not sentence.strip():
                reference_chunk_list.append("")
                continue

            # Get the corresponding embedding for this non-empty sentence
            embedding_index = sentence_index_map[i]
            sentence_embedding = sentence_embeddings[embedding_index]

            similar_chunks = await self.calculate_similarity(
                sentence_embedding, chunk_embeddings
            )

            # Sort by similarity
            reference_list = similar_chunks
            reference_list.sort(key=lambda x: x["similarity"], reverse=True)

            # If no similar chunks found, append empty string
            if not reference_list:
                reference_chunk_list.append("")
                continue

            most_similar_chunk = reference_list[0]
            reference_chunk_list.append(most_similar_chunk["documentKey"])

        return reference_chunk_list

    async def generate_summary(
        self,
        workspace_id: str,
        knowledge_base_id: str,
        query: str,
        chunks: List[Dict[str, Any]],
    ) -> str:
        """Generate summary from chunks"""

        logger.info("🚀 Starting summary generation")
        start_time = time.perf_counter()

        try:
            prompt = PROMPTS["summary_all_" + get_config_manager().language]

            placeholder = PROMPTS["REFERENCE_PLACEHOLDER"]

            parser = DictParser()

            clean_chunks = [
                {"id": i, "chunk": " ".join((c.get("text", "") or "").split())}
                for i, c in enumerate(chunks, start=1)
            ]
            data = (
                "Chunks\n" + parser.parse_list_of_dicts(clean_chunks, "table") + "\n\n"
            )

            prompt = prompt.format(
                data=data,
                max_report_length="5000",
                reference_placeholder=placeholder,
                user_query=query,
            )

            try:
                summary = await self.chat_complete(
                    prompt=prompt,
                    max_tokens=get_llm_config().max_tokens,
                    timeout=get_llm_config().timeout,
                    model=get_llm_config().model_name,
                )
            except Exception as e:
                log_error_info(
                    logging.ERROR,
                    "Summary generation failed",
                    e,
                    raise_error=True,
                    new_error_class=HiRAGException,
                )

            # Find all sentences that contain the placeholder
            ref_parser = ReferenceParser()

            ref_sentences = await ref_parser.parse_references(summary, placeholder)

            # for each sentence, do a query and find the best matching document key to find the referenced chunk
            result = []

            chunk_ids = [c["documentKey"] for c in chunks]

            # Generate embeddings for each reference sentence
            if not ref_sentences:
                logger.warning("No reference sentences found in summary")
                return summary

            # Create mapping between non-empty sentences and their indices
            non_empty_sentences = []
            sentence_index_map = {}
            for i, sentence in enumerate(ref_sentences):
                if sentence.strip():
                    sentence_index_map[i] = len(non_empty_sentences)
                    non_empty_sentences.append(sentence)

            # Only embed non-empty sentences
            if non_empty_sentences:
                sentence_embeddings = await get_embedding_service().create_embeddings(
                    texts=non_empty_sentences
                )
            else:
                sentence_embeddings = []

            chunk_embeddings = await self._query_service.query_chunk_embeddings(
                workspace_id=workspace_id,
                knowledge_base_id=knowledge_base_id,
                chunk_ids=chunk_ids,
            )

            for i, sentence in enumerate(ref_sentences):
                # If the sentence is empty, continue
                if not sentence.strip():
                    result.append("")
                    continue

                # Get the corresponding embedding for this non-empty sentence
                embedding_index = sentence_index_map[i]
                sentence_embedding = sentence_embeddings[embedding_index]

                similar_chunks = await self.calculate_similarity(
                    sentence_embedding, chunk_embeddings
                )

                # Sort by similarity
                reference_list = similar_chunks
                reference_list.sort(key=lambda x: x["similarity"], reverse=True)

                # If no similar chunks found, append empty string
                if not reference_list:
                    result.append("")
                    continue

                reference_threshold = get_hi_rag_config().similarity_threshold
                max_similarity_difference = (
                    get_hi_rag_config().similarity_max_difference
                )

                # If we have a most similar reference, only accept others with similarity having this difference or less
                most_similar = reference_list[0]
                if most_similar["similarity"] > reference_threshold:
                    reference_threshold = max(
                        most_similar["similarity"] - max_similarity_difference,
                        reference_threshold,
                    )

                # Filter references based on similarity threshold
                filtered_references = [
                    ref
                    for ref in reference_list
                    if ref["similarity"] >= reference_threshold
                ]

                # Limit the number of references to max references in HiRAGConfig
                filtered_references = filtered_references[
                    : get_hi_rag_config().max_references
                ]

                # If no references found, append empty string
                if not filtered_references:
                    result.append([])
                    continue

                # Separate the references by "," and sort by type as primary, similarity as secondary
                filtered_references.sort(
                    key=lambda x: (x["documentKey"].split("_")[0], -x["similarity"])
                )

                if len(filtered_references) == 1:
                    result.append([filtered_references[0]["documentKey"]])
                else:
                    # Join the document keys with ", "
                    result.append([ref["documentKey"] for ref in filtered_references])

            format_prompt = PROMPTS["REFERENCE_FORMAT"]

            # fill the summary by ref chunks
            summary = await ref_parser.fill_placeholders(
                text=summary,
                references=result,
                reference_placeholder=placeholder,
                format_prompt=format_prompt,
            )

            total_time = time.perf_counter() - start_time
            logger.info(f"✅ Summary generation completed in {total_time:.3f}s")

            return summary

        except Exception as e:
            total_time = time.perf_counter() - start_time
            log_error_info(
                logging.ERROR,
                f"❌ Summary generation failed after {total_time:.3f}s",
                e,
                raise_error=True,
            )

    # ========================================================================
    # Public interface methods
    # ========================================================================

    async def insert_to_kb(
        self,
        document_path: str,
        workspace_id: str,
        knowledge_base_id: str,
        content_type: str,
        with_graph: bool = True,
        document_meta: Optional[Dict] = None,
        loader_configs: Optional[Dict] = None,
        job_id: Optional[str] = None,
        overwrite: Optional[bool] = False,
        loader_type: Optional[LoaderType] = None,
    ) -> ProcessingMetrics:
        """
        Insert document into knowledge base

        Args:
            document_path: document path
            workspace_id: workspace id
            knowledge_base_id: knowledge base id
            content_type: document type
            with_graph: whether to process graph data (entities and relations)
            document_meta: document metadata
            loader_configs: loader configurations
            job_id: job id
            overwrite: whether to overwrite the document
            loader_type: loader type (optional, will route to appropriate loader based on content type)
        Returns:
            ProcessingMetrics: processing metrics
        """
        if not self._processor:
            raise HiRAGException("HiRAG instance not properly initialized")
        if not workspace_id:
            raise HiRAGException("Workspace ID (workspace_id) is required")
        if not knowledge_base_id:
            raise HiRAGException("Knowledge base ID (knowledge_base_id) is required")

        logger.info(f"🚀 Starting document processing: {document_path}")
        start_time = time.perf_counter()
        document_uri = (
            document_meta.get("uri", document_path) if document_meta else document_path
        )
        document_id = compute_mdhash_id(
            f"{document_uri}:{knowledge_base_id}:{workspace_id}", prefix="doc-"
        )

        document_meta["documentKey"] = document_id
        document_meta["knowledgeBaseId"] = knowledge_base_id
        document_meta["workspaceId"] = workspace_id
        document_meta["uploadedAt"] = datetime.now()

        if job_id and self._processor and self._processor.resume_tracker is not None:
            try:
                await self._processor.resume_tracker.set_job_status(
                    job_id=job_id,
                    status=JobStatus.PROCESSING,
                    document_uri=(
                        document_meta.get("uri")
                        if isinstance(document_meta, dict)
                        else str(document_path)
                    ),
                    with_graph=with_graph,
                )

            except Exception as e:
                log_error_info(
                    logging.WARNING, f"Failed to initialize external job {job_id}", e
                )

        if await self._processor.resume_tracker.is_document_already_completed(
            document_id, workspace_id, knowledge_base_id
        ):
            if overwrite:
                logger.info(
                    "⚠️ Document already processed in previous session, clearing and overwritting..."
                )
                try:
                    await self._processor.resume_tracker.reset_document(
                        document_id, workspace_id, knowledge_base_id
                    )
                    await self._processor.clear_document(
                        document_id, workspace_id, knowledge_base_id
                    )
                except Exception as e:
                    log_error_info(
                        logging.WARNING, f"Failed to reset document {document_id}", e
                    )
            else:
                logger.info("🎉 Document already fully processed in previous session!")
                if job_id:
                    try:
                        await self._processor.resume_tracker.set_job_completed(job_id)
                    except Exception as e:
                        log_error_info(
                            logging.ERROR,
                            "Failed to saving job status (completed) to Postgres",
                            e,
                        )
                total_time = time.perf_counter() - start_time
                metrics = ProcessingMetrics(
                    total_chunks=0,
                    total_entities=0,
                    total_relations=0,
                    processing_time=total_time,
                    job_id=job_id,
                )
                return metrics

        try:
            metrics = await self._processor.process_document(
                document_path=document_path,
                content_type=content_type,
                with_graph=with_graph,
                document_meta=document_meta,
                loader_configs=loader_configs,
                job_id=job_id,
                workspace_id=workspace_id,
                knowledge_base_id=knowledge_base_id,
                loader_type=loader_type,
            )

            total_time = time.perf_counter() - start_time
            metrics.processing_time = total_time
            logger.info(f"🏁 Total pipeline time: {total_time:.3f}s")

            if job_id and not metrics.job_id:
                metrics.job_id = job_id
            return metrics

        except Exception as e:
            total_time = time.perf_counter() - start_time
            log_error_info(
                logging.ERROR,
                f"❌ Document processing failed after {total_time:.3f}s",
                e,
            )
            if (
                self._processor
                and self._processor.resume_tracker is not None
                and job_id
            ):
                try:
                    await self._processor.resume_tracker.set_job_failed(job_id, str(e))
                except Exception as e:
                    log_error_info(
                        logging.ERROR,
                        "Failed to saving job status (failed) to Postgres",
                        e,
                    )
            raise

    async def query_chunks(self, *args, **kwargs) -> List[Dict[str, Any]]:
        """Query document chunks"""
        if not self._query_service:
            raise HiRAGException("HiRAG instance not properly initialized")

        return await self._query_service.query_chunks(*args, **kwargs)

    async def query(
        self,
        query: str,
        workspace_id: str,
        knowledge_base_id: str,
        summary: bool = False,
        threshold: float = 0.0,
        translation: Optional[List[str]] = None,
        translator: Literal["google", "qwen"] = "qwen",
        strategy: Literal["pagerank", "reranker", "hybrid"] = "hybrid",
    ) -> Dict[str, Any]:
        """Query all types of data"""
        if not self._query_service:
            raise HiRAGException("HiRAG instance not properly initialized")
        if not workspace_id:
            raise HiRAGException("Workspace ID (workspace_id) is required")
        if not knowledge_base_id:
            raise HiRAGException("Knowledge base ID (knowledge_base_id) is required")

        original_query = query
        query_list = [original_query]

        if translation:
            # Get translator from resource manager
            if translator == "qwen":
                translator = get_qwen_translator()
            elif translator == "google":
                translator = get_translator()

            if not translator:
                raise HiRAGException("Translator service not properly initialized")

            # Translate to each specified language
            for target_language in translation:
                try:
                    # Following the same pattern as cross_language_search
                    translated_result = await translator.translate(
                        original_query, dest=target_language
                    )
                    if (
                        translated_result.text
                        and translated_result.text != original_query
                    ):
                        query_list.append(translated_result.text)
                        logger.info(
                            f"🌐 Translated query to {target_language}: {translated_result.text}"
                        )
                except Exception as e:
                    logger.warning(f"⚠️ Failed to translate to {target_language}: {e}")

        query_results = await self._query_service.query(
            query=query_list if len(query_list) > 1 else original_query,
            workspace_id=workspace_id,
            knowledge_base_id=knowledge_base_id,
            strategy=strategy,
        )

        # Filter chunks by threshold on relevance score
        if threshold > 0.0 and query_results.get("chunks"):
            # If relevance_score missing for any chunk, show warning
            filtered_chunks = []
            warning_logged = False
            for chunk in query_results["chunks"]:
                if "relevance_score" not in chunk:
                    if not warning_logged:
                        logger.warning(
                            "⚠️ Some chunks missing relevance_score, cannot apply threshold filtering accurately"
                        )
                        warning_logged = True
                if chunk.get("relevance_score", 1.0) >= threshold:
                    filtered_chunks.append(chunk)

            query_results["chunks"] = filtered_chunks

        if summary:
            text_summary = await self.generate_summary(
                workspace_id=workspace_id,
                knowledge_base_id=knowledge_base_id,
                query=original_query,
                chunks=query_results["chunks"],
            )
            query_results["summary"] = text_summary

        return query_results

    async def get_health_status(self) -> Dict[str, Any]:
        """Get system health status"""
        if not self._storage:
            return {"status": "not_initialized"}

        health = await self._storage.health_check()

        return {
            "status": "healthy" if all(health.values()) else "unhealthy",
            "components": health,
            "metrics": self._metrics.metrics.to_dict() if self._metrics else {},
        }

    async def get_processing_metrics(self) -> Dict[str, Any]:
        """Get processing metrics"""
        if not self._metrics:
            return {}

        return {
            "metrics": self._metrics.metrics.to_dict(),
            "operation_times": self._metrics.operation_times,
        }

    async def clean_up(self) -> None:
        """Clean up resources"""
        logger.info("🧹 Cleaning up HiRAG resources...")

        try:
            if self._storage:
                await self._storage.cleanup()

            logger.info("✅ Cleanup completed")

        except Exception as e:
            log_error_info(logging.WARNING, f"⚠ Cleanup failed", e)

    # ========================================================================
    # Context manager support
    # ========================================================================

    async def __aenter__(self):
        """Async context manager entry"""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.clean_up()

    # ========================================================================
    # Backward compatibility property accessors
    # ========================================================================

    @property
    def chunks_table(self):
        """Backward compatibility: access chunks table"""
        return self._storage.chunks_table if self._storage else None

    @property
    def vdb(self):
        """Backward compatibility: access vector database"""
        return self._storage.vdb if self._storage else None

    @property
    def gdb(self):
        """Backward compatibility: access graph database"""
        return self._storage.gdb if self._storage else None

    # ========================================================================
    # DPR-like recall API
    # ========================================================================

    # TODO: whether to use this?
    async def dpr_recall_chunks(
        self,
        query: str,
        workspace_id: str,
        knowledge_base_id: str,
        topk: Optional[int] = None,
        pool_size: int = 500,
    ) -> Dict[str, Any]:
        """Dense Passage Retrieval-style recall using current embeddings and stored vectors.

        Steps:
        - Retrieve a candidate pool without rerank
        - Fetch embeddings of candidates and the query
        - Compute cosine similarities, min-max normalize
        - Return top-k chunk rows with scores and ids
        """
        # Step 1: candidate pool (no rerank)
        candidates = await self._query_service.query_chunks(
            query=query,
            topk=pool_size,
            topn=None,
            workspace_id=workspace_id,
            knowledge_base_id=knowledge_base_id,
        )
        candidate_ids = [
            c.get("documentKey") for c in candidates if c.get("documentKey")
        ]
        if not candidate_ids:
            return {"chunk_ids": [], "scores": [], "chunks": []}

        # Step 2: fetch candidate embeddings and query embedding
        chunk_vec_map = await self._query_service.query_chunk_embeddings(
            workspace_id=workspace_id,
            knowledge_base_id=knowledge_base_id,
            chunk_ids=candidate_ids,
        )
        # Filter out None vectors while preserving id order
        filtered_ids = [
            cid for cid in candidate_ids if chunk_vec_map.get(cid) is not None
        ]
        if not filtered_ids:
            return {"chunk_ids": [], "scores": [], "chunks": []}

        chunk_matrix = np.array(
            [chunk_vec_map[cid] for cid in filtered_ids], dtype=np.float32
        )
        query_vec = await get_embedding_service().create_embeddings([query])
        # embedding services return numpy array (n, d); take first row
        if hasattr(query_vec, "shape"):
            query_vec = np.array(query_vec[0], dtype=np.float32)
        else:
            # fallback for list-like
            query_vec = np.array(query_vec[0], dtype=np.float32)

        # Step 3: cosine similarity
        # Normalize rows of chunk_matrix and query vector
        def _l2_normalize(mat: np.ndarray, axis: int) -> np.ndarray:
            denom = np.linalg.norm(mat, ord=2, axis=axis, keepdims=True)
            denom[denom == 0] = 1.0
            return mat / denom

        chunk_matrix_norm = _l2_normalize(chunk_matrix, axis=1)
        query_vec_norm = query_vec / (np.linalg.norm(query_vec) + 1e-12)
        scores = chunk_matrix_norm @ query_vec_norm

        # Min-max normalize
        s_min = float(scores.min())
        s_max = float(scores.max())
        if s_max > s_min:
            norm_scores = (scores - s_min) / (s_max - s_min)
        else:
            norm_scores = np.zeros_like(scores)

        # Step 4: sort and select top-k
        order = np.argsort(-norm_scores)[
            : max(0, topk or get_hi_rag_config().default_query_top_k)
        ]
        top_ids = [filtered_ids[i] for i in order]
        top_scores = [float(norm_scores[i]) for i in order]
        top_rows = await self._query_service.get_chunks_by_ids(
            top_ids,
            workspace_id=workspace_id,
            knowledge_base_id=knowledge_base_id,
        )
        # Attach score for convenience
        by_id = {row.get("documentKey"): row for row in top_rows}
        result_rows = []
        for cid, sc in zip(top_ids, top_scores):
            row = by_id.get(cid, {"documentKey": cid})
            row = dict(row)
            row["dpr_score"] = sc
            result_rows.append(row)

        return {"chunk_ids": top_ids, "scores": top_scores, "chunks": result_rows}
