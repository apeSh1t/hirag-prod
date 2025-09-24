import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sqlalchemy import desc, func

from hirag_prod.cross_language_search.functions import (
    build_search_result,
    classify_search,
    create_embeddings_batch,
    embedding_search_by_search_sentence_list,
    get_synonyms_and_validate_and_translate,
    has_traditional_chinese,
    normalize_text,
    precise_search_by_search_sentence_list,
    search_by_search_keyword_list,
    validate_similarity,
)
from hirag_prod.resources.functions import (
    get_chinese_convertor,
)
from hirag_prod.schema import Item
from hirag_prod.storage.vdb_utils import get_item_info_by_scope


async def cross_language_search(
    knowledge_base_id: str, workspace_id: str, search_content: str
) -> List[Dict[str, Any]]:
    (
        synonym_list,
        synonym_embedding_np_array,
        is_english,
        translation_list,
        translation_embedding_np_array,
    ) = await get_synonyms_and_validate_and_translate(search_content)
    if is_english:
        search_list_original_language: List[str] = []
        search_list: List[str] = synonym_list
    else:
        search_list_original_language: List[str] = synonym_list
        search_list: List[str] = translation_list

    (
        search_keyword_list_original,
        keyword_embedding_np_array_original,
        search_sentence_list_original,
        sentence_embedding_np_array_original,
    ) = classify_search(search_list_original_language, synonym_embedding_np_array)
    (
        search_keyword_list,
        keyword_embedding_np_array,
        search_sentence_list,
        sentence_embedding_np_array,
    ) = classify_search(
        search_list,
        synonym_embedding_np_array if is_english else translation_embedding_np_array,
    )

    str_embedding_np_array_dict: Dict[str, np.ndarray] = {
        "search_keyword": np.concatenate(
            [keyword_embedding_np_array_original, keyword_embedding_np_array]
        ),
        "search_sentence": np.concatenate(
            [sentence_embedding_np_array_original, sentence_embedding_np_array], axis=0
        ),
    }

    chunk_list = await get_item_info_by_scope(
        knowledge_base_id=knowledge_base_id,
        workspace_id=workspace_id,
        columns_to_select=[
            "documentKey",
            "text",
            "fileName",
            "uri",
            "pageNumber",
            "chunkType",
            "pageWidth",
            "pageHeight",
            "bbox",
            "token_list",
            "token_start_index_list",
            "token_end_index_list",
            "translation",
            "translation_token_list",
            "translation_token_start_index_list",
            "translation_token_end_index_list",
        ],
        additional_data_to_select=(
            {
                "search_sentence_cosine_distance": func.least(
                    *[
                        Item.vector.cosine_distance(sentence_embedding)
                        for sentence_embedding in str_embedding_np_array_dict[
                            "search_sentence"
                        ]
                    ]
                )
            }
            if len(str_embedding_np_array_dict["search_sentence"]) > 0
            else None
        ),
        order_by=[Item.fileName, Item.pageNumber, desc(Item.bbox[2]), Item.bbox[1]],
    )
    if len(chunk_list) == 0:
        return []

    processed_chunk_list: List[Dict[str, Any]] = [
        {
            "original_normalized": normalize_text(chunk["text"]),
            "translation_normalized": normalize_text(chunk["translation"]),
            "original_token_list": chunk["token_list"],
            "translation_token_list": chunk["translation_token_list"],
            "original_token_start_index_list": chunk["token_start_index_list"],
            "original_token_end_index_list": chunk["token_end_index_list"],
            "translation_token_start_index_list": chunk[
                "translation_token_start_index_list"
            ],
            "translation_token_end_index_list": chunk[
                "translation_token_end_index_list"
            ],
            "has_traditional_chinese": has_traditional_chinese(chunk["text"]),
            "search_sentence_cosine_distance": chunk.get(
                "search_sentence_cosine_distance", None
            ),
        }
        for chunk in chunk_list
    ]

    str_list_dict_to_embed: Dict[str, List[str]] = {}

    matched_keyword_list_to_embed, matched_keyword_index_list_dict_batch = (
        await search_by_search_keyword_list(
            processed_chunk_list, search_keyword_list_original, search_keyword_list
        )
    )
    if len(matched_keyword_list_to_embed) > 0:
        str_list_dict_to_embed["matched_keyword"] = matched_keyword_list_to_embed

    matched_sentence_list_to_embed, matched_sentence_index_list_dict_batch = (
        await precise_search_by_search_sentence_list(
            processed_chunk_list, search_sentence_list_original, search_sentence_list
        )
    )
    if len(matched_sentence_list_to_embed) > 0:
        str_list_dict_to_embed["matched_sentence"] = matched_sentence_list_to_embed

    str_embedding_np_array_dict.update(
        await create_embeddings_batch(str_list_dict_to_embed)
    )

    if ("matched_keyword" in str_list_dict_to_embed) and (
        len(str_embedding_np_array_dict["search_keyword"]) > 0
    ):
        await validate_similarity(
            str_embedding_np_array_dict["matched_keyword"],
            str_embedding_np_array_dict["search_keyword"],
            matched_keyword_index_list_dict_batch,
        )

    if ("matched_sentence" in str_list_dict_to_embed) and (
        len(str_embedding_np_array_dict["search_sentence"]) > 0
    ):
        await validate_similarity(
            str_embedding_np_array_dict["matched_sentence"],
            str_embedding_np_array_dict["search_sentence"],
            matched_sentence_index_list_dict_batch,
        )

    embedding_similar_chunk_info_dict: Dict[int, float] = {}
    if len(str_embedding_np_array_dict["search_sentence"]) > 0:
        embedding_similar_chunk_info_dict = (
            await embedding_search_by_search_sentence_list(processed_chunk_list)
        )

    build_search_result(
        processed_chunk_list,
        matched_keyword_index_list_dict_batch,
        matched_sentence_index_list_dict_batch,
        embedding_similar_chunk_info_dict,
    )

    matched_blocks: List[Dict[str, Any]] = []
    similar_block_tuple_list: List[Tuple[Dict[str, Any], float]] = []
    for chunk, processed_chunk in zip(chunk_list, processed_chunk_list):
        result_tuple: Optional[Tuple[str, float]] = processed_chunk[
            "original_search_result"
        ]
        is_embedding_result: bool = False
        if result_tuple is None:
            result_tuple = processed_chunk["translation_search_result"]
        if (
            (result_tuple is None)
            and ("embedding_search_result" in processed_chunk)
            and (chunk["chunkType"] in ["text", "list", "table"])
            and (len(processed_chunk["original_token_list"]) > 1)
            and (len(processed_chunk["original_normalized"]) > 6)
            and (
                not re.sub(
                    r"\s", "", processed_chunk["original_normalized"]
                ).isnumeric()
            )
        ):
            result_tuple = processed_chunk["embedding_search_result"]
            is_embedding_result = True
        if result_tuple is not None:
            block = {
                "markdown": (
                    result_tuple[0]
                    if not processed_chunk["has_traditional_chinese"]
                    else get_chinese_convertor("s2hk").convert(result_tuple[0])
                ),
                "chunk": chunk,
            }
            if not is_embedding_result:
                matched_blocks.append(block)
            else:
                similar_block_tuple_list.append((block, result_tuple[1]))
    similar_block_tuple_list.sort(key=lambda x: x[1])

    return matched_blocks + [block_tuple[0] for block_tuple in similar_block_tuple_list]
