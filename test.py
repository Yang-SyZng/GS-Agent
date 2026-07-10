import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import asyncio
import logging
import re

from rag.parser.mineru_parser import mineru_parser
from rag.parser.splitter import splitter
from rag.parser.naive_process import load_markdown, split_section, split_title, level_resolver, build_tree, parse_nodes, classify_section_tree
from rag.chunks_builder import flatten_tree, splitter_chunks
from rag.embedding import embedding

from rag.vector import milvusvector

import json
logging.basicConfig(
    filename="y.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


async def main():
    # mineru_parser.parse_pdf(
    #     "test/pdf_samples/FSGS_OCR.pdf",
    #     backend="hybrid-engine",
    #     extra_args=["-m", "txt", "-high"],
    # )
    # md_doc = load_markdown("database/parser/AbsGS/AbsGS.md")
    # front_node, result = parse_nodes(md_doc)
    # roots = build_tree(result, "AbsGS")
    # roots = classify_section_tree(roots)
    # flatten_roots = flatten_tree(roots)
    # if front_node is not None:
    #     flatten_roots.append(front_node)
    # text_nodes = splitter_chunks(flatten_roots)
    # embed_nodes = embedding.embed_nodes(text_nodes)
    # with open(
    #     "database/parser/AbsGS/data.json",
    #     "r",
    #     encoding="utf-8"
    # ) as f:
    #     embed_nodes = json.load(f)
    # ids = milvusvector.add_documents(embed_nodes)
    results = milvusvector.search("method")
    print(results)

if __name__ == "__main__":
    asyncio.run(main())
