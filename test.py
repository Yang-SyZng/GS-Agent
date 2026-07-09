import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import asyncio
import logging
import re

from rag.parser.mineru_parser import mineru_parser
from rag.parser.splitter import splitter
from rag.parser.naive_process import load_markdown, split_section, split_title, level_resolver, build_tree, parse_nodes, classify_section_tree
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
    md_doc = load_markdown("database/parser/AbsGS/AbsGS.md")
    result = parse_nodes(md_doc)
    roots = build_tree(result, "AbsGS")
    roots = classify_section_tree(roots)
    print(roots)

if __name__ == "__main__":
    asyncio.run(main())
