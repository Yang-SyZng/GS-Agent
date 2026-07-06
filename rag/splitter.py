from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import setting
import logging


logger = logging.getLogger(__name__)

class PDFSplitter:
    def __init__(self,
        chunk_size: int = None,
        chunk_overlap: int = None,
        separator: list[str] = None
    ):
        self.chunk_size = chunk_size or setting.chunk_size
        self.chunk_overlap = chunk_overlap or setting.chunk_overlap

        self.separators = separator or [
            # "\n\n\n",  # 章节分隔
            "\n\n",  # 段落分隔
            "\n（",  # 行分隔
            "。", # Chinese
            "\n",
            "，", # Chinese
            "；", # Chinese
            # ". ",  # 句子分隔
            # ", ",  # 子句分隔
            # " ",  # 单词分隔
            # "",  # 字符分隔
        ]

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.separators,
            length_function=len,
            is_separator_regex=False
        )

        logger.info(
            f"初始化文本切分器: chunk_size={self.chunk_size}, "
            f"chunk_overlap={self.chunk_overlap}"
        )
    
    def SplitText(self, text: str) -> list[str]:
        logger.debug(f"分割文本，长度: {len(text)}")
        chunks = self.splitter.split_text(text)
        logger.info(f"文本分割完成，生成{len(chunks)}个分块")
        return chunks

    # def __call__(self, text: str) -> list[str]:
    #     return self.SplitText(text)
    
splitter = PDFSplitter()
