import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import torch
from transformers import AutoModel, AutoTokenizer
from config import setting
import logging
import tempfile, fitz

from langchain_core.tools import StructuredTool

logger = logging.getLogger(__name__)

class PDFProcess:
    def __init__(self):
        logger.info("正在初始化PDF Process...")

    @property
    def process(self):
        pass

    def extract_pdf_stats(self,
        file_path: str | Path,
    ) -> dict[str, Any]:
        """
        统计 PDF 内容特征，适用于工具调用。

        功能:
            1. 计算原生文本量
            2. 估算图片区域占比
            3. 统计文本块和图片块数量
            4. 判断是否包含可提取文本

        参数:
            file_path: PDF 文件路径。

        返回:
            dict，包含 file_path、page_count、pages、avg_text_length、
            avg_image_ratio、avg_text_block_count、avg_image_block_count、
            has_extractable_text、page_stats 等统计信息。
        """
        file_path = Path(file_path).expanduser().resolve()

        if not file_path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")

        if file_path.suffix.lower() != ".pdf":
            raise ValueError(f"当前只支持 PDF 文件: {file_path}")

        page_stats = []

        with fitz.open(file_path) as doc:
            page_count = doc.page_count

            for page_index in range(page_count):
                page = doc.load_page(page_index)

                page_width = page.rect.width
                page_height = page.rect.height
                page_area = page_width * page_height

                # 1. 提取原生文本
                text = page.get_text("text") or ""
                clean_text = "".join(text.split())
                text_length = len(clean_text)

                # 2. 分析页面 block，包括 text block / image block
                page_dict = page.get_text("dict")
                blocks = page_dict.get("blocks", [])

                text_block_count = 0
                image_block_count = 0
                image_area = 0.0

                for block in blocks:
                    block_type = block.get("type")

                    # type == 0: 文本块
                    if block_type == 0:
                        text_block_count += 1

                    # type == 1: 图片块
                    elif block_type == 1:
                        image_block_count += 1

                        bbox = block.get("bbox")
                        if bbox:
                            x0, y0, x1, y1 = bbox
                            width = max(0, x1 - x0)
                            height = max(0, y1 - y0)
                            image_area += width * height

                # 防止多个图片块重叠导致面积超过整页
                if page_area > 0:
                    image_area_ratio = min(image_area / page_area, 1.0)
                else:
                    image_area_ratio = 0.0

                page_stats.append(
                    {
                        "page_index": page_index,
                        "page_number": page_index + 1,
                        "text_length": text_length,
                        "image_area_ratio": round(image_area_ratio, 4),
                        "text_block_count": text_block_count,
                        "image_block_count": image_block_count,
                        "page_width": round(page_width, 2),
                        "page_height": round(page_height, 2),
                    }
                )

        pages = len(page_stats)

        if pages == 0:
            return {
                "file_path": str(file_path),
                "page_count": 0,
                "pages": 0,
                "avg_text_length": 0,
                "avg_image_ratio": 0.0,
                "avg_text_block_count": 0.0,
                "avg_image_block_count": 0.0,
                "has_extractable_text": False,
                "page_stats": [],
            }

        avg_text_length = sum(p["text_length"] for p in page_stats) / pages
        avg_image_ratio = sum(p["image_area_ratio"] for p in page_stats) / pages
        avg_text_block_count = sum(p["text_block_count"] for p in page_stats) / pages
        avg_image_block_count = sum(p["image_block_count"] for p in page_stats) / pages

        return {
            "file_path": str(file_path),
            "page_count": page_count,
            "avg_text_length": round(avg_text_length, 2),
            "avg_image_ratio": round(avg_image_ratio, 4),
            "avg_text_block_count": round(avg_text_block_count, 2),
            "avg_image_block_count": round(avg_image_block_count, 2),
            "has_extractable_text": avg_text_length > 30,
            "page_stats": page_stats,
        }
    
    def detect_layout_blocks(self, 
        file_path: str | Path,
    ) -> dict[str, Any]:
        """
        检测 PDF 页面中的版面元素，适用于工具调用。

        功能:
            1. 检测文本块和图片块
            2. 检测表格、标题、图注、页眉、页脚、公式等布局元素
            3. 返回每页块类型与整体布局标签

        参数:
            file_path: PDF 文件路径。

        返回:
            dict，包含 file_path、layout_blocks、has_text、has_image、
            has_table、has_caption、has_equation、has_header_footer、page_results。
        """
        file_path = Path(file_path).expanduser().resolve()

        if not file_path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")

        if file_path.suffix.lower() != ".pdf":
            raise ValueError(f"当前只支持 PDF 文件: {file_path}")

        layout_blocks = []
        page_results = []

        with fitz.open(file_path) as doc:
            page_count = doc.page_count

            for page_index in range(page_count):
                page = doc.load_page(page_index)
                page_height = page.rect.height

                blocks = page.get_text("dict").get("blocks", [])
                page_block_types = []

                table_count = 0
                try:
                    tables = page.find_tables()
                    table_count = len(tables.tables)
                    if table_count > 0:
                        page_block_types.append("table")
                        layout_blocks.append("table")
                except Exception:
                    table_count = 0

                for block in blocks:
                    block_type = block.get("type")
                    bbox = block.get("bbox", [])
                    text = ""

                    # type == 0 是文本块
                    if block_type == 0:
                        for line in block.get("lines", []):
                            for span in line.get("spans", []):
                                text += span.get("text", "")

                        text = text.strip()

                        if not text:
                            continue

                        page_block_types.append("text")
                        layout_blocks.append("text")

                        y0 = bbox[1] if len(bbox) == 4 else 0
                        y1 = bbox[3] if len(bbox) == 4 else 0

                        # 页眉
                        if y1 < page_height * 0.08:
                            page_block_types.append("header")
                            layout_blocks.append("header")

                        # 页脚
                        if y0 > page_height * 0.90:
                            page_block_types.append("footer")
                            layout_blocks.append("footer")

                        # 图注/表注
                        lower_text = text.lower()
                        if (
                            lower_text.startswith("fig.")
                            or lower_text.startswith("figure")
                            or lower_text.startswith("table")
                            or text.startswith("图")
                            or text.startswith("表")
                        ):
                            page_block_types.append("caption")
                            layout_blocks.append("caption")

                        # 标题：用字体大小判断
                        max_font_size = 0
                        for line in block.get("lines", []):
                            for span in line.get("spans", []):
                                max_font_size = max(
                                    max_font_size,
                                    span.get("size", 0)
                                )

                        if page_index == 0 and max_font_size >= 14:
                            page_block_types.append("title")
                            layout_blocks.append("title")

                        # 公式规则判断
                        if any(symbol in text for symbol in ["=", "∑", "∫", "√", "\\", "α", "β", "λ"]):
                            page_block_types.append("equation")
                            layout_blocks.append("equation")

                    # type == 1 是图片块
                    elif block_type == 1:
                        page_block_types.append("image")
                        layout_blocks.append("image")

                page_results.append(
                    {
                        "page_number": page_index + 1,
                        "block_types": sorted(set(page_block_types)),
                        "table_count": table_count,
                    }
                )

        unique_blocks = sorted(set(layout_blocks))

        return {
            "file_path": str(file_path),
            "layout_blocks": unique_blocks,
            "has_text": "text" in unique_blocks,
            "has_image": "image" in unique_blocks,
            "has_table": "table" in unique_blocks,
            "has_caption": "caption" in unique_blocks,
            "has_equation": "equation" in unique_blocks,
            "has_header_footer": "header" in unique_blocks or "footer" in unique_blocks,
            "page_results": page_results,
        }
        

class PDFProcess_OCR:
    """ OCR技术基于最新开源百度：https://github.com/baidu/Unlimited-OCR """

    def __init__(self):
        logger.info("正在初始化OCR...")
        model_name = setting.OCR_MODEL_ID

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            use_safetensors=True,
            torch_dtype=torch.bfloat16,
        )
        self.model = self.model.eval().cuda()

    def infer(self,
            prompt: str = '<image>Multi page parsing.', 
            image_files: str | list[str] = None, 
            output_path: str = '', 
            image_size: int = 1024, 
            save_results: bool = True, 
            max_length: int = 32768, 
            tps_interval: int = 0, 
            no_repeat_ngram_size: int = 35, 
            ngram_window: int = 1024, 
            temperature: int = 0.0,
            dpi: int | None = None
        ) -> str:
        """执行多图像OCR推理。

        支持三种输入形式:
        1. 单个图像路径；
        2. 图像路径列表；
        3. PDF 路径。

        如果输入为 PDF，会先调用 pdf_to_images() 将 PDF 转换为图像列表，再进行推理。
        当 dpi 为 None 时，使用 pdf_to_images() 的默认 DPI 值。

        参数:
            prompt: 包含一个 <image> 占位符的文本提示。
            image_files: 单张图像路径、图像路径列表，或 PDF 路径。
            output_path: 保存结果文件的目录，仅当 save_results=True 时生效。
            image_size: 输入图像缩放尺寸，用于视觉特征提取。
            save_results: 是否保存生成文本和可视化结果。
            max_length: 最大生成令牌长度。
            tps_interval: TPSTextStreamer 的每秒令牌日志间隔。
            no_repeat_ngram_size: 生成时避免重复的 n-gram 大小。
            ngram_window: no-repeat n-gram 处理的滑动窗口大小。
            temperature: 生成温度，0.0 表示贪心解码。
            dpi: 仅当输入为 PDF 时生效，指定 PDF 转换为图像的 DPI。

        返回:
            outputs: 解码后的文本结果。
        """
        if image_files is None:
            raise ValueError("image_files 不能为空，必须提供图像路径、图像路径列表或 PDF 路径。")

        pdf_dpi = dpi if dpi is not None else 300

        output, ouput_token = self.model.infer_multi(
            self.tokenizer,
            prompt=prompt,
            image_files=image_files,
            output_path=output_path,
            image_size=image_size,
            save_results=save_results,
            max_length=max_length,
            tps_interval=tps_interval,
            no_repeat_ngram_size=no_repeat_ngram_size, 
            ngram_window=ngram_window,
            temperature=temperature
        )

        logger.info(f"成功提取所有信息，长度 {len(output)} 个字符，共 {ouput_token} 个token")

        return output

def pdf_to_images(self, pdf_path: str, dpi: int = 300):
    doc = fitz.open(pdf_path)
    tmp_dir = tempfile.mkdtemp(prefix='pdf_ocr_')
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    paths = []
    for i, page in enumerate(doc):
        out = os.path.join(tmp_dir, f'page_{i+1:04d}.png')
        page.get_pixmap(matrix=mat).save(out)
        paths.append(out)
    doc.close()
    return paths

def extract_text(self, pdf_path: str | Path) -> Optional[str]:
    """
    适用于原生文本

    Args:
        pdf_path: PDF文件路径

    Returns:
        提取的文本内容
    """
    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        logger.error(f"PDF文件不存在: {pdf_path}")
        return None

    logger.info(f"提取PDF文本: {pdf_path.name}")

    try:
        doc = fitz.open(pdf_path)
        full_text = ""

        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            full_text += text

        doc.close()

        logger.info(f"成功提取文本，长度: {len(full_text)} 字符")
        return full_text

    except Exception as e:
        logger.error(f"PDF文本提取失败: {str(e)}")
        return None
        
pdf_process = PDFProcess()

PDFProcessTools = [
    StructuredTool.from_function(pdf_process.extract_pdf_stats),
    StructuredTool.from_function(pdf_process.detect_layout_blocks)
]