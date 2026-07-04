# 3DGS-Agent

面向 3D Gaussian Splatting 论文调研的自动化 Agent 系统。

3DGS-Agent 支持从 arXiv 与本地 Zotero 文献库获取论文元数据和 PDF 附件，将论文检索、PDF 解析、结构化信息抽取、向量检索问答和多论文对比串联成完整流程，用于沉淀 3DGS 方法库，辅助论文阅读、实验复现选型和综述整理。

**Repository:** [github.com/Yang-SyZng/3DGS-agent](https://github.com/Yang-SyZng/3DGS-agent)

**技术栈:** LangChain | LangGraph | Multi Agent | Agentic RAG | Gradio | Milvus | FastAPI | Uvicorn | PyMuPDF | Pydantic | Zotero | SQLite

## 主要能力

- 论文检索与入库：封装 arXiv 检索、Zotero 数据库读取、论文下载、元数据解析和 PDF 分块清洗工具，将论文内容向量化后写入 Milvus。
- 结构化信息抽取：面向 3DGS 领域整理任务类型、输入数据、COLMAP 依赖、数据集、PSNR/SSIM/LPIPS、代码链接和方法局限性等关键信息。
- Agent 流程编排：基于 LangGraph 编排“检索 - 解析 - 抽取 - 入库 - 问答 - 对比”流程，使 Agent 能够按用户问题选择工具调用路径，并处理检索失败、PDF 解析异常等情况。
- Agentic RAG 问答：支持追问方法创新、实验设置、baseline 差异和适用场景，并生成对比表格与阅读笔记。
- 系统指标统计：设计处理规模、解析成功率、字段抽取准确率和检索命中率等统计指标，为后续优化系统稳定性和问答质量提供量化依据。

## 可以这样提问

- 检索最近的 3DGS 动态场景重建论文，并总结它们的主要创新。
- 对比几篇 3DGS 加速渲染方法的实验设置、baseline 和指标表现。
- 从 Zotero 文献库中查找依赖 COLMAP 的方法，并整理数据集和代码链接。
- 根据论文 PDF 提取任务类型、输入数据、评价指标和方法局限性。
