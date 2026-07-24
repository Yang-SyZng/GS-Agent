# GS RAGAS Dataset

`questions.jsonl` 汇总全部问答，每行一个 JSON 对象。生成命令：

```bash
python -m ragas_generator --resume
```

先试跑一篇论文（约 5 条）：

```bash
python -m ragas_generator --max-documents 1
```

数据包含 `user_input`、`reference_contexts`、`reference` 等 RAGAS 字段，并用
`scenario`、`task_type`、`source_documents` 标注场景、任务与证据来源。
