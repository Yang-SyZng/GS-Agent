from pathlib import Path


def test_documented_commands_point_to_existing_modules():
    required = [
        Path("ragas_generator.py"),
        Path("tools/milvus_stats.py"),
        Path("tools/embedding_finetune/dataset.py"),
        Path("tools/embedding_finetune/train.py"),
        Path("tools/embedding_finetune/evaluate.py"),
    ]

    assert all(path.is_file() for path in required)


def test_runtime_and_test_dependencies_are_declared():
    runtime = Path("requirement.txt").read_text(encoding="utf-8").lower()
    development = Path("requirement-dev.txt").read_text(encoding="utf-8").lower()

    assert all(name in runtime for name in ("ollama", "ragas", "peft"))
    assert all(name in development for name in ("pytest", "pytest-asyncio"))
