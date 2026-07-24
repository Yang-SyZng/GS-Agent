import json

from tools.embedding_finetune.dataset import build_training_examples
from tools.embedding_finetune.evaluate import retrieval_metrics


def test_dataset_builds_positive_and_hard_negatives(tmp_path):
    dataset = tmp_path / "ragas.jsonl"
    rows = [
        {
            "sample_id": "a:1",
            "user_input": "How does A work?",
            "reference_contexts": ["positive A"],
            "source_documents": ["A/A.md"],
        },
        {
            "sample_id": "b:1",
            "user_input": "How does B work?",
            "reference_contexts": ["confusing B"],
            "source_documents": ["B/B.md"],
        },
        {
            "sample_id": "c:1",
            "user_input": "How does C work?",
            "reference_contexts": ["random C"],
            "source_documents": ["C/C.md"],
        },
    ]
    dataset.write_text(
        "\n".join(json.dumps(row) for row in rows),
        encoding="utf-8",
    )
    mined = {"a:1": ["positive A", "retrieved hard negative"]}

    examples = build_training_examples(dataset, mined, negatives_per_query=3, seed=7)

    first = examples[0]
    assert first["positive"] == "positive A"
    assert "positive A" not in first["negatives"]
    assert "retrieved hard negative" in first["negatives"]
    assert len(first["negatives"]) == 3
    assert first["split"] in {"train", "validation", "test"}


def test_retrieval_metrics_use_known_relevant_document_ids():
    rankings = {"q1": ["d2", "d1"], "q2": ["d2", "d3"]}
    positives = {"q1": "d1", "q2": "d2"}

    result = retrieval_metrics(rankings, positives, ks=(1, 2))

    assert result == {"recall@1": 0.5, "recall@2": 1.0, "mrr": 0.75}
