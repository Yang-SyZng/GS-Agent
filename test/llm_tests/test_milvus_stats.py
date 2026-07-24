from tools.milvus_stats import collect_index_stats


class FakeMilvus:
    def get_collection_stats(self, collection_name):
        return {"row_count": 9001}

    def query(self, collection_name, filter, output_fields, limit):
        return [
            {"paper_id": "paper-a"},
            {"paper_id": "paper-a"},
            {"paper_id": "paper-b"},
        ]


def test_index_stats_report_chunks_and_unique_papers(tmp_path):
    (tmp_path / "paper-a").mkdir()
    (tmp_path / "paper-a" / "paper-a.md").write_text("a", encoding="utf-8")
    (tmp_path / "paper-b").mkdir()
    (tmp_path / "paper-b" / "paper-b.md").write_text("b", encoding="utf-8")

    report = collect_index_stats(FakeMilvus(), "papers", tmp_path)

    assert report["indexed_chunks"] == 9001
    assert report["indexed_papers"] == 2
    assert report["parsed_papers"] == 2
