from llama_index.core import Document

from rag.parser.naive_process import parse_nodes


def test_parse_nodes_accepts_title_only_front_matter():
    front_node, section_nodes = parse_nodes(
        Document(text="Paper Title\n\n# 1 Introduction\n\nBody"),
        paper_id="paper",
    )

    assert front_node is not None
    assert front_node.title == "Paper Title"
    assert front_node.content == ""
    assert len(section_nodes) == 1


def test_parse_nodes_preserves_front_matter_body():
    front_node, _ = parse_nodes(
        Document(text="Paper Title\n\nAuthors and affiliations\n\n# 1 Introduction\n\nBody"),
        paper_id="paper",
    )

    assert front_node is not None
    assert front_node.title == "Paper Title"
    assert front_node.content == "Authors and affiliations"
