from pathlib import Path


def test_external_search_stop_route_always_reaches_writer():
    source = Path("workflow/graph.py").read_text(encoding="utf-8")

    external_block = source.split(
        'builder.add_conditional_edges(\n    "external_search"', 1
    )[1]
    assert '"stop": "writer"' in external_block


def test_writer_receives_external_search_evidence():
    source = Path("workflow/nodes.py").read_text(encoding="utf-8")

    assert 'external_search_results=state.get("external_search_results", [])' in source
    assert 'external_search_errors=state.get("external_search_errors", [])' in source
