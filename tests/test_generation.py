from langchain_core.documents import Document

from utils.generation import (
    ThinkStreamParser, build_prompt, estimate_num_ctx, source_label, source_record, format_context,
    NUM_CTX_MIN, NUM_CTX_MAX,
)


def _stream(text, size):
    p = ThinkStreamParser()
    for i in range(0, len(text), size):
        p.feed(text[i:i + size])
    return p.finish()


def test_think_parser_handles_tags_split_across_tokens():
    text = "<think>step 1\nstep 2</think>The answer is 42."
    for size in (1, 2, 3, 5, 100):
        assert _stream(text, size) == ("step 1\nstep 2", "The answer is 42.")


def test_think_parser_without_tags():
    assert _stream("Just an answer with <b>html</b>.", 3) == ("", "Just an answer with <b>html</b>.")


def test_think_parser_unclosed_think_goes_to_thinking():
    assert _stream("<think>never closed", 4) == ("never closed", "")


def test_source_label_and_record():
    doc = Document(page_content="ctx\n\nbody", metadata={"source": "a.pdf", "page": 3, "original_text": "body"})
    assert source_label(doc.metadata) == "a.pdf · p.3"
    assert source_label({"source": "notes.txt"}) == "notes.txt"
    rec = source_record(doc)
    assert rec["text"] == "body" and rec["label"] == "a.pdf · p.3"
    assert "[Source 1] (a.pdf · p.3)" in format_context([doc])


def test_build_prompt_omits_empty_history():
    p = build_prompt("Q?")
    assert "Chat History" not in p and "Question: Q?" in p
    assert "<think>" in build_prompt("Q?", show_thinking=True)
    assert "cover this" in build_prompt("Q?", context="c", crag_low=True)


def test_estimate_num_ctx_grows_with_prompt_and_is_capped():
    assert estimate_num_ctx("x" * 100) == NUM_CTX_MIN
    assert estimate_num_ctx("x" * 30000) > NUM_CTX_MIN
    assert estimate_num_ctx("x" * 10_000_000) == NUM_CTX_MAX
