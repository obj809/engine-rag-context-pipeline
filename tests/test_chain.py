"""Unit tests for chain.py: format_docs, and the two chain halves wired to a
fake LLM. build_answer_chain's streaming is covered because the backend's
/chat streams through it."""

from itertools import cycle

from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from conftest import FAKE_ANSWER, FAKE_ROWS, RecordingChatModel
from chain import SYSTEM_PROMPT, build_answer_chain, build_chain, format_docs


def _doc(text, page=None, volume=None):
    metadata = {}
    if page is not None:
        metadata["page"] = page
    if volume is not None:
        metadata["volume"] = volume
    return Document(page_content=text, metadata=metadata)


def test_format_docs_cites_volume_and_page_and_separates_chunks():
    out = format_docs([_doc("first chunk", 96, "Volume 1"), _doc("second chunk", 4, "Volume 2")])
    assert out == "[Volume 1, p.96]\nfirst chunk\n\n---\n\n[Volume 2, p.4]\nsecond chunk"


def test_format_docs_missing_page_becomes_question_mark():
    assert format_docs([_doc("no page")]) == "[p.?]\nno page"


def test_build_chain_answers_from_query_string(retriever):
    llm = GenericFakeChatModel(messages=cycle([AIMessage(content=FAKE_ANSWER)]))
    answer = build_chain(retriever, llm).invoke("what is the target year?")
    assert answer == FAKE_ANSWER


def test_chain_sends_context_and_question_to_the_llm(retriever):
    llm = RecordingChatModel(calls=[])
    build_chain(retriever, llm).invoke("what is the target year?")

    (system, human), = llm.calls
    assert system.content == SYSTEM_PROMPT
    assert "Question: what is the target year?" in human.content
    for content, page, volume in FAKE_ROWS:  # retrieved chunks land in the prompt, citation-tagged
        assert f"[{volume}, p.{page}]\n{content}" in human.content


def test_answer_chain_streams_text_fragments():
    llm = GenericFakeChatModel(messages=cycle([AIMessage(content=FAKE_ANSWER)]))
    chunks = list(
        build_answer_chain(llm).stream({"context": "[Volume 1, p.96]\nsome context", "question": "q?"})
    )
    assert len(chunks) > 1  # actually incremental, not one blob
    assert "".join(chunks) == FAKE_ANSWER
