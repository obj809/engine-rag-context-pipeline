"""Unit tests for chain.py: format_docs, and the two chain halves wired to a
fake LLM. build_answer_chain's streaming is covered because the backend's
/chat streams through it."""

from itertools import cycle

from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from conftest import FAKE_ANSWER, FAKE_ROWS, RecordingChatModel
from chain import SYSTEM_PROMPT, build_answer_chain, build_chain, format_docs


def _doc(text, page=None):
    return Document(page_content=text, metadata={} if page is None else {"page": page})


def test_format_docs_cites_pages_and_separates_chunks():
    out = format_docs([_doc("first chunk", 96), _doc("second chunk", 4)])
    assert out == "[page 96]\nfirst chunk\n\n---\n\n[page 4]\nsecond chunk"


def test_format_docs_missing_page_becomes_question_mark():
    assert format_docs([_doc("no page")]) == "[page ?]\nno page"


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
    for content, page in FAKE_ROWS:  # retrieved chunks land in the prompt, page-tagged
        assert f"[page {page}]\n{content}" in human.content


def test_answer_chain_streams_text_fragments():
    llm = GenericFakeChatModel(messages=cycle([AIMessage(content=FAKE_ANSWER)]))
    chunks = list(
        build_answer_chain(llm).stream({"context": "[page 96]\nsome context", "question": "q?"})
    )
    assert len(chunks) > 1  # actually incremental, not one blob
    assert "".join(chunks) == FAKE_ANSWER
