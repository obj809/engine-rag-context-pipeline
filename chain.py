"""Build the LCEL question-answering chain: retriever → prompt → LLM → text."""

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import Runnable, RunnablePassthrough

SYSTEM_PROMPT = (
    "You are a retrieval assistant for the Environment Protection and Biodiversity "
    "Conservation Act 1999 (Cth) ('the EPBC Act'). You answer questions about that "
    "Act using ONLY the provided context.\n"
    "\n"
    "When the context contains the answer: answer concisely, cite specific phrases "
    "where possible, and reference the volume and page (shown as [Volume N, p.M]) for "
    "the facts you use.\n"
    "\n"
    "When the provided context does NOT contain information that answers the question "
    "— including questions that are unrelated to the EPBC Act, general-knowledge "
    "questions, or chit-chat — do NOT guess, do NOT use outside knowledge, and do NOT "
    "invent citations. Instead reply with exactly this message and nothing else:\n"
    "\n"
    "\"I can only answer questions about the Environment Protection and Biodiversity "
    "Conservation Act 1999 (Cth) — for example, environmental approvals, controlled "
    "actions, listed threatened species, protected areas, and offences and penalties "
    "under the Act. Your question looks like it's outside that scope, so try "
    "rephrasing it as a question about the Act.\"\n"
    "\n"
    "Do not mention the words 'context' or 'documents' to the user; refer to your "
    "source as the Act."
)

CONDENSE_PROMPT = (
    "Given a conversation and a follow-up question, rephrase the follow-up into a "
    "standalone question that can be understood without the conversation. Resolve "
    "any pronouns or references (e.g. 'it', 'that', 'those') to what they refer to "
    "in the conversation. If the follow-up is already self-contained, return it "
    "unchanged. Output ONLY the rewritten question, with no preamble or quotes."
)


def _citation(doc: Document) -> str:
    page = doc.metadata.get("page", "?")
    volume = doc.metadata.get("volume")
    return f"{volume}, p.{page}" if volume else f"p.{page}"


def format_docs(docs: list[Document]) -> str:
    return "\n\n---\n\n".join(f"[{_citation(d)}]\n{d.page_content}" for d in docs)


def format_history(messages: list[tuple[str, str]], max_messages: int = 6) -> str:
    """Render the most recent (role, content) turns into a plain transcript.

    Bounded to the last `max_messages` so the condensation prompt stays cheap;
    that window covers typical pronoun/ellipsis references without bloating it.
    """
    return "\n".join(f"{role}: {content}" for role, content in messages[-max_messages:])


def build_condense_chain(llm: BaseChatModel) -> Runnable:
    """History + follow-up → a self-contained question string, for retrieval.

    A cheap pre-retrieval LLM pass: {"chat_history": str, "question": str} →
    prompt → LLM → text. Callers with no prior history should skip this and
    retrieve on the raw question. Keeps the retriever and answer chain
    single-question — the rewrite only sharpens what gets embedded.
    """
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", CONDENSE_PROMPT),
            (
                "human",
                "Conversation:\n{chat_history}\n\n"
                "Follow-up question: {question}\n\nStandalone question:",
            ),
        ]
    )
    return prompt | llm | StrOutputParser()


def build_answer_chain(llm: BaseChatModel) -> Runnable:
    """The generation half: {"context": str, "question": str} → prompt → LLM → text.

    Exposed separately so callers that retrieve eagerly (e.g. the backend's
    streaming /chat) can stream just the LLM part without holding a DB connection.
    """
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", "<context>\n{context}\n</context>\n\nQuestion: {question}"),
        ]
    )
    return prompt | llm | StrOutputParser()


def build_chain(retriever: BaseRetriever, llm: BaseChatModel) -> Runnable:
    # Chain input is the query string: it fans out to the retriever (for context)
    # and straight through (as the question), then prompt → LLM → plain text.
    return (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | build_answer_chain(llm)
    )
