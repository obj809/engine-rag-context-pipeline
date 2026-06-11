"""Build the LCEL question-answering chain: retriever → prompt → LLM → text."""

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import Runnable, RunnablePassthrough

SYSTEM_PROMPT = (
    "You answer questions about a document using only the provided context. "
    "If the answer is not in the context, say so plainly. Cite specific phrases when "
    "possible, and reference the page number (shown as [page N]) for the facts you use."
)


def format_docs(docs: list[Document]) -> str:
    return "\n\n---\n\n".join(
        f"[page {d.metadata.get('page', '?')}]\n{d.page_content}" for d in docs
    )


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
