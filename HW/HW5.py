import glob
import json
import os
import sys

# ChromaDB needs a newer sqlite3 than Streamlit Cloud ships with.
# Swap in pysqlite3 before chromadb is imported. Harmless if not installed.
try:
    __import__("pysqlite3")
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass

import chromadb
import streamlit as st
from bs4 import BeautifulSoup
from openai import OpenAI

HTML_DIR = "html"
# Deliberately the same collection and path as HW4. The documents and the
# chunking are identical, so reusing it means the pages are embedded once no
# matter which page the user opens first.
COLLECTION_NAME = "HW4Collection"
CHROMA_PATH = "./ChromaDB_for_hw4"
EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4.1-mini"

# Short-term memory: the last 5 interactions, where one interaction is a user
# message plus the assistant's reply.
BUFFER_INTERACTIONS = 5
BUFFER_SIZE = BUFFER_INTERACTIONS * 2

TOP_K = 5
EMBED_BATCH = 100

BASE_SYSTEM = (
    "You are an advisor who helps Syracuse University students find student "
    "organizations.\n\n"
    "You have a tool, relevant_club_info, that searches a database of "
    "Syracuse student organization pages. Call it whenever answering needs "
    "information about specific clubs. Do not call it for greetings, thanks, "
    "or follow-up questions you can already answer from the conversation.\n\n"
    "When you do search, write the query as a short description of what the "
    "student is looking for rather than passing their message through "
    "verbatim. For example, if they say 'I miss home, is there anything for "
    "people from my country', search for something like 'international "
    "student cultural organization'."
)

ANSWER_SYSTEM = (
    "You are an advisor who helps Syracuse University students find student "
    "organizations. Answer using the organization pages below.\n\n"
    "Name the organizations you draw on so the student knows where the "
    "information came from. If the pages do not answer the question, say so "
    "plainly. You can only see a few organizations at a time, so if you are "
    "asked for a complete list of every club matching some criterion, explain "
    "that you can only speak to the ones you found.\n\n"
    "--- ORGANIZATION PAGES ---\n{context}"
)


def get_client():
    return OpenAI(api_key=st.secrets["OPENAI_API_KEY"])


def html_to_text(path):
    """Strip an organization page down to its readable text."""
    with open(path, encoding="utf-8", errors="ignore") as f:
        soup = BeautifulSoup(f.read(), "html.parser")

    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()

    title = soup.title.get_text(strip=True) if soup.title else os.path.basename(path)
    org_name = title.split(" - ")[0].strip()

    lines = [line.strip() for line in soup.get_text("\n").split("\n") if line.strip()]
    text = "\n".join(
        line for line in lines
        if line != "This application requires JavaScript to be enabled."
    )
    return org_name, text


def chunk(text):
    """Split one organization page into two mini-documents.

    Structure-aware split on the page's own "Contact Information" heading, so
    chunk 1 describes the group and chunk 2 holds meeting and contact details.
    Falls back to a midpoint split on a line boundary when the heading is
    absent. Same method as HW4.
    """
    marker = "Contact Information"
    if marker in text:
        index = text.index(marker)
        return text[:index].strip(), text[index:].strip()

    lines = text.split("\n")
    middle = len(lines) // 2
    return "\n".join(lines[:middle]).strip(), "\n".join(lines[middle:]).strip()


def embed_batch(texts):
    response = get_client().embeddings.create(model=EMBED_MODEL, input=texts)
    return [item.embedding for item in response.data]


def build_vector_db():
    """Create the collection from the HTML pages, once."""
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)

    if collection.count() > 0:
        return collection

    ids, documents, metadatas = [], [], []
    for path in sorted(glob.glob(os.path.join(HTML_DIR, "*.html"))):
        slug = os.path.splitext(os.path.basename(path))[0]
        org_name, text = html_to_text(path)
        first, second = chunk(text)

        for part_number, part_text in ((1, first), (2, second)):
            if not part_text:
                continue
            ids.append(f"{slug}::{part_number}")
            documents.append(f"{org_name}\n\n{part_text}")
            metadatas.append({
                "organization": org_name,
                "source": os.path.basename(path),
                "part": part_number,
            })

    progress = st.progress(0.0, text="Embedding organization pages...")
    for start in range(0, len(documents), EMBED_BATCH):
        stop = start + EMBED_BATCH
        collection.add(
            ids=ids[start:stop],
            documents=documents[start:stop],
            embeddings=embed_batch(documents[start:stop]),
            metadatas=metadatas[start:stop],
        )
        progress.progress(min(stop / len(documents), 1.0))
    progress.empty()

    return collection


def relevant_club_info(collection, query, k=TOP_K):
    """The tool body: vector search, returning [(organization, text), ...]."""
    results = collection.query(
        query_embeddings=[embed_batch([query])[0]],
        n_results=k,
    )
    return list(zip(
        [m["organization"] for m in results["metadatas"][0]],
        results["documents"][0],
    ))


TOOLS = [{
    "type": "function",
    "function": {
        "name": "relevant_club_info",
        "description": (
            "Search a database of Syracuse University student organization "
            "pages and return the most relevant ones. Each page covers what "
            "the organization does, when and where it meets, who its officers "
            "are, and how to join."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "What to search for, written as a short description "
                        "of the kind of organization or the specific "
                        "information needed."
                    ),
                },
            },
            "required": ["query"],
        },
    },
}]


st.title("HW 5: Student Organization Chatbot with a Search Tool")

st.write(
    "Ask about Syracuse student organizations. Unlike HW4, which embedded "
    "every message and searched whether or not it made sense to, this version "
    "gives the LLM a search tool and lets it decide. It can skip the search "
    "for a greeting or a follow-up it can already answer, and when it does "
    "search it writes its own query rather than reusing your words. Results "
    "come back through a second call that has the pages in its system prompt "
    f"and no tool access. The conversation keeps the last {BUFFER_INTERACTIONS} "
    "interactions."
)

# ---------- Build or reuse the vector DB ----------
if "HW5_VectorDB" not in st.session_state:
    if not glob.glob(os.path.join(HTML_DIR, "*.html")):
        st.error(f"No HTML files found in the '{HTML_DIR}' folder.")
        st.stop()
    st.session_state.HW5_VectorDB = build_vector_db()

collection = st.session_state.HW5_VectorDB
st.caption(f"{COLLECTION_NAME}: {collection.count()} chunks indexed")

# ---------- Chat ----------
if "hw5_messages" not in st.session_state:
    st.session_state.hw5_messages = [
        {"role": "assistant",
         "content": "What kind of student organization are you looking for?"}
    ]

for message in st.session_state.hw5_messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask about student organizations"):
    st.session_state.hw5_messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    history = st.session_state.hw5_messages[-BUFFER_SIZE:]
    client = get_client()

    # ---- First call: does this need a search, and if so, for what? ----
    decision = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "system", "content": BASE_SYSTEM}] + history,
        tools=TOOLS,
        tool_choice="auto",
    )
    reply = decision.choices[0].message

    with st.chat_message("assistant"):
        if not reply.tool_calls:
            # No search needed, so the first response is the answer.
            answer = reply.content
            st.markdown(answer)
            st.caption("No search needed")
        else:
            matches, queries = [], []
            for call in reply.tool_calls:
                arguments = json.loads(call.function.arguments or "{}")
                query = arguments.get("query", prompt)
                queries.append(query)
                matches.extend(relevant_club_info(collection, query))

            # ---- Second call: answer from the results, with no tools ----
            context = "\n\n".join(f"--- {org} ---\n{text}" for org, text in matches)
            stream = client.chat.completions.create(
                model=CHAT_MODEL,
                messages=[{
                    "role": "system",
                    "content": ANSWER_SYSTEM.format(context=context),
                }] + history,
                stream=True,
            )
            answer = st.write_stream(stream)

            st.caption("Searched for: " + " | ".join(queries))
            st.caption("Retrieved: " + ", ".join(
                dict.fromkeys(org for org, _ in matches)
            ))

    st.session_state.hw5_messages.append({"role": "assistant", "content": answer})