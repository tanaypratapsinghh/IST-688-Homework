import glob
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
COLLECTION_NAME = "HW4Collection"
CHROMA_PATH = "./ChromaDB_for_hw4"
EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4.1-mini"

# Conversation memory: the last 5 interactions, where one interaction is a
# user message plus the assistant's reply.
BUFFER_INTERACTIONS = 5
BUFFER_SIZE = BUFFER_INTERACTIONS * 2

# Chunks pulled from the vector DB per question.
TOP_K = 5

# Embeddings are requested in batches; the API accepts a list of inputs.
EMBED_BATCH = 100

SYSTEM_TEMPLATE = (
    "You are an advisor who helps Syracuse University students find student "
    "organizations. Answer using the organization pages provided below.\n\n"
    "Name the organizations you are drawing on, so the student knows where the "
    "information came from. If the provided pages do not answer the question, "
    "say so plainly rather than guessing. Remember that you can only see a few "
    "organizations at a time, so if you are asked for a complete list of every "
    "club matching some criterion, explain that you can only speak to the ones "
    "you found.\n\n"
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
    """Split one organization page into exactly two mini-documents.

    CHUNKING METHOD: structure-aware split on the page's own section heading,
    with a positional fallback.

    Every page on this site follows the same layout: a name and an "About"
    narrative describing what the organization does, followed by a
    "Contact Information" section holding the meeting day, time, location,
    officer names, emails, and joining instructions. Splitting at that heading
    produces two chunks that are each about one thing: chunk 1 answers "what is
    this group and would I like it", chunk 2 answers "when do they meet and how
    do I reach them".

    I chose this over a fixed character or token split because the two kinds of
    question students ask map cleanly onto those two halves. A blind split at,
    say, 1000 characters would cut the About paragraph in half on a long page
    and would bundle half the officer list with the mission statement on a
    short one, so neither chunk would embed cleanly toward either kind of
    query. Splitting on meaning keeps each vector focused.

    About 70% of the pages contain the "Contact Information" heading. For the
    rest, the fallback splits at the midpoint on a line boundary, which at
    least avoids cutting mid-sentence.
    """
    marker = "Contact Information"
    if marker in text:
        index = text.index(marker)
        return text[:index].strip(), text[index:].strip()

    lines = text.split("\n")
    middle = len(lines) // 2
    return "\n".join(lines[:middle]).strip(), "\n".join(lines[middle:]).strip()


def embed_batch(texts):
    """Embed a list of strings in one API call."""
    response = get_client().embeddings.create(model=EMBED_MODEL, input=texts)
    return [item.embedding for item in response.data]


def build_vector_db():
    """Create the HW4 collection from the HTML pages, once.

    If the collection already holds documents, it is reused as-is, so repeated
    runs of the app never pay to embed the pages again.
    """
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)

    if collection.count() > 0:
        return collection

    paths = sorted(glob.glob(os.path.join(HTML_DIR, "*.html")))
    ids, documents, metadatas = [], [], []

    for path in paths:
        slug = os.path.splitext(os.path.basename(path))[0]
        org_name, text = html_to_text(path)
        first, second = chunk(text)

        for part_number, part_text in ((1, first), (2, second)):
            if not part_text:
                continue
            # Prefix the org name so a chunk is identifiable on its own.
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
        progress.progress(min(stop / len(documents), 1.0),
                          text=f"Embedding organization pages... {min(stop, len(documents))}/{len(documents)}")
    progress.empty()

    return collection


def search(collection, query, k=TOP_K):
    """Return [(organization, chunk_text), ...] for the k closest chunks."""
    query_embedding = embed_batch([query])[0]
    results = collection.query(query_embeddings=[query_embedding], n_results=k)
    return list(zip(
        [m["organization"] for m in results["metadatas"][0]],
        results["documents"][0],
    ))


st.title("HW 4: Syracuse Student Organization Chatbot")

st.write(
    "Ask about Syracuse student organizations: what a club does, when it meets, "
    "who runs it, or how to join. Every organization page is split into two "
    "chunks, one describing the group and one holding its meeting and contact "
    f"details, and stored in a ChromaDB vector database. Each question retrieves "
    f"the {TOP_K} most relevant chunks and passes them to the LLM. The "
    f"conversation keeps the last {BUFFER_INTERACTIONS} interactions in memory."
)

# ---------- Build the vector DB once ----------
if "HW4_VectorDB" not in st.session_state:
    if not glob.glob(os.path.join(HTML_DIR, "*.html")):
        st.error(f"No HTML files found in the '{HTML_DIR}' folder.")
        st.stop()
    st.session_state.HW4_VectorDB = build_vector_db()

collection = st.session_state.HW4_VectorDB
st.caption(f"{COLLECTION_NAME}: {collection.count()} chunks indexed")

# ---------- Chat ----------
if "hw4_messages" not in st.session_state:
    st.session_state.hw4_messages = [
        {"role": "assistant",
         "content": "What kind of student organization are you looking for?"}
    ]

for message in st.session_state.hw4_messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask about student organizations"):
    st.session_state.hw4_messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    matches = search(collection, prompt)
    context = "\n\n".join(f"--- {org} ---\n{text}" for org, text in matches)
    system_prompt = SYSTEM_TEMPLATE.format(context=context)

    history = st.session_state.hw4_messages[-BUFFER_SIZE:]
    request_messages = [{"role": "system", "content": system_prompt}] + history

    with st.chat_message("assistant"):
        stream = get_client().chat.completions.create(
            model=CHAT_MODEL,
            messages=request_messages,
            stream=True,
        )
        answer = st.write_stream(stream)
        st.caption("Retrieved: " + ", ".join(dict.fromkeys(org for org, _ in matches)))

    st.session_state.hw4_messages.append({"role": "assistant", "content": answer})