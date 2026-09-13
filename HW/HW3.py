import requests
import streamlit as st
from bs4 import BeautifulSoup
from google import genai
from google.genai import errors as genai_errors
from google.genai import types
from openai import OpenAI

# ---------- Providers: two vendors, each with its strongest available model ----------
PROVIDERS = {
    "OpenAI (gpt-4.1)": {
        "vendor": "openai",
        "secret": "OPENAI_API_KEY",
        "model": "gpt-4.1",
    },
    "Google Gemini (gemini-3.8-flash)": {
        "vendor": "gemini",
        "secret": "GEMINI_API_KEY",
        "model": "gemini-3.8-flash",
    },
}

# Conversation memory: a buffer of the last 6 messages (3 user-assistant exchanges).
BUFFER_SIZE = 6

# Guard against very large pages blowing up the request.
MAX_DOC_CHARS = 20000

BASE_INSTRUCTIONS = (
    "You are a helpful assistant that answers questions using only the web "
    "pages provided below. Explain things clearly for someone new to the "
    "topic. If the provided pages do not contain the answer, say so plainly "
    "instead of guessing or using outside knowledge."
)


def read_url_content(url):
    try:
        # A browser User-Agent is needed because many sites return 403 to the
        # default python-requests agent.
        response = requests.get(
            url,
            headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                                   "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0 Safari/537.36"},
            timeout=15,
        )
        response.raise_for_status()  # Raise an exception for HTTP errors
        soup = BeautifulSoup(response.content, 'html.parser')
        return soup.get_text()
    except requests.RequestException as e:
        print(f"Error reading {url}: {e}")
        return None


@st.cache_data(show_spinner=False)
def fetch(url):
    """Cache page text so it is downloaded once per URL, not once per message."""
    text = read_url_content(url)
    return text[:MAX_DOC_CHARS] if text else None


def build_system_prompt(sources):
    """Instructions plus the full text of every loaded URL."""
    parts = [BASE_INSTRUCTIONS]
    for i, (url, text) in enumerate(sources, start=1):
        parts.append(f"\n\n--- DOCUMENT {i} ({url}) ---\n{text}")
    return "".join(parts)


def openai_stream(model, key, system_prompt, history):
    client = OpenAI(api_key=key)
    messages = [{"role": "system", "content": system_prompt}] + history
    stream = client.chat.completions.create(
        model=model, messages=messages, stream=True
    )
    return st.write_stream(stream)


def gemini_stream(model, key, system_prompt, history):
    client = genai.Client(api_key=key)
    contents = [
        {
            "role": "user" if m["role"] == "user" else "model",
            "parts": [{"text": m["content"]}],
        }
        for m in history
    ]
    stream = client.models.generate_content_stream(
        model=model,
        contents=contents,
        config=types.GenerateContentConfig(system_instruction=system_prompt),
    )

    def chunks():
        for chunk in stream:
            if chunk.text:
                yield chunk.text

    return st.write_stream(chunks())


# ---------- Page ----------
st.title("HW 3: Chat About a Web Page")

st.write(
    "This chatbot answers questions about up to two web pages that you choose. "
    "Enter one or two URLs in the sidebar and pick which LLM to use. The full "
    "text of each page is fetched, then placed in a system prompt that is sent "
    "with every request, so the chatbot always has the source material in front "
    "of it. For conversation memory the app keeps a buffer of the last "
    f"{BUFFER_SIZE} messages, which is {BUFFER_SIZE // 2} user-assistant "
    "exchanges. Older messages drop out of the buffer, so the chatbot will "
    "forget the early part of a long conversation, but the system prompt with "
    "the page content is added separately at request time and is never "
    "discarded. The chatbot is instructed to answer only from the pages you "
    "provide and to say so when the answer is not there."
)

# ---------- Sidebar ----------
url1 = st.sidebar.text_input("URL 1", placeholder="https://example.com")
url2 = st.sidebar.text_input("URL 2 (optional)", placeholder="https://example.com")
choice = st.sidebar.selectbox("LLM", list(PROVIDERS))

config = PROVIDERS[choice]

if config["secret"] not in st.secrets:
    st.error(f"Missing {config['secret']} in secrets.toml.")
    st.stop()
api_key = st.secrets[config["secret"]]

# ---------- Load the URLs ----------
sources = []
for url in (url1, url2):
    if url.strip():
        text = fetch(url.strip())
        if text:
            sources.append((url.strip(), text))
        else:
            st.sidebar.error(f"Could not read {url.strip()}")

if not sources:
    st.info("Add at least one URL in the sidebar to start chatting.")
    st.stop()

st.sidebar.success(f"{len(sources)} page(s) loaded")
system_prompt = build_system_prompt(sources)

# ---------- Chat ----------
# Session state is shared across pages, so use a key specific to this page.
if "hw3_messages" not in st.session_state:
    st.session_state.hw3_messages = []

for message in st.session_state.hw3_messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question about the page(s)"):
    st.session_state.hw3_messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    history = st.session_state.hw3_messages[-BUFFER_SIZE:]

    with st.chat_message("assistant"):
        if config["vendor"] == "openai":
            answer = openai_stream(config["model"], api_key, system_prompt, history)
        else:
            try:
                answer = gemini_stream(config["model"], api_key, system_prompt, history)
            except genai_errors.ServerError:
                answer = "Gemini is busy right now. Please ask again in a moment."
                st.write(answer)
        st.caption(f"{choice} | {len(history)} of {len(st.session_state.hw3_messages)} "
                   f"messages in buffer | {len(sources)} document(s)")

    st.session_state.hw3_messages.append({"role": "assistant", "content": answer})