import time

import requests
import streamlit as st
from bs4 import BeautifulSoup
from google import genai
from openai import OpenAI

# ---------- Providers and models ----------
# Each provider has a cheaper "basic" model and a stronger "advanced" model.
PROVIDERS = {
    "OpenAI": {
        "secret": "OPENAI_API_KEY",
        "basic": "gpt-4.1-nano",
        "advanced": "gpt-4.1",
    },
    "Google Gemini": {
        "secret": "GEMINI_API_KEY",
        "basic": "gemini-2.5-flash",
        "advanced": "gemini-2.5-pro",
    },
}

SUMMARY_TYPES = {
    "Summarize the document in 100 words":
        "Summarize the page in about 100 words.",
    "Summarize the document in 2 connecting paragraphs":
        "Summarize the page in exactly two connecting paragraphs.",
    "Summarize the document in 5 bullet points":
        "Summarize the page in exactly five bullet points.",
}

LANGUAGES = ["English", "Spanish", "French", "German", "Hindi"]


def read_url_content(url):
    try:
        # A browser User-Agent is needed because many sites (Wikipedia included)
        # return 403 to the default python-requests agent.
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
def key_is_valid(provider: str, key: str) -> bool:
    """Confirm the key works for the selected provider. Cached so it runs once."""
    try:
        if provider == "OpenAI":
            OpenAI(api_key=key).models.list()
        else:
            list(genai.Client(api_key=key).models.list())
        return True
    except Exception:
        return False


def summarize(provider: str, model: str, key: str, prompt: str):
    """Stream a summary from the selected provider."""
    if provider == "OpenAI":
        client = OpenAI(api_key=key)
        stream = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
        )
        st.write_stream(stream)
    else:
        client = genai.Client(api_key=key)
        stream = client.models.generate_content_stream(model=model, contents=prompt)
        st.write_stream(chunk.text for chunk in stream if chunk.text)


st.title("HW 2: URL Summarizer")

# Item 4: the URL goes at the top of the screen, not in the sidebar.
url = st.text_input("Web page URL", placeholder="https://example.com")

# ---------- Sidebar controls ----------
provider = st.sidebar.selectbox("LLM provider", list(PROVIDERS))
use_advanced = st.sidebar.checkbox("Use advanced model")
summary_type = st.sidebar.selectbox("Type of summary", list(SUMMARY_TYPES))
language = st.sidebar.selectbox("Output language", LANGUAGES)

config = PROVIDERS[provider]
model = config["advanced"] if use_advanced else config["basic"]
st.sidebar.caption(f"Model: {model}")

# ---------- Key handling (item 11b) ----------
secret_name = config["secret"]
if secret_name not in st.secrets:
    st.error(f"Missing {secret_name} in secrets.toml.")
    st.stop()

api_key = st.secrets[secret_name]
if not key_is_valid(provider, api_key):
    st.error(f"The {provider} API key is not valid.")
    st.stop()

# ---------- Summarize ----------
if url:
    document = read_url_content(url)

    if not document:
        st.error("Could not read that URL. Check the address and try again.")
    else:
        prompt = (
            f"Here's the content of a web page: {document}\n\n---\n\n"
            f"{SUMMARY_TYPES[summary_type]} Write the summary in {language}."
        )

        with st.spinner(f"Summarizing with {model}..."):
            start = time.time()
            summarize(provider, model, api_key, prompt)
        st.caption(f"{provider} / {model} — {time.time() - start:.1f}s")