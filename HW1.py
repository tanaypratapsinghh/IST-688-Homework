import time

import streamlit as st
from openai import OpenAI, AuthenticationError, OpenAIError
from pypdf import PdfReader

# ---------- Models to compare (HW1, item 3b) ----------
MODELS = ["gpt-3.5-turbo", "gpt-4.1", "gpt-5-chat-latest", "gpt-5-nano"]


def read_pdf(uploaded_file) -> str:
    """Read an uploaded PDF file and return its text."""
    reader = PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        text += (page.extract_text() or "") + "\n"
    return text


def validate_api_key(key: str) -> bool:
    """Make a cheap call to OpenAI to confirm the key works."""
    try:
        OpenAI(api_key=key).models.list()
        return True
    except AuthenticationError:
        return False
    except OpenAIError:
        return False


st.title("📄 Document question answering")
st.write(
    "Upload a .txt or .pdf document below and ask a question about it – GPT will answer! "
    "To use this app, you need an OpenAI API key, which you can get "
    "[here](https://platform.openai.com/account/api-keys)."
)

openai_api_key = st.text_input("OpenAI API Key", type="password")

if not openai_api_key:
    st.info("Please add your OpenAI API key to continue.", icon="🗝️")
elif not validate_api_key(openai_api_key):
    st.error("Invalid OpenAI API key. Please check it and try again.", icon="🚨")
else:
    st.success("API key is valid.", icon="✅")
    client = OpenAI(api_key=openai_api_key)

    model = st.sidebar.selectbox("Model", MODELS)

    uploaded_file = st.file_uploader(
        "Upload a document (.txt or .pdf)", type=("txt", "pdf")
    )

    question = st.text_area(
        "Now ask a question about the document!",
        placeholder="Is this course hard?",
        disabled=not uploaded_file,
    )

    if uploaded_file and question:
        document = None
        file_extension = uploaded_file.name.split('.')[-1]
        if file_extension == 'txt':
            document = uploaded_file.read().decode()
        elif file_extension == 'pdf':
            document = read_pdf(uploaded_file)
        else:
            st.error("Unsupported file type.")

        if document:
            messages = [
                {
                    "role": "user",
                    "content": f"Here's a document: {document} \n\n---\n\n {question}",
                }
            ]

            st.caption(f"Model: {model}")
            start = time.time()
            stream = client.chat.completions.create(
                model=model,
                messages=messages,
                stream=True,
            )
            st.write_stream(stream)
            st.caption(f"Response time: {time.time() - start:.1f}s")
