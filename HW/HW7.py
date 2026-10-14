import difflib
import json
import re
import sys
import time

# ChromaDB needs a newer sqlite3 than Streamlit Cloud ships with.
try:
    __import__("pysqlite3")
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass

import chromadb
import numpy as np
import pandas as pd
import streamlit as st
from openai import OpenAI

INDEX_CSV = "data/news_index.csv"
INDEX_EMB = "data/news_embeddings.npy"
EMBED_MODEL = "text-embedding-3-small"

# The lower-cost and higher-cost LLMs being compared. Each one uses the
# importance scores it produced itself in build_news_db.py.
MODELS = {
    "Lower cost: gpt-4.1-nano": ("gpt-4.1-nano", "low"),
    "Higher cost: gpt-4.1": ("gpt-4.1", "high"),
}

BUFFER_SIZE = 10          # last 5 user-assistant exchanges

# Common names people use that differ from the names in the dataset.
ALIASES = {
    "google": "Alphabet", "meta": "Facebook", "jpmorgan": "JPMorgan Chase",
    "jp morgan": "JPMorgan Chase", "chase": "JPMorgan Chase", "disney": "Walt Disney",
    "samsung": "Samsung Electronics", "foxconn": "Hon Hai Precision",
    "hpe": "Hewlett Packard Enterprise", "tcs": "Tata Consultancy Services",
    "coke": "Coca-Cola", "amd": "Advanced Micro Devices (AMD)", "bbva": "BBVA-Banco Bilbao Vizcaya",
    "aws": "Amazon", "blackstone": "Blackstone Inc.", "carlyle": "The Carlyle Group",
    "stmicro": "STMicroelectronics",
}

SYSTEM_PROMPT = """You are a news analyst for a large global law firm that monitors news about its clients.
You report ONLY on articles in the firm's news database, which covers client news from July 30 to August 6, 2024,
and you can only see those articles through your tools. Never invent articles, dates, sources, or links, and don't
add facts that aren't in the articles.

- When asked for the most interesting or important news, call rank_interesting_news. Present the stories as a
  numbered list in the order returned. For each one give the headline as a markdown link, the client, the date,
  and one or two sentences of context on why it matters to a law firm: the legal, regulatory, deal, or risk angle.
  Mention when several outlets covered the same story.
- When asked for news about a company or topic, call find_news. Put the company in the company field when the user
  names a client. If the client isn't in the database, say so and offer the closest matches the tool returns.
- If the tools return nothing relevant, say that plainly.
- If a request is ambiguous, ask one short clarifying question.
- Be concise and professional."""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "rank_interesting_news",
            "description": (
                "Rank the most interesting news stories for the law firm across ALL articles in the database, "
                "most important first, one entry per story. Use for requests like 'most interesting news', "
                "'top stories', or 'what should we know about this week'. Optionally limit to one client."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "company": {"type": "string", "description": "Optional client company to limit the ranking to."},
                    "top_n": {"type": "integer", "description": "How many stories to return, 1 to 20. Default 10."},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_news",
            "description": (
                "Find news about a specific client company or topic, for example 'Nvidia', 'antitrust', "
                "or 'patent lawsuits'. Returns matching stories with their importance scores."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What to search for, written as a short description."},
                    "company": {"type": "string", "description": "The client company, if the user named one."},
                    "top_n": {"type": "integer", "description": "How many stories to return, 1 to 20. Default 8."},
                },
                "required": ["query"],
            },
        },
    },
]


def get_client():
    return OpenAI(api_key=st.secrets["OPENAI_API_KEY"])


@st.cache_resource(show_spinner="Loading the news index...")
def load_index():
    """Load the prebuilt index. No embedding or scoring happens here."""
    df = pd.read_csv(INDEX_CSV)
    df["description"] = df["description"].fillna("")
    for key in ("low", "high"):
        df[f"reason_{key}"] = df[f"reason_{key}"].fillna("")
    emb = np.load(INDEX_EMB)

    collection = chromadb.EphemeralClient().get_or_create_collection("hw7_news")
    if collection.count() == 0:
        for start in range(0, len(df), 500):
            part = df.iloc[start:start + 500]
            collection.add(
                ids=part["id"].astype(str).tolist(),
                embeddings=emb[start:start + 500].tolist(),
                documents=(part["title"] + ". " + part["description"]).tolist(),
                metadatas=[{"company": c, "date": d, "story_id": int(s)}
                           for c, d, s in zip(part["company_name"], part["date"], part["story_id"])],
            )
    return df, collection


def resolve_company(name, companies):
    """Map what the user typed to a company name in the dataset, or None."""
    if not name:
        return None
    text = name.strip().lower()
    text = ALIASES.get(text, text).lower()
    exact = {c.lower(): c for c in companies}
    if text in exact:
        return exact[text]
    # "samsung" matches "Samsung Electronics"
    word_match = [c for c in companies if re.search(rf"\b{re.escape(text)}\b", c.lower())]
    if word_match:
        return min(word_match, key=len)
    # "banco santander" or "nvidia corp" contain a company name
    contained = [c for c in companies if re.search(rf"\b{re.escape(c.lower())}\b", text)]
    if contained:
        return max(contained, key=len)
    close = difflib.get_close_matches(text, list(exact), n=1, cutoff=0.8)
    return exact[close[0]] if close else None


def closest_companies(name, companies, n=5):
    lower = {c.lower(): c for c in companies}
    return [lower[m] for m in difflib.get_close_matches(name.lower(), list(lower), n=n, cutoff=0.4)]


def story_rows(rows, key):
    """Format articles for the model: one entry per story, with its context."""
    out = []
    for r in rows.itertuples():
        out.append({
            "headline": r.title,
            "client": r.company_name,
            "date": r.date,
            "url": r.URL,
            "summary": r.description[:500],
            "importance": int(getattr(r, f"score_{key}")),
            "category": getattr(r, f"category_{key}"),
            "why_it_matters": getattr(r, f"reason_{key}"),
            "outlets_covering_story": int(r.sources),
        })
    return out


def rank_interesting_news(df, key, company=None, top_n=10):
    top_n = max(1, min(int(top_n or 10), 20))
    pool = df
    if company:
        resolved = resolve_company(company, df["company_name"].unique())
        if not resolved:
            return {"error": f"No client named '{company}' in the database.",
                    "closest_matches": closest_companies(company, df["company_name"].unique())}
        pool = df[df["company_name"] == resolved]
    ranked = (pool.sort_values([f"score_{key}", "sources", "date"], ascending=[False, False, False])
                  .drop_duplicates("story_id")
                  .head(top_n))
    return {"stories_considered": int(pool["story_id"].nunique()), "results": story_rows(ranked, key)}


def find_news(df, collection, key, query, company=None, top_n=8):
    top_n = max(1, min(int(top_n or 8), 20))
    resolved = resolve_company(company, df["company_name"].unique()) if company else None
    if company and not resolved:
        return {"error": f"No client named '{company}' in the database.",
                "closest_matches": closest_companies(company, df["company_name"].unique())}

    if resolved:
        # A named client is a structured filter: every article for that client,
        # most important first. Vector search here could miss some of them.
        pool = df[df["company_name"] == resolved]
        ranked = (pool.sort_values([f"score_{key}", "sources", "date"], ascending=[False, False, False])
                      .drop_duplicates("story_id").head(top_n))
        return {"client": resolved, "articles_for_client": int(len(pool)),
                "stories_for_client": int(pool["story_id"].nunique()), "results": story_rows(ranked, key)}

    # A topic is a semantic search over the vector index.
    vector = get_client().embeddings.create(model=EMBED_MODEL, input=query).data[0].embedding
    hits = collection.query(query_embeddings=[vector], n_results=min(top_n * 4, len(df)))
    ids = [int(i) for i in hits["ids"][0]]
    # Embeddings are unit length, so squared L2 distance d relates to cosine as 1 - d/2.
    similarity = {i: 1 - d / 2 for i, d in zip(ids, hits["distances"][0])}
    pool = df.set_index("id").loc[ids].reset_index()
    pool["similarity"] = pool["id"].map(similarity)
    pool = pool[pool["similarity"] >= 0.25].drop_duplicates("story_id").head(top_n)
    results = story_rows(pool, key)
    for item, sim in zip(results, pool["similarity"]):
        item["match"] = round(float(sim), 2)
    return {"query": query, "results": results}


def run_tool(name, args, df, collection, key):
    if name == "rank_interesting_news":
        return rank_interesting_news(df, key, args.get("company"), args.get("top_n", 10))
    if name == "find_news":
        return find_news(df, collection, key, args.get("query", ""), args.get("company"), args.get("top_n", 8))
    return {"error": f"Unknown tool {name}"}


# ---------------- Page ----------------
st.title("HW 7: Client News Monitor")
st.write(
    "A news bot for a law firm that monitors its clients. It answers only from the firm's article set "
    "(July 30 to August 6, 2024). Ask for the most interesting news, or for news about a client or topic."
)

df, collection = load_index()

choice = st.sidebar.radio("LLM", list(MODELS))
model, key = MODELS[choice]
st.sidebar.caption(f"Ranking uses the importance scores {model} assigned when the index was built.")
st.sidebar.write(f"{len(df):,} articles · {df['story_id'].nunique():,} stories · "
                 f"{df['company_name'].nunique()} clients")
if st.sidebar.button("Clear conversation"):
    st.session_state.hw7_messages = []

if "hw7_messages" not in st.session_state:
    st.session_state.hw7_messages = []

for message in st.session_state.hw7_messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Find the most interesting news, or find news about a client"):
    st.session_state.hw7_messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    client = get_client()
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + st.session_state.hw7_messages[-BUFFER_SIZE:]
    start = time.time()

    with st.chat_message("assistant"):
        # First call: the model decides whether it needs a tool, and with what arguments.
        first = client.chat.completions.create(
            model=model, messages=messages, tools=TOOLS, tool_choice="auto", temperature=0
        )
        reply = first.choices[0].message

        if not reply.tool_calls:
            answer = reply.content or ""
            st.markdown(answer)
            used = []
        else:
            messages.append(reply)
            used, shown = [], []
            for call in reply.tool_calls:
                args = json.loads(call.function.arguments or "{}")
                result = run_tool(call.function.name, args, df, collection, key)
                used.append(f"{call.function.name}({', '.join(f'{k}={v!r}' for k, v in args.items())})")
                shown.extend(result.get("results", []))
                messages.append({"role": "tool", "tool_call_id": call.id, "content": json.dumps(result)})

            # Second call: write the answer from the tool results, with no tools available.
            stream = client.chat.completions.create(model=model, messages=messages, stream=True, temperature=0)
            answer = st.write_stream(stream)

            if shown:
                with st.expander("Articles the answer was based on"):
                    st.dataframe(
                        pd.DataFrame(shown)[["importance", "category", "client", "date", "headline", "url"]],
                        hide_index=True,
                    )

        st.caption(f"{model} · {time.time() - start:.1f}s" + (" · " + "; ".join(used) if used else ""))

    st.session_state.hw7_messages.append({"role": "assistant", "content": answer})