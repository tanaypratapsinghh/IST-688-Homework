"""Build the HW7 news index once, outside the app.

Run from the repo root:   python3 build_news_db.py

It reads data/news.csv and writes three files the app loads at startup:
  data/news_index.csv       one row per unique article, with story ids and scores
  data/news_embeddings.npy  one embedding per row, same order as the CSV
  data/build_log.json       tokens, time, and story counts for the write-up

Doing this offline means the app never embeds or scores 1,090 articles while a
user waits. It only embeds the user's search query.
"""
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from openai import OpenAI

SRC = Path("data/news.csv")
OUT_CSV = Path("data/news_index.csv")
OUT_EMB = Path("data/news_embeddings.npy")
OUT_LOG = Path("data/build_log.json")

EMBED_MODEL = "text-embedding-3-small"
# The two LLMs being compared: one lower-cost, one higher-cost.
SCORERS = {"low": "gpt-4.1-nano", "high": "gpt-4.1"}
BATCH = 20
WORKERS = 4          # parallel scoring calls; lower this if you hit rate limits

# Two headlines count as the same story if their titles overlap heavily or their
# embeddings are very close. Both checks only run within the same company.
TITLE_JACCARD = 0.30
EMBED_COSINE = 0.85

CATEGORIES = ["Litigation", "Regulatory", "M&A", "Cybersecurity", "Labor",
              "Financial", "Leadership", "Operations", "Product", "Noise"]

RUBRIC = """You score news articles for a large global law firm that monitors news about its clients.
For each article, judge how important it is for the firm to know about, from the client's legal and business-risk point of view.

Scoring guide (1 to 10):
9-10  Litigation, enforcement, investigations, antitrust, sanctions, major fines or settlements,
      large M&A or take-privates involving the client, data breaches or cyberattacks, bankruptcy.
6-8   Regulatory approvals or new rules affecting the client, mass layoffs or strikes, IP disputes,
      executive departures, large write-downs or earnings shocks, major contracts.
3-5   Routine business news: product launches, partnerships, funding rounds, earnings previews, expansion.
1-2   Not really about this client (name collision), consumer deals or coupons, product reviews,
      stock-promotion pieces, lifestyle or travel content.

Also decide about_company: false if the article only matches the client's name by coincidence
(for example "HP" meaning horsepower, or a person who shares the company's name).
Give a one-sentence reason a lawyer would find useful. Score every article you are given."""

SCORE_TOOL = {
    "type": "function",
    "function": {
        "name": "record_scores",
        "description": "Record a score for every article in the batch.",
        "parameters": {
            "type": "object",
            "properties": {
                "scores": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "integer"},
                            "score": {"type": "integer", "minimum": 1, "maximum": 10},
                            "category": {"type": "string", "enum": CATEGORIES},
                            "about_company": {"type": "boolean"},
                            "reason": {"type": "string"},
                        },
                        "required": ["id", "score", "category", "about_company", "reason"],
                    },
                }
            },
            "required": ["scores"],
        },
    },
}

STOP = set("a an the and or of to in on for with by at from as is are be was were its it this "
           "that after over into amid new says say said will can could has have how what why who".split())


def api_key():
    """Read the key from the same secrets file the Streamlit app uses."""
    try:
        text = Path(".streamlit/secrets.toml").read_text()
        return re.search(r'^\s*OPENAI_API_KEY\s*=\s*"([^"]+)"', text, re.M).group(1)
    except Exception:
        sys.exit("Put OPENAI_API_KEY in .streamlit/secrets.toml first.")


def load():
    df = pd.read_csv(SRC)
    df["company_name"] = df["company_name"].str.strip()
    before = len(df)
    df = df.drop_duplicates(["company_name", "URL"]).reset_index(drop=True)
    parts = df["Document"].str.split(" Description:", n=1, expand=True)
    # Collapse line breaks and runs of spaces so each article stays on one line.
    df["title"] = parts[0].str.replace(r"\s+", " ", regex=True).str.strip()
    df["description"] = parts[1].fillna("").str.replace(r"\s+", " ", regex=True).str.strip()
    df["date"] = pd.to_datetime(df["Date"], utc=True, format="mixed").dt.strftime("%Y-%m-%d")
    df["id"] = df.index.astype(int)
    print(f"Loaded {before} rows, {len(df)} after removing duplicate company+URL pairs")
    return df, before


def title_tokens(title, company):
    company_words = set(re.findall(r"[a-z0-9]+", company.lower()))
    words = {w.strip(".") for w in re.findall(r"[a-z0-9$€£.]+", title.lower())}
    return {w for w in words if w not in STOP and w not in company_words and len(w) > 1}


def assign_stories(df, emb):
    """Union-find over same-company pairs that are near-duplicates."""
    parent = list(range(len(df)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    tokens = [title_tokens(t, c) for t, c in zip(df["title"], df["company_name"])]
    for _, group in df.groupby("company_name"):
        idx = group.index.to_list()
        sims = emb[idx] @ emb[idx].T            # embeddings are unit length
        for a in range(len(idx)):
            for b in range(a + 1, len(idx)):
                i, j = idx[a], idx[b]
                inter = len(tokens[i] & tokens[j])
                union = len(tokens[i] | tokens[j]) or 1
                if inter / union >= TITLE_JACCARD or sims[a, b] >= EMBED_COSINE:
                    parent[find(i)] = find(j)

    roots = [find(i) for i in range(len(df))]
    remap = {r: n for n, r in enumerate(dict.fromkeys(roots))}
    df["story_id"] = [remap[r] for r in roots]
    df["sources"] = df.groupby("story_id")["id"].transform("count").astype(int)
    print(f"Grouped into {df['story_id'].nunique()} distinct stories")


def embed(client, texts):
    vectors = []
    for start in range(0, len(texts), 100):
        resp = client.embeddings.create(model=EMBED_MODEL, input=texts[start:start + 100])
        vectors.extend(item.embedding for item in resp.data)
    arr = np.array(vectors, dtype=np.float32)
    return arr / np.linalg.norm(arr, axis=1, keepdims=True)


def score_batch(client, model, rows):
    lines = [f"[{r.id}] Client: {r.company_name} | {r.date} | {r.title} | {r.description[:400]}"
             for r in rows.itertuples()]
    for attempt in range(4):
        try:
            resp = client.chat.completions.create(
                model=model,
                temperature=0,
                messages=[{"role": "system", "content": RUBRIC},
                          {"role": "user", "content": "\n".join(lines)}],
                tools=[SCORE_TOOL],
                tool_choice={"type": "function", "function": {"name": "record_scores"}},
            )
            args = json.loads(resp.choices[0].message.tool_calls[0].function.arguments)
            return args["scores"], resp.usage
        except Exception as error:
            wait = 2 ** attempt
            print(f"    retrying after error ({error.__class__.__name__}), waiting {wait}s")
            time.sleep(wait)
    return [], None


def score_all(client, df, key, model):
    print(f"Scoring with {model} ...")
    results, usage = {}, {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0}
    start = time.time()
    pending = df
    for _ in range(3):                         # retry any articles the model skipped
        batches = [pending.iloc[s:s + BATCH] for s in range(0, len(pending), BATCH)]
        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            outcomes = list(pool.map(lambda b: score_batch(client, model, b), batches))
        for scores, u in outcomes:
            for item in scores:
                results[int(item["id"])] = item
            if u:
                usage["prompt_tokens"] += u.prompt_tokens
                usage["completion_tokens"] += u.completion_tokens
                usage["calls"] += 1
        pending = df[~df["id"].isin(results)]
        if pending.empty:
            break
    usage["seconds"] = round(time.time() - start, 1)
    usage["unscored"] = int(len(pending))

    df[f"score_{key}"] = [max(1, min(10, int(results.get(i, {}).get("score", 1)))) for i in df["id"]]
    df[f"category_{key}"] = [results.get(i, {}).get("category", "Noise") for i in df["id"]]
    df[f"about_{key}"] = [bool(results.get(i, {}).get("about_company", True)) for i in df["id"]]
    df[f"reason_{key}"] = [results.get(i, {}).get("reason", "") for i in df["id"]]
    # A name collision can never rank as important news about the client.
    df.loc[~df[f"about_{key}"], f"score_{key}"] = 1
    print(f"  done in {usage['seconds']}s, {usage['calls']} calls, {usage['unscored']} unscored")
    return usage


def main():
    client = OpenAI(api_key=api_key())
    df, raw_rows = load()

    print("Embedding articles ...")
    t = time.time()
    emb = embed(client, (df["company_name"] + ": " + df["title"] + ". " + df["description"]).tolist())
    embed_seconds = round(time.time() - t, 1)

    assign_stories(df, emb)
    usage = {key: score_all(client, df, key, model) for key, model in SCORERS.items()}

    df.to_csv(OUT_CSV, index=False)
    np.save(OUT_EMB, emb)
    OUT_LOG.write_text(json.dumps({
        "raw_rows": raw_rows, "articles": len(df), "stories": int(df["story_id"].nunique()),
        "embedding_model": EMBED_MODEL, "embedding_seconds": embed_seconds,
        "scorers": {key: {"model": SCORERS[key], **usage[key]} for key in SCORERS},
    }, indent=2))
    print(f"Wrote {OUT_CSV}, {OUT_EMB}, {OUT_LOG}")


if __name__ == "__main__":
    main()