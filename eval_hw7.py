"""Evaluate the HW7 news bot. Run after build_news_db.py:   python3 eval_hw7.py

Writes data/eval_results.md, which feeds the write-up. Tests:
  1. Ranking quality for each LLM's scores against a hand-labeled gold set
  2. Agreement between the lower-cost and higher-cost LLM
  3. "Find news about X" for clients: does the filter return every article?
  4. "Find news about X" for topics: does semantic search surface the right stories?
  5. Cost and time from the build log
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from openai import OpenAI

DF = pd.read_csv("data/news_index.csv")
EMB = np.load("data/news_embeddings.npy")
GOLD = pd.read_csv("data/gold_set.csv")
LOG = json.loads(Path("data/build_log.json").read_text())
RAW = pd.read_csv("data/news.csv")
RAW["company_name"] = RAW["company_name"].str.strip()

SCORERS = {"low": LOG["scorers"]["low"]["model"], "high": LOG["scorers"]["high"]["model"]}

# Topic queries a lawyer might ask, with headline fragments that should appear in the top 5.
TOPIC_TESTS = [
    ("antitrust investigations", ["Antitrust", "antitrust probes"]),
    ("patent infringement lawsuit", ["Amazon sues Nokia", "patent lawsuit", "patent infringement"]),
    ("shareholder class action lawsuits", ["Class Action", "Sued Over", "Fraud Investigation"]),
    ("cyberattack causing a service outage", ["DDoS", "Cyberattack"]),
    ("private equity take-private deals", ["Thoughtworks", "R1 RCM", "Instructure"]),
    ("tax court ruling", ["Tax Court", "back taxes"]),
    ("mass layoffs", ["laying off", "15,000"]),
    ("data privacy settlement", ["settles Texas lawsuit", "biometric"]),
]

# Client names typed the way a user might, and the dataset name they should map to.
CLIENT_TESTS = [("Nvidia", "Nvidia"), ("Google", "Alphabet"), ("Meta", "Facebook"),
                ("JPMorgan", "JPMorgan Chase"), ("Samsung", "Samsung Electronics"),
                ("Coca-Cola", "Coca-Cola"), ("Nokia", "Nokia"), ("HP", "HP")]


def api_key():
    text = Path(".streamlit/secrets.toml").read_text()
    return re.search(r'^\s*OPENAI_API_KEY\s*=\s*"([^"]+)"', text, re.M).group(1)


def ranked_stories(key):
    """Same ordering as rank_interesting_news in HW7.py."""
    return (DF.sort_values([f"score_{key}", "sources", "date"], ascending=[False, False, False])
              .drop_duplicates("story_id").reset_index(drop=True))


def story_of(url, company):
    match = DF[(DF["URL"] == url) & (DF["company_name"] == company)]
    return int(match["story_id"].iloc[0]) if len(match) else None


def ranking_report(key):
    ranked = ranked_stories(key)
    position = {sid: i + 1 for i, sid in enumerate(ranked["story_id"])}
    gold = GOLD.copy()
    gold["story_id"] = [story_of(u, c) for u, c in zip(gold["url"], gold["company_name"])]
    gold["rank"] = gold["story_id"].map(position)
    gold["score"] = [int(DF.loc[(DF["URL"] == u) & (DF["company_name"] == c), f"score_{key}"].iloc[0])
                     for u, c in zip(gold["url"], gold["company_name"])]
    gold["about"] = [bool(DF.loc[(DF["URL"] == u) & (DF["company_name"] == c), f"about_{key}"].iloc[0])
                     for u, c in zip(gold["url"], gold["company_name"])]
    high, noise = gold[gold["label"] == "HIGH"], gold[gold["label"] == "NOISE"]
    collisions = noise[noise["why"].str.startswith("Name collision")]
    return {
        "high_in_top10": int((high["rank"] <= 10).sum()),
        "high_in_top25": int((high["rank"] <= 25).sum()),
        "high_total": int(len(high)),
        "high_median_rank": float(high["rank"].median()),
        "noise_in_top50": int((noise["rank"] <= 50).sum()),
        "noise_total": int(len(noise)),
        "mean_score_high": round(float(high["score"].mean()), 2),
        "mean_score_noise": round(float(noise["score"].mean()), 2),
        "collisions_flagged": int((~collisions["about"]).sum()),
        "collisions_total": int(len(collisions)),
        "top10": ranked.head(10)[["company_name", "title", f"score_{key}", f"category_{key}", "sources"]],
        "gold": gold,
    }


def client_tests():
    sys_resolve = load_resolver()
    rows = []
    for typed, expected in CLIENT_TESTS:
        got = sys_resolve(typed)
        truth = RAW.drop_duplicates(["company_name", "URL"])
        expected_count = int((truth["company_name"] == expected).sum())
        returned = int((DF["company_name"] == got).sum()) if got else 0
        rows.append((typed, expected, got, expected_count, returned,
                     "pass" if got == expected and returned == expected_count else "FAIL"))
    return rows


def load_resolver():
    """Pull resolve_company and its aliases out of HW7.py so we test the real code."""
    src = Path("HW/HW7.py").read_text()
    start = src.index("ALIASES = {")
    alias_src = src[start:src.index("}", start) + 1]
    fn_src = src[src.index("def resolve_company"):src.index("def closest_companies")]
    ns = {"re": re, "difflib": __import__("difflib")}
    exec(alias_src, ns)
    exec(fn_src, ns)
    companies = DF["company_name"].unique()
    return lambda name: ns["resolve_company"](name, companies)


def topic_tests(client):
    rows = []
    for query, expected in TOPIC_TESTS:
        q = np.array(client.embeddings.create(model="text-embedding-3-small", input=query).data[0].embedding)
        q = q / np.linalg.norm(q)
        order = np.argsort(-(EMB @ q))      # cosine similarity; same ranking Chroma gives
        seen, top = set(), []
        for i in order:
            sid = DF["story_id"].iloc[i]
            if sid in seen:
                continue
            seen.add(sid)
            top.append(DF["title"].iloc[i])
            if len(top) == 5:
                break
        hit = any(any(e.lower() in t.lower() for e in expected) for t in top)
        rows.append((query, "hit" if hit else "MISS", top[0][:80]))
    return rows


def main():
    out = ["# HW7 evaluation results", ""]
    reports = {k: ranking_report(k) for k in SCORERS}

    out += ["## 1. Ranking quality against the gold set", "",
            "| Metric | " + " | ".join(SCORERS.values()) + " |", "|---|---|---|"]
    def row(label, fn):
        out.append(f"| {label} | " + " | ".join(fn(reports[k]) for k in SCORERS) + " |")
    row("HIGH stories in top 10", lambda r: f"{r['high_in_top10']} of {r['high_total']}")
    row("HIGH stories in top 25", lambda r: f"{r['high_in_top25']} of {r['high_total']}")
    row("Median rank of HIGH stories", lambda r: f"{r['high_median_rank']:.0f}")
    row("NOISE stories in top 50", lambda r: f"{r['noise_in_top50']} of {r['noise_total']}")
    row("Mean score, HIGH vs NOISE", lambda r: f"{r['mean_score_high']} vs {r['mean_score_noise']}")
    row("Name collisions caught", lambda r: f"{r['collisions_flagged']} of {r['collisions_total']}")

    for k in SCORERS:
        out += ["", f"### Top 10 from {SCORERS[k]}", "", "| # | Client | Headline | Score | Category | Outlets |", "|---|---|---|---|---|---|"]
        for i, r in enumerate(reports[k]["top10"].itertuples(index=False), 1):
            out.append(f"| {i} | {r[0]} | {r[1][:90]} | {r[2]} | {r[3]} | {r[4]} |")

    low_top = set(ranked_stories("low").head(10)["story_id"])
    high_top = set(ranked_stories("high").head(10)["story_id"])
    corr = DF["score_low"].rank().corr(DF["score_high"].rank())   # Spearman, without needing scipy
    exact = (DF["score_low"] == DF["score_high"]).mean()
    within1 = ((DF["score_low"] - DF["score_high"]).abs() <= 1).mean()
    out += ["", "## 2. Agreement between the two LLMs", "",
            f"- Stories in both top-10 lists: {len(low_top & high_top)} of 10",
            f"- Spearman correlation of scores across all {len(DF)} articles: {corr:.2f}",
            f"- Same score: {exact:.0%} of articles. Within 1 point: {within1:.0%}"]

    out += ["", "## 3. Find news about a client", "", "| Typed | Expected | Resolved to | Articles in CSV | Returned | Result |", "|---|---|---|---|---|---|"]
    for r in client_tests():
        out.append("| " + " | ".join(str(x) for x in r) + " |")

    out += ["", "## 4. Find news about a topic (story expected in top 5)", "", "| Query | Result | Top result |", "|---|---|---|"]
    for r in topic_tests(OpenAI(api_key=api_key())):
        out.append("| " + " | ".join(r) + " |")

    out += ["", "## 5. Cost and time to build the index", "", "| | Calls | Input tokens | Output tokens | Seconds | Unscored |", "|---|---|---|---|---|---|"]
    for k, v in LOG["scorers"].items():
        out.append(f"| {v['model']} | {v['calls']} | {v['prompt_tokens']:,} | {v['completion_tokens']:,} | {v['seconds']} | {v['unscored']} |")
    out += [f"", f"{LOG['raw_rows']} rows in, {LOG['articles']} unique articles, {LOG['stories']} distinct stories. "
            f"Embedding took {LOG['embedding_seconds']}s."]

    text = "\n".join(out)
    Path("data/eval_results.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()