# HW7 evaluation results

## 1. Ranking quality against the gold set

| Metric | gpt-4.1-nano | gpt-4.1 |
|---|---|---|
| HIGH stories in top 10 | 6 of 20 | 7 of 20 |
| HIGH stories in top 25 | 11 of 20 | 11 of 20 |
| Median rank of HIGH stories | 23 | 22 |
| NOISE stories in top 50 | 0 of 20 | 0 of 20 |
| Mean score, HIGH vs NOISE | 7.1 vs 1.85 | 8.8 vs 1.45 |
| Name collisions caught | 5 of 7 | 7 of 7 |

### Top 10 from gpt-4.1-nano

| # | Client | Headline | Score | Category | Outlets |
|---|---|---|---|---|---|
| 1 | Alphabet | Google Antitrust Ruling: Landmark Decision Challenges Big Tech Dominance | 10 | Litigation | 1 |
| 2 | Apax Partners | Apax Partners Taking Thoughtworks Private | 9 | M&A | 6 |
| 3 | Nokia | Amazon sues Nokia for infringing multiple AWS cloud computing patents | 9 | Litigation | 5 |
| 4 | Coca-Cola | Coca-Cola to pay $6 billion in back taxes and interest to the IRS after tax court ruling | 9 | Tax/Financial | 4 |
| 5 | Lightspeed Venture Partners | Cybersecurity innovator Clutch Security nets $8.5m in funding led by Lightspeed | 9 | Cybersecurity | 4 |
| 6 | JPMorgan Chase | JPMorgan considers litigation against consumer watchdog over Zelle | 9 | Litigation | 3 |
| 7 | Clayton, Dubilier & Rice | R1 RCM to be Acquired by TowerBrook and CD&R for $8.9 Billion | 9 | M&A | 3 |
| 8 | Tesla | Tesla was in full self-driving mode when it fatally hit Seattle-area motorcyclist: Police | 9 | Litigation | 3 |
| 9 | Advent International | Nuvei, in a take-private acquisition deal, to buy Brazil’s Pay2All | 9 | M&A | 2 |
| 10 | DXC Technology | DXC Investors Have Opportunity to Lead DXC Technology Company Securities Fraud Lawsuit wit | 9 | Litigation | 2 |

### Top 10 from gpt-4.1

| # | Client | Headline | Score | Category | Outlets |
|---|---|---|---|---|---|
| 1 | Apax Partners | Apax Partners Taking Thoughtworks Private | 10 | M&A | 6 |
| 2 | Coca-Cola | Coca-Cola to pay $6 billion in back taxes and interest to the IRS after tax court ruling | 10 | Financial | 4 |
| 3 | Clayton, Dubilier & Rice | R1 RCM to Go Private in Deal With TowerBrook, CD&R | 10 | M&A | 3 |
| 4 | Alphabet | Google Antitrust Ruling: Landmark Decision Challenges Big Tech Dominance | 10 | Litigation | 1 |
| 5 | Nvidia | Nvidia faces two DOJ antitrust probes over market dominance | 10 | Antitrust | 1 |
| 6 | Leidos | Defense Contractor Leidos’ Data Leak Exposes Pentagon’s Documents on the Dark Web | 10 | Cybersecurity | 1 |
| 7 | Coca-Cola | Coca-Cola Could Owe $2.6 Billion in Transfer Pricing Dispute (1) | 10 | Financial | 1 |
| 8 | Facebook | Facebook parent Meta settles Texas lawsuit for $1.4B | 10 | Litigation | 1 |
| 9 | Facebook | Meta agrees to $1.4 billion settlement in Texas biometric data lawsuit over Facebook image | 10 | Litigation | 1 |
| 10 | Nokia | Amazon sues Nokia for infringing multiple AWS cloud computing patents | 9 | Litigation | 5 |

## 2. Agreement between the two LLMs

- Stories in both top-10 lists: 5 of 10
- Spearman correlation of scores across all 1090 articles: 0.44
- Same score: 25% of articles. Within 1 point: 54%

## 3. Find news about a client

| Typed | Expected | Resolved to | Articles in CSV | Returned | Result |
|---|---|---|---|---|---|
| Nvidia | Nvidia | Nvidia | 9 | 9 | pass |
| Google | Alphabet | Alphabet | 11 | 11 | pass |
| Meta | Facebook | Facebook | 9 | 9 | pass |
| JPMorgan | JPMorgan Chase | JPMorgan Chase | 8 | 8 | pass |
| Samsung | Samsung Electronics | Samsung Electronics | 7 | 7 | pass |
| Coca-Cola | Coca-Cola | Coca-Cola | 9 | 9 | pass |
| Nokia | Nokia | Nokia | 9 | 9 | pass |
| HP | HP | HP | 11 | 11 | pass |

## 4. Find news about a topic (story expected in top 5)

| Query | Result | Top result |
|---|---|---|
| antitrust investigations | hit | Big Tech's Legal Battles: Google, Apple, Meta, and Amazon Under Siege |
| patent infringement lawsuit | hit | Amazon hits Nokia with a patent lawsuit, implicating many of Nokia’s cloud offer |
| shareholder class action lawsuits | hit | DXC Investors Have Opportunity to Lead DXC Technology Company Securities Fraud L |
| cyberattack causing a service outage | hit | Microsoft Confirms New Outage Was Triggered By Cyberattack |
| private equity take-private deals | MISS | Heading for the exit: Buyout bosses prepare to sell after summer lull |
| tax court ruling | hit | Coca-Cola Owes $6 Billion to IRS, Tax Court Rules |
| mass layoffs | hit | Intel is laying off 15,000 employees |
| data privacy settlement | hit | Meta agrees to $1.4 billion settlement in Texas biometric data lawsuit over Face |

## 5. Cost and time to build the index

| | Calls | Input tokens | Output tokens | Seconds | Unscored |
|---|---|---|---|---|---|
| gpt-4.1-nano | 55 | 95,504 | 38,916 | 76.8 | 0 |
| gpt-4.1 | 84 | 143,547 | 46,814 | 74.7 | 0 |

1290 rows in, 1090 unique articles, 932 distinct stories. Embedding took 14.4s.