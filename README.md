# Project 3: AI Recommendation Logic — Tech Stack Recommender
**DecodeLabs Industrial Training Kit — AI Track (Batch 2026)**

## Goal
Build a simple recommendation system that maps user preferences (skills
and career interests) to relevant items (job roles), using content-based
similarity logic — no historical user-behavior data required.

## Approach: Content-Based Filtering with TF-IDF + Cosine Similarity
This project follows the exact pipeline outlined in the training deck's
"4-Step Ranking Pipeline":

| Step | What happens |
|---|---|
| **1. Ingestion** | Load `raw_skills.csv` (job roles + skill tags) and capture the user's skills (min. 3 required) |
| **2. Scoring** | Vector-map both into a shared TF-IDF vocabulary space, then compute Cosine Similarity between the user profile and every job role |
| **3. Sorting** | Rank all job roles descending by similarity score |
| **4. Filtering** | Truncate to the Top-N list to prevent choice overload |

## Why Content-Based Filtering (not Collaborative)?
Per the deck: collaborative filtering needs a large history of user
behavior ("users who picked X also picked Y"). Content-based filtering
works from item attributes alone, so it works immediately — including on
day one, with zero historical data, and it's naturally robust against the
**Item Cold Start** problem (new job roles can be recommended the moment
they're added to the dataset).

## Why TF-IDF instead of raw binary vectors?
A simple 1/0 overlap count treats a generic skill like `python` the same
as a highly specific one like `kubernetes`. TF-IDF weighting:
- **Term Frequency (TF):** rewards skills that appear prominently in a
  role's tag list
- **Inverse Document Frequency (IDF):** penalizes skills that appear
  across *many* roles (too generic to be distinguishing)

## Why Cosine Similarity instead of Euclidean Distance?
Euclidean distance is sensitive to vector *magnitude* — a job role with
9 tags vs. a user with 3 skills would appear "far apart" even with a
perfect directional match. Cosine similarity measures the **angle**
between vectors, so it stays invariant to how many tags either side has;
it only cares about *orientation* (which skills matter, not how many).
Since TF-IDF values are non-negative, scores land cleanly in [0, 1] —
an intuitive percentage match.

## Handling the Cold Start Problem
If a user's skills share **zero vocabulary overlap** with the dataset
(e.g. a typo, or a totally novel skill not yet in `raw_skills.csv`),
every cosine score comes back 0. Rather than showing an empty result,
the engine falls back to a **Trending** list of broadly in-demand roles
— exactly the "Bypassing the Cold Start" strategy from the deck.

## Project Structure
```
TechStackRecommender/
├── tech_stack_recommender.py   # Full pipeline (run this)
├── notebook.ipynb               # Same pipeline, notebook format
├── requirements.txt             # pip dependencies
├── README.md                     # This file
├── data/
│   └── raw_skills.csv             # 20 job roles + their skill tags
└── outputs/                       # Generated on each run
    ├── top_matches.png             # Bar chart of Top-N recommendations
    ├── full_ranking.png            # All 20 roles ranked by similarity
    └── recommendations.json        # Machine-readable results
```

## How to Run

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run with specific skills:
   ```bash
   python tech_stack_recommender.py --skills python "cloud computing" automation
   python tech_stack_recommender.py --skills java sql apis --top-n 5
   ```

3. Or run interactively (prompts for comma-separated skills):
   ```bash
   python tech_stack_recommender.py
   ```

4. Optional flags:
   ```bash
   python tech_stack_recommender.py --skills ... --no-plots   # skip PNGs
   ```

Or open `notebook.ipynb` in Jupyter for a cell-by-cell walkthrough.

## Key Design Decisions

- **Shared vocabulary normalization**: user input is lowercased and
  multi-word phrases are hyphenated (`"Cloud Computing"` → `cloud-computing`)
  to match the dataset's tagging convention — this is the exact failure
  mode the deck warns about ("Bridging the Language Barrier").
- **Minimum 3 skills enforced**: per the deck's Step 1 requirement,
  ensuring sufficient data density for accurate matching. Fewer than 3
  raises a clear `ValueError`.
- **Cold-start fallback instead of silent failure**: a zero-overlap
  query still returns a useful (if generic) result rather than nothing.
- **Matched-skill transparency**: each recommendation shows exactly
  which of the user's skills drove the match, so results are explainable
  rather than a black box.

## Sample Results

Input: `["Python", "Cloud Computing", "Automation"]`

| Rank | Job Role | Score | Matched On |
|---|---|---|---|
| 1 | Cloud Architect | 0.2927 | cloud-computing |
| 2 | QA/Test Automation Engineer | 0.2596 | automation, python |
| 3 | Systems Administrator | 0.1685 | automation |

Input: `["Java", "SQL", "APIs"]`

| Rank | Job Role | Score | Matched On |
|---|---|---|---|
| 1 | Backend Developer | 0.5572 | apis, java, sql |
| 2 | Full Stack Developer | 0.3267 | apis, sql |
| 3 | Mobile App Developer | 0.1991 | java |

A perfect 3-for-3 skill match (Backend Developer) scores highest, and
partial overlaps rank sensibly below it — exactly the "Digital
Matchmaker" behavior the recommendation engine is designed for.

## Next Steps (per the deck's "Graduation to Commercial-Grade Logic")
This project mastered content-based filtering with a static dataset.
The natural next step is collaborative filtering (using real user
interaction history) or hybrid approaches that blend both — the same
principles (feature extraction + similarity math) remain the bedrock
either way.
