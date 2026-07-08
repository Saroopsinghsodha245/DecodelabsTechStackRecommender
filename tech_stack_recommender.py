"""
Project 3: AI Recommendation Logic — Tech Stack Recommender
DecodeLabs Industrial Training Kit (AI Track)

Goal
----
Build a content-based recommendation engine that maps a user's raw
skills / career interests to the most relevant job roles, using pure
similarity logic (no historical user-behavior data required).

This reproduces the exact pipeline from the training deck:

    INPUT   : Capture user state (>=3 skills) + ingest the job-role
              dataset (raw_skills.csv).
    PROCESS : Vector-map skills into a shared vocabulary space using
              TF-IDF weighting, then score every job role against the
              user profile using Cosine Similarity (not Euclidean —
              see "Why Euclidean Distance Fails at Scale").
    OUTPUT  : Sort scores descending, filter to the Top-N list.

The 4-step ranking pipeline ("Ingestion -> Scoring -> Sorting ->
Filtering") from the deck maps directly onto the functions below.

Also handles the "Cold Start" problem: if a user's skills don't
overlap with the vocabulary at all (all-zero vector), the engine
falls back to a "Trending" list instead of returning nothing.

Usage
-----
    python tech_stack_recommender.py --skills python cloud automation
    python tech_stack_recommender.py --skills java sql apis --top-n 5
    python tech_stack_recommender.py                      # interactive prompt
    python tech_stack_recommender.py --skills foo bar baz --no-plots

Author: Saroop (DecodeLabs AI Track — Project 3)
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA_PATH = Path(__file__).parent / "data" / "raw_skills.csv"
OUTPUT_DIR = Path(__file__).parent / "outputs"

MIN_REQUIRED_SKILLS = 3  # per the deck: "must accept a minimum of three user inputs"


@dataclass
class Recommendation:
    job_role: str
    score: float
    description: str
    matched_skills: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Step 1: Ingestion — load the dataset + capture user state
# --------------------------------------------------------------------------- #
def load_job_dataset(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load job roles + their skill tags (the 'items' in our engine)."""
    df = pd.read_csv(path)
    df["skills"] = df["skills"].astype(str).str.lower().str.strip()
    return df


def normalize_skills(raw_skills: list[str]) -> list[str]:
    """Normalize user-provided skills so they map to the SAME vocabulary
    as the item dataset (per 'Bridging the Language Barrier' slide —
    naming discrepancies like 'Web Design' vs 'Frontend Development'
    break the similarity math). We lowercase and replace spaces with
    hyphens so multi-word tags match the dataset's convention."""
    normalized = []
    for skill in raw_skills:
        cleaned = re.sub(r"[^a-zA-Z0-9\s-]", "", skill).strip().lower()
        cleaned = re.sub(r"\s+", "-", cleaned)
        if cleaned:
            normalized.append(cleaned)
    return normalized


def validate_skills(skills: list[str]) -> list[str]:
    if len(skills) < MIN_REQUIRED_SKILLS:
        raise ValueError(
            f"At least {MIN_REQUIRED_SKILLS} skills are required for accurate "
            f"matching (got {len(skills)}: {skills}). Add more skills and try again."
        )
    return skills


# --------------------------------------------------------------------------- #
# Step 2: Scoring — TF-IDF vector mapping + Cosine Similarity
# --------------------------------------------------------------------------- #
def build_tfidf_matrix(df: pd.DataFrame) -> tuple[TfidfVectorizer, np.ndarray]:
    """Fit a TF-IDF vectorizer over the job-role skill documents.

    TF-IDF (per the deck) rewards specific, descriptive tags (e.g.
    'kubernetes') and penalizes generic ones that appear across many
    roles (e.g. 'python' appearing in half the dataset gets down-
    weighted relative to a rare, highly descriptive tag).
    """
    vectorizer = TfidfVectorizer(token_pattern=r"[a-zA-Z0-9\-]+")
    item_matrix = vectorizer.fit_transform(df["skills"])
    return vectorizer, item_matrix


def score_items(vectorizer: TfidfVectorizer, item_matrix, user_skills: list[str]) -> np.ndarray:
    """Transform the user's skills into the SAME TF-IDF vocabulary
    space, then compute Cosine Similarity against every job role.

    Cosine similarity is used instead of Euclidean distance because it
    measures the ANGLE between vectors (orientation of preferences),
    making it invariant to how many skills the user listed vs. how
    many tags a job role has ('Why Euclidean Distance Fails at Scale').
    """
    user_doc = " ".join(user_skills)
    user_vector = vectorizer.transform([user_doc])
    scores = cosine_similarity(user_vector, item_matrix).flatten()
    return scores


# --------------------------------------------------------------------------- #
# Steps 3 & 4: Sorting + Filtering -> Top-N list
# --------------------------------------------------------------------------- #
def rank_and_filter(
    df: pd.DataFrame, scores: np.ndarray, user_skills: list[str], top_n: int = 3
) -> list[Recommendation]:
    """Sort descending by score, then truncate to the Top-N list to
    prevent 'choice overload' (per the deck's Step 4: Filtering)."""
    ranked_indices = np.argsort(scores)[::-1]

    recommendations = []
    for idx in ranked_indices[:top_n]:
        role = df.iloc[idx]
        role_skills = set(role["skills"].split())
        matched = sorted(set(user_skills) & role_skills)
        recommendations.append(
            Recommendation(
                job_role=role["job_role"],
                score=float(scores[idx]),
                description=role["description"],
                matched_skills=matched,
            )
        )
    return recommendations


def is_cold_start(scores: np.ndarray) -> bool:
    """The 'Achilles Heel': if the user's skill vector shares NO
    vocabulary overlap with any item, every cosine score is 0."""
    return bool(np.allclose(scores, 0.0))


def trending_fallback(df: pd.DataFrame, top_n: int = 3) -> list[Recommendation]:
    """Cold-start bypass: default to a 'Trending' list (here: the
    roles with the broadest, most in-demand skill sets) instead of
    returning nothing, per the 'Bypassing the Cold Start' slide."""
    # Proxy for "trending": roles whose skill tags overlap most with
    # the rest of the dataset (i.e. broadly applicable, high-demand skills)
    trending_order = ["Data Scientist", "Full Stack Developer", "DevOps Engineer",
                       "Cloud Architect", "Machine Learning Engineer"]
    fallback_roles = [r for r in trending_order if r in df["job_role"].values][:top_n]
    recs = []
    for role_name in fallback_roles:
        role = df[df["job_role"] == role_name].iloc[0]
        recs.append(Recommendation(job_role=role_name, score=0.0, description=role["description"]))
    return recs


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def plot_top_matches(recommendations: list[Recommendation], out_path: Path) -> None:
    names = [r.job_role for r in recommendations][::-1]
    scores = [r.score for r in recommendations][::-1]

    plt.figure(figsize=(8, 5))
    bars = plt.barh(names, scores, color="#2b6cb0")
    plt.xlabel("Cosine Similarity Score")
    plt.title("Top Matches: Tech Stack Recommender")
    plt.xlim(0, 1)
    for bar, score in zip(bars, scores):
        plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                  f"{score:.2f}", va="center")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_full_ranking(df: pd.DataFrame, scores: np.ndarray, out_path: Path) -> None:
    order = np.argsort(scores)[::-1]
    names = df.iloc[order]["job_role"].tolist()
    sorted_scores = scores[order]

    plt.figure(figsize=(9, 7))
    colors = ["#2b6cb0" if s > 0 else "#cbd5e0" for s in sorted_scores]
    plt.barh(names[::-1], sorted_scores[::-1], color=colors[::-1])
    plt.xlabel("Cosine Similarity Score")
    plt.title("All Job Roles Ranked by Similarity")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def run_pipeline(
    raw_skills: list[str],
    top_n: int = 3,
    make_plots: bool = True,
    output_dir: Path = OUTPUT_DIR,
    data_path: Path = DATA_PATH,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- INPUT: Ingestion ----
    print("=" * 60)
    print("STEP 1: INGESTION")
    print("=" * 60)
    df = load_job_dataset(data_path)
    print(f"Loaded {len(df)} job roles from {data_path.name}")

    user_skills = normalize_skills(raw_skills)
    user_skills = validate_skills(user_skills)
    print(f"User skills (normalized): {user_skills}")

    # ---- PROCESS: Scoring ----
    print("\n" + "=" * 60)
    print("STEP 2: SCORING (TF-IDF + Cosine Similarity)")
    print("=" * 60)
    vectorizer, item_matrix = build_tfidf_matrix(df)
    scores = score_items(vectorizer, item_matrix, user_skills)
    print(f"Scored {len(scores)} job roles against user profile")

    cold_start = is_cold_start(scores)
    if cold_start:
        print("\n[!] COLD START DETECTED: no vocabulary overlap with dataset.")
        print("    Falling back to trending roles instead of an empty result.")

    # ---- PROCESS: Sorting + OUTPUT: Filtering ----
    print("\n" + "=" * 60)
    print("STEPS 3 & 4: SORTING + FILTERING -> Top-N")
    print("=" * 60)
    if cold_start:
        recommendations = trending_fallback(df, top_n=top_n)
    else:
        recommendations = rank_and_filter(df, scores, user_skills, top_n=top_n)

    for i, rec in enumerate(recommendations, start=1):
        print(f"\n#{i}: {rec.job_role}  (score={rec.score:.4f})")
        print(f"    {rec.description}")
        if rec.matched_skills:
            print(f"    Matched on: {', '.join(rec.matched_skills)}")

    if make_plots and not cold_start:
        plot_top_matches(recommendations, output_dir / "top_matches.png")
        plot_full_ranking(df, scores, output_dir / "full_ranking.png")
        print(f"\nPlots saved to: {output_dir.resolve()}")

    result = {
        "user_skills": user_skills,
        "cold_start": cold_start,
        "recommendations": [
            {
                "job_role": r.job_role,
                "score": r.score,
                "description": r.description,
                "matched_skills": r.matched_skills,
            }
            for r in recommendations
        ],
    }

    with open(output_dir / "recommendations.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"Results saved to: {(output_dir / 'recommendations.json').resolve()}")

    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Project 3: AI Recommendation Logic — Tech Stack Recommender")
    parser.add_argument("--skills", nargs="+", default=None, help="List of skills, e.g. --skills python cloud automation")
    parser.add_argument("--top-n", type=int, default=3, help="Number of recommendations to return (default: 3)")
    parser.add_argument("--no-plots", action="store_true", help="Skip generating PNG plots")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    skills = args.skills
    if skills is None:
        raw = input(
            f"Enter at least {MIN_REQUIRED_SKILLS} skills or interests, "
            f"separated by commas (e.g. Python, Cloud Computing, Automation): "
        )
        skills = [s.strip() for s in raw.split(",") if s.strip()]

    run_pipeline(raw_skills=skills, top_n=args.top_n, make_plots=not args.no_plots)
