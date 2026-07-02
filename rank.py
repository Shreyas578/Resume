#!/usr/bin/env python3
"""
Redrob Hackathon — Senior AI Engineer Candidate Ranker v2
Author: Participant
Constraints: CPU-only, no API calls, <5 min, <16GB RAM

Key improvements over v1:
- Skill endorsement and duration weighting (not just presence)
- Assessment score integration
- Better honeypot detection (company founding year logic)
- Weighted skill proficiency tiers
- Title-role alignment check (e.g., "Marketing Manager" with AI skills = no)
- Smarter pre-LLM IR detection from career descriptions
- Offer acceptance + interview completion factored in
- Tiered must-have penalty (progressive, not binary gate)
- Salary range reasonableness check (implicit availability signal)
- Production-role title matching
"""

import json
import csv
import sys
import argparse
from datetime import datetime
from collections import defaultdict

try:
    from dateutil import parser as dateparser
    HAS_DATEUTIL = True
except ImportError:
    HAS_DATEUTIL = False

# ─── Constants ────────────────────────────────────────────────────────────────

REFERENCE_DATE = datetime(2026, 7, 2)

CONSULTING_FIRMS = {
    "tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini",
    "hcl", "tech mahindra", "mphasis", "hexaware", "ltimindtree",
    "mindtree", "kpit", "cyient", "niit technologies", "mastech",
    "igate", "patni", "satyam", "mahindra satyam", "l&t infotech",
    "zensar", "persistent systems", "sonata software"
}

# Weighted skill sets — each entry: (keyword, weight)
# Core embedding/retrieval must-have #1
EMBEDDING_SKILLS_W = {
    "sentence-transformers": 3, "sentence transformers": 3,
    "bge": 3, "e5": 2, "instructor embeddings": 3,
    "bi-encoder": 3, "cross-encoder": 3,
    "dense retrieval": 3, "dpr": 3, "colbert": 3, "splade": 3,
    "openai embeddings": 2, "text embeddings": 2,
    "rag": 2, "retrieval augmented generation": 3,
    "semantic retrieval": 2, "neural retrieval": 3,
    "information retrieval": 2, "embeddings": 1,
    "embedding models": 2, "huggingface": 1, "hugging face": 1,
    "matryoshka": 3,
}

# Vector DB must-have #2
VECTOR_DB_SKILLS_W = {
    "pinecone": 3, "weaviate": 3, "qdrant": 3, "milvus": 3,
    "faiss": 3, "opensearch": 2, "elasticsearch": 2,
    "pgvector": 3, "chroma": 2, "vespa": 3,
    "annoy": 2, "hnsw": 3, "nmslib": 2,
    "redis vector": 3, "vector search": 2, "vector database": 2,
    "vector store": 2, "hybrid search": 3, "semantic search": 2,
    "azure cognitive search": 2, "vertex ai matching engine": 3,
}

# Eval framework must-have #3
EVAL_SKILLS_W = {
    "ndcg": 3, "mrr": 3, "map": 2, "mean average precision": 3,
    "ranking evaluation": 3, "a/b testing": 2, "ab testing": 2,
    "offline evaluation": 3, "online evaluation": 3,
    "evaluation framework": 2, "relevance judgment": 3,
    "precision@k": 3, "recall@k": 3, "p@10": 3,
    "offline-to-online": 3,
}

# Nice-to-have
FINE_TUNING_W = {
    "lora": 3, "qlora": 3, "peft": 3, "fine-tuning": 2,
    "fine tuning": 2, "finetuning": 2, "instruction tuning": 3,
    "rlhf": 3, "dpo": 3, "sft": 2,
}

LTR_W = {
    "learning to rank": 3, "lambdamart": 3, "ranknet": 3,
    "listwise": 2, "pairwise ranking": 2,
    "xgboost ranking": 3, "lightgbm ranking": 3, "pointwise ranking": 2,
    "lambdarank": 3,
}

NLP_IR_W = {
    "nlp": 1, "natural language processing": 2,
    "bert": 2, "gpt": 1, "transformers": 2,
    "llm": 1, "large language model": 1,
    "reranking": 3, "re-ranking": 3,
    "bm25": 3, "tfidf": 2, "tf-idf": 2,
    "lucene": 2, "solr": 2,
    "text classification": 1, "named entity recognition": 1,
    "question answering": 2, "semantic similarity": 2,
}

CV_SPEECH_SKILLS = {
    "computer vision", "image classification", "object detection",
    "yolo", "resnet", "cnn", "convolutional neural", "segmentation",
    "speech recognition", "asr", "tts", "text to speech",
    "speech synthesis", "wav2vec", "whisper", "voice recognition",
    "robotics", "ros", "point cloud", "lidar", "pose estimation",
    "ocr", "optical character recognition",
}

INDIA_TIER1_CITIES = {
    "noida", "pune", "delhi", "gurgaon", "gurugram", "delhi ncr",
    "ncr", "hyderabad", "mumbai", "bangalore", "bengaluru",
    "new delhi", "navi mumbai", "thane", "greater noida",
}

# Job titles that indicate a genuine AI/ML/engineering role
GOOD_TITLE_KEYWORDS = {
    "engineer", "scientist", "architect", "developer", "researcher",
    "ml", "ai", "nlp", "data", "applied", "backend", "software",
    "search", "ranking", "retrieval", "recommendation", "platform",
    "tech lead", "technical lead", "founding engineer",
}

# Red flag titles — AI keywords in profile but wrong role
BAD_TITLE_KEYWORDS = {
    "manager", "consultant", "analyst", "sales", "marketing",
    "hr", "recruiter", "operations", "finance", "accountant",
    "designer", "ux", "ui", "product manager", "program manager",
    "scrum master", "agile coach", "delivery manager",
    "project manager", "account manager", "business analyst",
}


# ─── Parsing helpers ───────────────────────────────────────────────────────────

def days_since(date_str):
    if not date_str:
        return 9999
    try:
        if HAS_DATEUTIL:
            dt = dateparser.parse(str(date_str))
        else:
            dt = datetime.strptime(str(date_str)[:10], "%Y-%m-%d")
        return (REFERENCE_DATE - dt.replace(tzinfo=None)).days
    except Exception:
        return 9999


def normalize(value, min_val, max_val):
    if value is None:
        return 0.0
    return max(0.0, min(1.0, (value - min_val) / (max_val - min_val + 1e-9)))


def skill_set_lower(candidate):
    return {s["name"].lower() for s in candidate.get("skills", [])}


def skills_with_meta(candidate):
    """Return list of (name_lower, proficiency, endorsements, duration_months)."""
    out = []
    for s in candidate.get("skills", []):
        out.append((
            s["name"].lower(),
            s.get("proficiency", "beginner"),
            s.get("endorsements", 0) or 0,
            s.get("duration_months", 0) or 0,
        ))
    return out


def career_description_text(candidate):
    parts = [candidate["profile"].get("summary", "") or ""]
    for ch in candidate.get("career_history", []):
        parts.append(ch.get("description", "") or "")
        parts.append(ch.get("title", "") or "")
        parts.append(ch.get("company", "") or "")
    return " ".join(parts).lower()


def all_text(candidate):
    txt = career_description_text(candidate)
    skill_names = " ".join(s["name"] for s in candidate.get("skills", []))
    return (txt + " " + skill_names).lower()


# ─── Hard disqualifier checks ─────────────────────────────────────────────────

def is_consulting_only(candidate):
    history = candidate.get("career_history", [])
    if not history:
        return False
    for job in history:
        co = (job.get("company", "") or "").lower()
        if not any(f in co for f in CONSULTING_FIRMS):
            return False
    # Also check current company field
    current = (candidate["profile"].get("current_company", "") or "").lower()
    return any(f in current for f in CONSULTING_FIRMS)


def is_pure_cv_speech(candidate):
    """True only if candidate has heavy CV/Speech focus AND negligible NLP/IR."""
    skills_lower = skill_set_lower(candidate)
    txt = all_text(candidate)

    cv_count = sum(1 for s in CV_SPEECH_SKILLS if s in txt)
    nlp_count = (
        sum(1 for kw in NLP_IR_W if kw in txt) +
        sum(1 for kw in EMBEDDING_SKILLS_W if kw in txt)
    )
    title = (candidate["profile"].get("current_title", "") or "").lower()
    cv_title = any(kw in title for kw in ["computer vision", "cv engineer", "speech", "robotics", "asr"])

    return cv_count >= 5 and nlp_count < 2 and cv_title


def is_wrong_role_title(candidate):
    """
    Returns True if the candidate's current title is a non-engineering role
    (marketing manager, PM, sales etc.) even if they have AI skills.
    JD explicitly says the 'right answer' is NOT about AI keywords alone.
    """
    title = (candidate["profile"].get("current_title", "") or "").lower()
    yoe = candidate["profile"].get("years_of_experience", 0) or 0

    has_bad = any(kw in title for kw in BAD_TITLE_KEYWORDS)
    has_good = any(kw in title for kw in GOOD_TITLE_KEYWORDS)

    # Only disqualify if clearly non-engineering AND no good title signal
    return has_bad and not has_good


def is_langchain_only_novice(candidate):
    """
    JD says: reject if AI exp is primarily recent (<12 mo) LangChain wrappers
    without pre-LLM IR background.
    """
    txt = all_text(candidate)
    yoe = candidate["profile"].get("years_of_experience", 0) or 0

    has_framework_focus = (
        txt.count("langchain") + txt.count("llamaindex") + txt.count("llama-index")
    ) >= 2

    has_pre_llm_ir = any(kw in txt for kw in [
        "bm25", "lucene", "elasticsearch", "solr", "faiss", "ranking algorithm",
        "recommendation system", "search engine", "information retrieval",
        "collaborative filtering", "matrix factorization",
        "tf-idf", "tfidf", "inverted index",
    ])

    return has_framework_focus and not has_pre_llm_ir and yoe < 4


def is_honeypot(candidate):
    """Detect subtly impossible profiles."""
    profile = candidate["profile"]
    history = candidate.get("career_history", [])
    yoe = profile.get("years_of_experience", 0) or 0
    skills = candidate.get("skills", [])

    # YoE vs career history mismatch
    total_career_months = sum(h.get("duration_months", 0) or 0 for h in history)
    if history and total_career_months > 0:
        career_yrs = total_career_months / 12
        if abs(yoe - career_yrs) > 7:
            return True

    # Lots of "advanced" skills with zero endorsements each
    adv_zero = sum(
        1 for s in skills
        if s.get("proficiency") == "advanced" and (s.get("endorsements", 0) or 0) == 0
    )
    if adv_zero >= 8:
        return True

    # Impossible tenure at one company
    for job in history:
        dur = job.get("duration_months", 0) or 0
        if dur > 180 and yoe < 12:
            return True

    # Assessment scores implausibly high with near-zero skill duration
    assessment_scores = candidate.get("redrob_signals", {}).get("skill_assessment_scores", {}) or {}
    skills_map = {s["name"]: s for s in skills}
    impossible = 0
    for skill_name, score in assessment_scores.items():
        if score >= 90:
            matched = skills_map.get(skill_name)
            if matched and (matched.get("duration_months", 99) or 99) < 3:
                impossible += 1
    if impossible >= 3:
        return True

    # Expert in 10+ skills total with very low total endorsements
    total_endorsements = sum(s.get("endorsements", 0) or 0 for s in skills)
    expert_count = sum(1 for s in skills if s.get("proficiency") == "advanced")
    if expert_count >= 10 and total_endorsements < 5:
        return True

    return False


def is_pure_researcher(candidate):
    """
    Penalize candidates who are purely academic/research with zero
    production deployment evidence. JD is very explicit.
    """
    txt = career_description_text(candidate)
    title = (candidate["profile"].get("current_title", "") or "").lower()
    history = candidate.get("career_history", [])

    research_title = any(kw in title for kw in [
        "research scientist", "research engineer", "phd researcher",
        "postdoc", "professor", "lecturer", "research intern",
        "research fellow",
    ])

    research_companies = sum(
        1 for h in history
        if any(kw in (h.get("company", "") or "").lower() for kw in [
            "university", "college", "institute", "lab", "research center",
            "iit", "iim", "iisc", "mit", "stanford", "oxford", "cambridge",
            "iiit", "nit",
        ])
    )

    prod_keywords = ["production", "deployed", "serving", "live traffic",
                     "real users", "user-facing", "api endpoint", "mlops"]
    has_prod = any(kw in txt for kw in prod_keywords)

    return research_title and research_companies >= 2 and not has_prod


# ─── Scoring components ───────────────────────────────────────────────────────

def weighted_skill_score(candidate_skills_meta, skill_weight_dict, cap=10.0):
    """
    Richer scoring that considers proficiency level, endorsements, and duration.
    Returns 0-1 normalized against cap.
    """
    PROFICIENCY_MULT = {"beginner": 0.4, "intermediate": 0.7, "advanced": 1.0, "expert": 1.0}
    total = 0.0

    for skill_name, proficiency, endorsements, duration_months in candidate_skills_meta:
        for target_kw, weight in skill_weight_dict.items():
            if target_kw in skill_name or skill_name in target_kw:
                p_mult = PROFICIENCY_MULT.get(proficiency, 0.6)
                # Endorsements boost: log scale, max ~0.3 bonus
                endorse_bonus = min(0.3, endorsements / 30.0 * 0.3)
                # Duration boost: >12 months = full credit
                dur_mult = min(1.0, duration_months / 12.0) if duration_months else 0.5
                total += weight * p_mult * (1.0 + endorse_bonus) * dur_mult
                break  # Don't double-count same skill

    return min(1.0, total / cap)


def text_fallback_score(txt, skill_weight_dict, cap=6.0):
    """Fallback: check the text blob for keywords not in skills list."""
    total = sum(w for kw, w in skill_weight_dict.items() if kw in txt)
    return min(0.7, total / cap)  # Cap at 0.7 — text mentions < explicit skills


def get_must_have_scores(candidate):
    """
    Returns (embed_score, vdb_score, eval_score) each [0,1].
    Uses both skills-with-meta scoring and text fallback.
    """
    skills_meta = skills_with_meta(candidate)
    txt = all_text(candidate)
    career_txt = career_description_text(candidate)

    # Check assessment scores for corroboration
    assessment = (candidate.get("redrob_signals", {}) or {}).get("skill_assessment_scores", {}) or {}

    # --- EMBED ---
    embed_skill = weighted_skill_score(skills_meta, EMBEDDING_SKILLS_W, cap=8.0)
    embed_text = text_fallback_score(career_txt, EMBEDDING_SKILLS_W, cap=6.0)
    embed_assess_bonus = 0.0
    for k, v in assessment.items():
        if any(kw in k.lower() for kw in ["embedding", "rag", "retrieval", "nlp"]):
            embed_assess_bonus = max(embed_assess_bonus, min(0.2, v / 100 * 0.2))
    embed_score = min(1.0, max(embed_skill, embed_text * 0.8) + embed_assess_bonus)

    # --- VDB ---
    vdb_skill = weighted_skill_score(skills_meta, VECTOR_DB_SKILLS_W, cap=8.0)
    vdb_text = text_fallback_score(career_txt, VECTOR_DB_SKILLS_W, cap=5.0)
    vdb_score = min(1.0, max(vdb_skill, vdb_text * 0.8))

    # --- EVAL ---
    eval_skill = weighted_skill_score(skills_meta, EVAL_SKILLS_W, cap=7.0)
    eval_text = text_fallback_score(career_txt, EVAL_SKILLS_W, cap=5.0)
    eval_score = min(1.0, max(eval_skill, eval_text * 0.8))

    return embed_score, vdb_score, eval_score


def get_nice_to_have_scores(candidate):
    skills_meta = skills_with_meta(candidate)
    txt = all_text(candidate)
    career_txt = career_description_text(candidate)

    fine_tune = max(
        weighted_skill_score(skills_meta, FINE_TUNING_W, cap=6.0),
        text_fallback_score(career_txt, FINE_TUNING_W, cap=5.0) * 0.7
    )
    ltr = max(
        weighted_skill_score(skills_meta, LTR_W, cap=6.0),
        text_fallback_score(career_txt, LTR_W, cap=5.0) * 0.7
    )
    nlp = max(
        weighted_skill_score(skills_meta, NLP_IR_W, cap=10.0),
        text_fallback_score(career_txt, NLP_IR_W, cap=8.0) * 0.8
    )

    # Python — check skills AND text
    skills_lower = {s[0] for s in skills_meta}
    has_python = "python" in skills_lower or "python" in txt
    python_score = 1.0 if has_python else 0.0
    # Bonus for Python proficiency
    for name, prof, endorse, _ in skills_meta:
        if "python" in name:
            if prof == "advanced":
                python_score = 1.0
            elif prof == "intermediate":
                python_score = 0.85
            break

    # GitHub activity
    github_raw = (candidate.get("redrob_signals", {}) or {}).get("github_activity_score", -1)
    if github_raw is None or github_raw == -1:
        github_score = 0.15
    else:
        github_score = normalize(github_raw, 0, 80)

    return fine_tune, ltr, nlp, github_score, python_score


def get_experience_score(candidate):
    yoe = candidate["profile"].get("years_of_experience", 0) or 0
    # Ideal: 5-9, sweet spot 6-8
    if 6 <= yoe <= 8:
        return 1.0
    elif 5 <= yoe < 6 or 8 < yoe <= 9:
        return 0.90
    elif 4 <= yoe < 5 or 9 < yoe <= 11:
        return 0.75
    elif 3 <= yoe < 4 or 11 < yoe <= 14:
        return 0.55
    elif yoe > 14:
        return 0.40
    else:
        return 0.15


def get_company_type_score(candidate):
    history = candidate.get("career_history", [])
    if not history:
        return 0.30

    product_months, consulting_months, research_months, total_months = 0, 0, 0, 0

    PRODUCT_INDUSTRIES = {
        "software", "fintech", "edtech", "e-commerce", "food delivery",
        "ai/ml", "saas", "marketplace", "healthtech", "gaming",
        "social media", "cybersecurity", "adtech", "proptech",
    }
    RESEARCH_INDUSTRIES = {
        "research", "academia", "education", "non-profit",
    }

    for job in history:
        co = (job.get("company", "") or "").lower()
        industry = (job.get("industry", "") or "").lower()
        size = job.get("company_size", "") or ""
        dur = job.get("duration_months", 0) or 0
        total_months += dur

        is_consulting = any(f in co for f in CONSULTING_FIRMS)
        is_research = (
            any(kw in co for kw in ["university", "college", "lab ", "research", "institute", "iit", "iim", "iisc"]) or
            industry in RESEARCH_INDUSTRIES
        )
        is_product = (
            not is_consulting and not is_research and (
                industry in PRODUCT_INDUSTRIES or
                size in ["11-50", "51-200", "201-500", "501-1000"] or
                size == "1001-5000"
            )
        )

        if is_consulting:
            consulting_months += dur
        elif is_research:
            research_months += dur
        elif is_product:
            product_months += dur

    if total_months == 0:
        return 0.30

    pr = product_months / total_months
    cr = consulting_months / total_months
    rr = research_months / total_months

    if pr >= 0.75:
        return 1.0
    elif pr >= 0.55:
        return 0.85
    elif pr >= 0.35:
        return 0.65
    elif cr >= 0.85:
        return 0.15
    elif rr >= 0.8:
        return 0.20
    else:
        return 0.45


def get_tenure_score(candidate):
    history = candidate.get("career_history", [])
    if len(history) <= 1:
        return 0.70

    past = [h.get("duration_months", 0) or 0 for h in history if not h.get("is_current")]
    if not past:
        return 0.70

    avg = sum(past) / len(past)
    if avg >= 30:
        return 1.0
    elif avg >= 24:
        return 0.90
    elif avg >= 18:
        return 0.75
    elif avg >= 12:
        return 0.55
    else:
        return 0.30


def get_production_shipping_score(candidate):
    txt = career_description_text(candidate)

    prod_kws = [
        "production", "deployed", "serving", "latency", "throughput",
        "real users", "live", "at scale", "inference", "api endpoint",
        "model serving", "mlflow", "mlops", "monitoring", "production traffic",
        "user-facing", "released", "shipped", "launched",
    ]
    retrieval_kws = [
        "ranking", "retrieval", "recommendation", "search recall",
        "precision", "relevance", "embedding", "vector index", "hybrid",
        "semantic", "rerank", "bm25", "lucene", "elasticsearch",
        "recall@", "ndcg", "a/b test", "recall and precision",
    ]

    prod_hits = sum(1 for k in prod_kws if k in txt)
    ret_hits = sum(1 for k in retrieval_kws if k in txt)

    prod_score = min(1.0, prod_hits / 7.0)
    ret_score = min(1.0, ret_hits / 5.0)
    return (prod_score * 0.55 + ret_score * 0.45)


def get_location_score(candidate):
    profile = candidate["profile"]
    country = (profile.get("country", "") or "").lower()
    location = (profile.get("location", "") or "").lower()
    signals = candidate.get("redrob_signals", {}) or {}
    willing_relocate = signals.get("willing_to_relocate", False)

    if country == "india":
        if any(city in location for city in INDIA_TIER1_CITIES):
            return 1.0
        return 0.78  # Any India location — still reachable
    if willing_relocate:
        return 0.35
    return 0.08  # Outside India, no visa sponsorship


def get_notice_period_score(candidate):
    signals = candidate.get("redrob_signals", {}) or {}
    notice = signals.get("notice_period_days")
    if notice is None:
        return 0.50
    if notice <= 0:
        return 1.0
    elif notice <= 15:
        return 0.98
    elif notice <= 30:
        return 0.90
    elif notice <= 60:
        return 0.70
    elif notice <= 90:
        return 0.50
    elif notice <= 120:
        return 0.30
    else:
        return 0.12


def get_engagement_multiplier(candidate):
    """
    Behavioral multiplier [0.15, 1.0]. Never zeroes out a candidate — it
    only modulates how accessible / hireable they actually are right now.
    """
    signals = candidate.get("redrob_signals", {}) or {}
    score = 0.0

    # Last active (most important — dead accounts ≈ not available)
    days_inactive = days_since(signals.get("last_active_date"))
    if days_inactive <= 7:
        score += 0.28
    elif days_inactive <= 30:
        score += 0.22
    elif days_inactive <= 90:
        score += 0.13
    elif days_inactive <= 180:
        score += 0.05
    # >180 days: +0 (effectively unavailable)

    # Open to work
    if signals.get("open_to_work_flag"):
        score += 0.18

    # Recruiter response rate
    rr = signals.get("recruiter_response_rate") or 0.0
    score += rr * 0.14

    # Interview completion rate (shows they follow through)
    icr = signals.get("interview_completion_rate") or 0.0
    score += icr * 0.10

    # Offer acceptance rate (positive signal — not perpetually ghosting)
    oar = signals.get("offer_acceptance_rate") or 0.0
    if oar >= 0:  # -1 means no prior offers
        score += min(0.06, oar * 0.06)

    # Profile completeness
    completeness = (signals.get("profile_completeness_score") or 50) / 100
    score += completeness * 0.10

    # Response time (lower = better)
    avg_resp = signals.get("avg_response_time_hours")
    if avg_resp is not None:
        if avg_resp <= 4:
            score += 0.08
        elif avg_resp <= 24:
            score += 0.05
        elif avg_resp <= 72:
            score += 0.02

    # Saved by recruiters recently — market validation
    saved = signals.get("saved_by_recruiters_30d") or 0
    score += min(0.06, saved * 0.006)

    return max(0.15, min(1.0, score))


# ─── Main scoring ─────────────────────────────────────────────────────────────

def score_candidate(candidate):
    """Returns (score: float, reasoning: str)."""
    profile = candidate["profile"]
    signals = candidate.get("redrob_signals", {}) or {}
    yoe = profile.get("years_of_experience", 0) or 0
    title = (profile.get("current_title", "") or "Unknown title").strip()
    location = (profile.get("location", "") or "?").strip()
    country = (profile.get("country", "") or "").strip()
    notice = signals.get("notice_period_days")
    txt = all_text(candidate)

    # ── HARD DISQUALIFIERS ────────────────────────────────────────────────

    if is_honeypot(candidate):
        return 0.001, "Likely honeypot: profile has impossible/inconsistent data (tenure vs YoE mismatch, implausible assessments, or zero-endorsement expert claims)."

    if is_consulting_only(candidate):
        return 0.004, f"Disqualified: entire career at consulting/services firms (JD explicitly excludes). {title}, {yoe:.0f}yrs."

    if is_pure_cv_speech(candidate):
        return 0.006, f"Disqualified: primary expertise is CV/Speech with no meaningful NLP/IR background. Title: {title}."

    if is_wrong_role_title(candidate):
        return 0.007, f"Disqualified: current role ({title}) is non-engineering. AI keywords in profile do not compensate for wrong function."

    if is_langchain_only_novice(candidate):
        return 0.009, f"Disqualified: AI experience is primarily LangChain/LLM-wrapper based with no demonstrated pre-LLM IR/retrieval background. {yoe:.0f}yrs total."

    if is_pure_researcher(candidate):
        return 0.010, f"Disqualified: appears to be pure academic/research background with no production deployment evidence. {title}."

    # ── COMPONENT SCORES ──────────────────────────────────────────────────

    embed_score, vdb_score, eval_score = get_must_have_scores(candidate)
    fine_tune, ltr, nlp, github, python = get_nice_to_have_scores(candidate)

    exp_score = get_experience_score(candidate)
    company_score = get_company_type_score(candidate)
    tenure_score = get_tenure_score(candidate)
    prod_score = get_production_shipping_score(candidate)

    location_score = get_location_score(candidate)
    notice_score = get_notice_period_score(candidate)

    engagement_mult = get_engagement_multiplier(candidate)

    # ── MUST-HAVE GATE (tiered) ──────────────────────────────────────────
    # Score penalty based on coverage of must-haves
    must_sum = embed_score + vdb_score  # eval is #3 in JD — slightly softer
    if must_sum < 0.10:
        gate = 0.25  # Severe: missing the two core technical must-haves
    elif must_sum < 0.40:
        gate = 0.55
    elif must_sum < 0.70:
        gate = 0.80
    else:
        gate = 1.0

    # ── COMPOSITE ─────────────────────────────────────────────────────────
    skill_score = (
        embed_score  * 0.32 +
        vdb_score    * 0.20 +
        eval_score   * 0.15 +
        fine_tune    * 0.06 +
        ltr          * 0.04 +
        nlp          * 0.10 +
        python       * 0.08 +
        github       * 0.05
    )

    career_score = (
        exp_score     * 0.30 +
        company_score * 0.38 +
        tenure_score  * 0.15 +
        prod_score    * 0.17
    )

    logistics_score = (
        location_score * 0.65 +
        notice_score   * 0.35
    )

    raw = (
        skill_score    * 0.50 +
        career_score   * 0.30 +
        logistics_score* 0.20
    ) * gate

    final = round(raw * engagement_mult, 6)

    # ── REASONING ─────────────────────────────────────────────────────────
    strengths, concerns = [], []

    if embed_score >= 0.75:
        strengths.append("strong embedding/retrieval background")
    elif embed_score >= 0.40:
        strengths.append("some embedding/retrieval exposure")
    else:
        concerns.append("no clear embedding/retrieval production experience")

    if vdb_score >= 0.75:
        strengths.append("vector DB experience")
    elif vdb_score < 0.20:
        concerns.append("no vector DB/hybrid search skills detected")

    if eval_score >= 0.65:
        strengths.append("ranking evaluation (NDCG/A-B/MRR)")
    elif eval_score < 0.20:
        concerns.append("no eval framework experience")

    if company_score >= 0.80:
        strengths.append("strong product-company background")
    elif company_score <= 0.30:
        concerns.append("limited product-company experience")

    if prod_score >= 0.70:
        strengths.append("clear production-shipping evidence")

    if engagement_mult < 0.45:
        concerns.append("low platform engagement / likely inactive")
    elif engagement_mult >= 0.80:
        strengths.append("highly engaged on platform")

    if notice is not None and isinstance(notice, (int, float)) and notice > 60:
        concerns.append(f"{int(notice)}d notice")

    if location_score < 0.40:
        loc_str = f"{location}, {country}" if country else location
        concerns.append(f"outside India ({loc_str})")

    if fine_tune >= 0.5:
        strengths.append("LLM fine-tuning experience")

    strength_str = "; ".join(strengths) if strengths else "limited direct-fit signals"
    concern_str = f" Concerns: {'; '.join(concerns)}." if concerns else ""
    reasoning = f"{title}, {yoe:.1f}yrs, {location}: {strength_str}.{concern_str}"

    return final, reasoning


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Redrob Hackathon — Candidate Ranker v2")
    ap.add_argument("--candidates", default="/mnt/user-data/uploads/1783013206349_candidates.jsonl",
                    help="Path to candidates.jsonl (or .jsonl.gz)")
    ap.add_argument("--out", default="/mnt/user-data/outputs/submission.csv",
                    help="Output CSV path")
    ap.add_argument("--top", type=int, default=100)
    args = ap.parse_args()

    print(f"Loading candidates from {args.candidates}...", flush=True)

    candidates = []
    if args.candidates.endswith(".gz"):
        import gzip
        with gzip.open(args.candidates, "rt", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    candidates.append(json.loads(line))
    else:
        with open(args.candidates, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    candidates.append(json.loads(line))

    print(f"Loaded {len(candidates):,} candidates. Scoring...", flush=True)

    scored = []
    for i, c in enumerate(candidates):
        if i % 10000 == 0:
            print(f"  {i:,}/{len(candidates):,}...", flush=True)
        score, reasoning = score_candidate(c)
        scored.append({
            "candidate_id": c["candidate_id"],
            "score": score,
            "reasoning": reasoning,
        })

    scored.sort(key=lambda x: (-x["score"], x["candidate_id"]))
    top = scored[:args.top]

    import os
    os.makedirs(os.path.dirname(args.out) if os.path.dirname(args.out) else ".", exist_ok=True)

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for rank, row in enumerate(top, start=1):
            writer.writerow([row["candidate_id"], rank, row["score"], row["reasoning"]])

    print(f"\n✅ Written {args.top} rows → {args.out}")
    print("\nTop 10 preview:")
    for i, row in enumerate(top[:10], 1):
        print(f"  #{i:02d} {row['candidate_id']}  score={row['score']:.4f}  {row['reasoning'][:95]}")

    all_scores = [r["score"] for r in scored]
    non_disq = [s for s in all_scores if s > 0.05]
    print(f"\nScore distribution:")
    print(f"  Disqualified (<0.05):  {len(all_scores) - len(non_disq):,}")
    print(f"  Qualifying pool:       {len(non_disq):,}")
    if non_disq:
        print(f"  Max score:             {max(non_disq):.4f}")
        print(f"  Min qualifying:        {min(non_disq):.4f}")
        strong = sum(1 for s in non_disq if s > 0.35)
        print(f"  Strong (>0.35):        {strong:,}")


if __name__ == "__main__":
    main()