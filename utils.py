"""
utils.py
────────
All shared logic for DeepCSAT+:
  - CSS loader
  - ML model loader
  - Text analysis functions
  - AI feature functions (Star Predictor + Recommendations)
  - UI helper functions (badges, pills, insight boxes)
"""

import re
import json
import pickle
import pathlib

import streamlit as st
import anthropic
from keybert import KeyBERT


# ─────────────────────────────────────────────
#  CSS LOADER
# ─────────────────────────────────────────────
def load_css():
    """Reads style.css from the project root and injects it into Streamlit."""
    css_path = pathlib.Path(__file__).parent / "style.css"
    with open(css_path, "r") as f:
        css = f.read()
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  MODEL LOADERS
# ─────────────────────────────────────────────
@st.cache_resource
def load_ml_model():
    """
    Loads the pre-trained sentiment ML model and vectorizer from pickle files.
    Cached so it only loads once per Streamlit session.
    """
    model      = pickle.load(open("churn_model.pkl",  "rb"))
    vectorizer = pickle.load(open("vectorizer.pkl",   "rb"))
    return model, vectorizer


@st.cache_resource
def load_keybert():
    """Loads KeyBERT model. Cached so it only loads once."""
    return KeyBERT()


# ─────────────────────────────────────────────
#  TEXT CLEANING
# ─────────────────────────────────────────────
def clean_text(text: str) -> str:
    """Strips non-alphabetic characters and lowercases the text."""
    text = str(text)
    text = re.sub(r'[^a-zA-Z ]', '', text)
    return text.lower().strip()


# ─────────────────────────────────────────────
#  SENTIMENT ANALYSIS  (ML model)
# ─────────────────────────────────────────────
def analyze_sentiment(text: str, model, vectorizer) -> tuple[str, float]:
    """
    Runs the ML model on input text.
    Returns (sentiment_label, polarity_score).
      - sentiment_label : "Positive" | "Negative" | "Neutral"
      - polarity_score  : float in range [-1.0, +1.0]
    """
    vec  = vectorizer.transform([text])
    pred = model.predict(vec)[0]

    try:
        proba = model.predict_proba(vec).max()
    except Exception:
        proba = 1.0

    if pred == 1:
        return "Negative", float(-proba)
    elif pred == 0:
        return "Positive", float(proba)
    else:
        return "Neutral", 0.0


# ─────────────────────────────────────────────
#  CSAT SCORE  (rule-based from polarity)
# ─────────────────────────────────────────────
def calculate_csat(polarity: float) -> int:
    if   polarity >= 0.7:   return 5
    elif polarity >= 0.3:   return 4
    elif polarity >= -0.3:  return 3   # wider neutral zone
    elif polarity >= -0.7:  return 2
    else:                   return 1


# ─────────────────────────────────────────────
#  KEYWORD EXTRACTION  (KeyBERT)
# ─────────────────────────────────────────────
def get_keywords(text: str, kw_model, top_n: int = 6) -> list:
    """
    Extracts top-N keywords from the input text using KeyBERT.
    Returns a list of (keyword, score) tuples, or [] if text is empty.
    """
    cleaned = clean_text(text)
    if not cleaned:
        return []
    return kw_model.extract_keywords(cleaned, top_n=top_n)


# ─────────────────────────────────────────────
#  FALLBACK SUGGESTION  (no API needed)
# ─────────────────────────────────────────────
def generate_suggestion(sentiment: str) -> str:
    """
    Returns a generic hardcoded suggestion based on sentiment label.
    Used as the quick fallback when the Claude API is not called.
    """
    suggestions = {
        "Negative": "⚠️ Address recurring complaints immediately. Focus on quality, response time, and issue resolution.",
        "Positive": "✅ Customers are happy! Reward loyalty, gather testimonials, and maintain this standard.",
        "Neutral":  "📊 Feedback is mixed. Dig deeper with follow-up surveys to find hidden pain points.",
    }
    return suggestions.get(sentiment, "Collect more data for better insights.")


# ─────────────────────────────────────────────
#  FEATURE 8 ── AI STAR RATING PREDICTOR
# ─────────────────────────────────────────────
def predict_star_with_ai(review_text: str, polarity: float, sentiment: str) -> dict:
    """
    Sends review text + polarity + sentiment to Claude API.
    Returns a dict: { stars (int 1-5), confidence (float), reasoning (str) }

    Works for ANY product or service — no domain assumptions in the prompt.
    Falls back to calculate_csat() rule if API call fails.
    """
    try:
        api_key = st.secrets.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not set in .streamlit/secrets.toml")

        client = anthropic.Anthropic(api_key=api_key)

        prompt = f"""You are a review scoring expert. Predict the star rating for this customer review.

Review: "{review_text}"
Sentiment detected: {sentiment}
Polarity score: {polarity:.3f}  (range: -1.0 = very negative, +1.0 = very positive)

Instructions:
- Do NOT assume the product or service type.
- Use only the text content, sentiment label, and polarity score to decide.
- Return ONLY a valid JSON object — no markdown, no extra text.
- JSON keys: "stars" (integer 1-5), "confidence" (float 0.0-1.0), "reasoning" (string, max 15 words)

Example:
{{"stars": 2, "confidence": 0.81, "reasoning": "Mixed tone with clear dissatisfaction about core experience."}}"""

        message = client.messages.create(
            model="claude-opus-4-5",
            max_tokens=150,
            messages=[{"role": "user", "content": prompt}]
        )

        raw = message.content[0].text.strip()
        # Strip markdown fences if model wraps response in them
        raw = re.sub(r"^```[a-z]*\n?", "", raw)
        raw = re.sub(r"\n?```$",        "", raw)
        result = json.loads(raw)

        # Clamp values to safe ranges
        result["stars"]      = max(1, min(5, int(result["stars"])))
        result["confidence"] = max(0.0, min(1.0, float(result["confidence"])))
        result["reasoning"]  = str(result.get("reasoning", ""))
        return result

    except Exception:
        # Silent fallback — rule-based result
        stars      = calculate_csat(polarity)
        confidence = round(min(0.99, abs(polarity) * 0.8 + 0.2), 2)
        return {
            "stars":      stars,
            "confidence": confidence,
            "reasoning":  "Estimated from polarity score (AI unavailable)."
        }


# ─────────────────────────────────────────────
#  FEATURE 1 ── AI BULK RECOMMENDATIONS
# ─────────────────────────────────────────────
def generate_ai_recommendations(analysis_data: dict) -> list:
    """
    Sends bulk dataset summary to Claude API.
    Returns a list of 4 recommendation dicts:
      { title (str), detail (str), priority ("High"|"Medium"|"Low") }

    Prompt is product-agnostic — Claude reads keywords to infer context.
    Falls back to _fallback_recs() if API call fails.
    """
    try:
        api_key = st.secrets.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not set in .streamlit/secrets.toml")

        client = anthropic.Anthropic(api_key=api_key)

        prompt = f"""You are a senior customer experience analyst.
Analyze this CSAT dataset summary and generate exactly 4 specific, actionable recommendations.

Dataset Summary:
- Total Reviews        : {analysis_data['total']:,}
- Positive Reviews     : {analysis_data['pos_pct']}%
- Negative Reviews     : {analysis_data['neg_pct']}%
- Average Polarity     : {analysis_data['avg_pol']} (scale -1.0 to +1.0)
- Average CSAT Score   : {analysis_data['avg_csat']}/5
- Top Keywords (all)   : {analysis_data['top_keywords']}
- Top Keywords (negative only): {analysis_data['neg_keywords']}
- CSAT Distribution    : {analysis_data['csat_dist']}

Rules:
- Do NOT assume the product or service category. Derive all insights from keywords and numbers only.
- Each recommendation must reference at least one specific keyword from the data.
- Use varied priorities: at least one High, one Medium, one Low.
- Return ONLY a valid JSON array — no markdown, no preamble.
- Each item must have: "title" (max 8 words), "detail" (1-2 sentences), "priority" ("High"|"Medium"|"Low")

Example:
[
  {{"title": "Fix top complaint driver urgently", "detail": "Keyword X dominates negative reviews. Investigate root cause and resolve.", "priority": "High"}},
  ...
]"""

        message = client.messages.create(
            model="claude-opus-4-5",
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}]
        )

        raw = message.content[0].text.strip()
        raw = re.sub(r"^```[a-z]*\n?", "", raw)
        raw = re.sub(r"\n?```$",        "", raw)
        recs = json.loads(raw)

        validated = []
        for r in recs:
            if all(k in r for k in ("title", "detail", "priority")):
                r["priority"] = r["priority"].capitalize()
                if r["priority"] not in ("High", "Medium", "Low"):
                    r["priority"] = "Medium"
                validated.append(r)
        return validated if validated else _fallback_recs(analysis_data)

    except Exception:
        return _fallback_recs(analysis_data)


def _fallback_recs(analysis_data: dict) -> list:
    """Rule-based fallback recommendations when Claude API is unavailable."""
    recs     = []
    neg_pct  = analysis_data["neg_pct"]
    avg_csat = analysis_data["avg_csat"]

    if neg_pct > 40:
        recs.append({
            "title":    "High negative volume requires immediate action",
            "detail":   f"{neg_pct}% of reviews are negative. Review top issue keywords and prioritise rapid resolution.",
            "priority": "High"
        })
    if avg_csat < 3.5:
        recs.append({
            "title":    "CSAT score is below acceptable threshold",
            "detail":   f"Average CSAT of {avg_csat}/5 signals widespread dissatisfaction. Focus on the most-cited negative keywords.",
            "priority": "High"
        })
    recs.append({
        "title":    "Leverage positive reviews for marketing",
        "detail":   f"{analysis_data['pos_pct']}% of reviews are positive. Extract top positive keywords for campaigns.",
        "priority": "Medium"
    })
    recs.append({
        "title":    "Survey neutral customers for quick wins",
        "detail":   "Neutral reviewers are on the fence. A short follow-up survey can reveal easy improvements.",
        "priority": "Low"
    })
    return recs


# ─────────────────────────────────────────────
#  UI HELPER FUNCTIONS
# ─────────────────────────────────────────────
def sentiment_badge(s: str) -> str:
    """Returns an HTML badge span for a sentiment label."""
    cls = {"Positive": "badge-pos", "Negative": "badge-neg", "Neutral": "badge-neu"}.get(s, "badge-neu")
    return f'<span class="badge {cls}">{s}</span>'


def render_kw_pills(keywords: list, issue: bool = False) -> str:
    """Returns HTML for keyword pill tags."""
    cls   = "kw-pill issue" if issue else "kw-pill"
    pills = "".join(
        f'<span class="{cls}">{kw} <span class="kw-score">{score:.2f}</span></span>'
        for kw, score in keywords
    )
    return f'<div style="line-height:2.2">{pills}</div>'


def insight_box(text: str, kind: str = "info") -> str:
    """Returns an HTML insight/alert box."""
    return f'<div class="insight-box {kind}">{text}</div>'
