"""
pages/single_review.py
───────────────────────
Single Review Analyzer page for DeepCSAT+.
Renders the left input panel and right result panel.
All logic is imported from utils.py.
"""

import streamlit as st
from deep_translator import GoogleTranslator
import plotly.graph_objects as go

# Import shared utilities
import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent))

from utils import (
    load_ml_model,
    load_keybert,
    analyze_sentiment,
    calculate_csat,
    get_keywords,
    generate_suggestion,
    predict_star_with_ai,
    sentiment_badge,
    render_kw_pills,
    insight_box,
)


def render(source_lang: str, translate_toggle: bool):
    """
    Main render function called by app.py.
    Receives language settings from the sidebar.
    """

    # Load models (cached — no repeated loading)
    model, vectorizer = load_ml_model()
    kw_model          = load_keybert()

    col_form, col_result = st.columns([1, 1])

    # ── LEFT PANEL: INPUT ──────────────────────────────────────────
    with col_form:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">Enter Customer Review</div>', unsafe_allow_html=True)

        review = st.text_area(
            "Paste feedback",
            placeholder="e.g. The support team resolved my issue instantly, or The item arrived damaged...",
            height=160,
            label_visibility="collapsed",
        )

        # AI star predictor toggle
        use_ai_predictor = st.checkbox(
            "⭐ Predict Star Rating with AI",
            value=True,
            help="Uses Claude AI to predict the likely star rating. Requires ANTHROPIC_API_KEY in secrets.toml"
        )

        analyze_btn = st.button("🔍 Analyze Review")
        st.markdown('</div>', unsafe_allow_html=True)

    # ── RIGHT PANEL: RESULT ────────────────────────────────────────
    with col_result:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="panel">', unsafe_allow_html=True)

        if analyze_btn:

            if not review.strip():
                st.warning("Please enter a review before analyzing.")

            else:
                with st.spinner("Analyzing…"):

                    # Step 1 — optionally translate
                    processed = review
                    if translate_toggle:
                        try:
                            processed = GoogleTranslator(
                                source=source_lang, target='en'
                            ).translate(review)
                            st.info(f"**Translated:** {processed}")
                        except Exception as e:
                            st.warning(f"Translation failed: {e}. Analyzing original text.")

                    # Step 2 — sentiment + polarity
                    sentiment, polarity = analyze_sentiment(processed, model, vectorizer)

                    # Step 3 — CSAT score
                    csat = calculate_csat(polarity)

                    # Step 4 — keywords
                    keywords = get_keywords(processed, kw_model, top_n=6)

                    # Step 5 — fallback suggestion
                    suggestion = generate_suggestion(sentiment)

                    chip_color = {
                        "Positive": "chip-positive",
                        "Negative": "chip-negative",
                        "Neutral":  "chip-neutral"
                    }.get(sentiment, "")

                # ── RESULTS TITLE
                st.markdown('<div class="panel-title">Analysis Results</div>', unsafe_allow_html=True)

                # ── RESULT CHIPS (Sentiment / Polarity / CSAT)
                st.markdown(f"""
                <div class="result-chips">
                    <div class="result-chip">
                        <div class="result-chip-label">Sentiment</div>
                        <div class="result-chip-val {chip_color}" style="font-size:16px;padding-top:4px">{sentiment}</div>
                    </div>
                    <div class="result-chip">
                        <div class="result-chip-label">Polarity</div>
                        <div class="result-chip-val">{polarity:.2f}</div>
                    </div>
                    <div class="result-chip">
                        <div class="result-chip-label">CSAT Score</div>
                        <div class="result-chip-val chip-csat">
                            {csat}
                            <span style="font-size:14px;color:rgba(148,163,220,0.5)">/5</span>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # ── POLARITY GAUGE
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=round(polarity, 2),
                    number={"font": {"family": "DM Sans", "size": 28, "color": "#c7d2fe"}},
                    gauge={
                        "axis": {
                            "range": [-1, 1],
                            "tickwidth": 1,
                            "tickcolor": "rgba(99,102,241,0.3)",
                            "tickfont": {"family": "DM Sans", "size": 10, "color": "rgba(148,163,220,0.5)"}
                        },
                        "bar": {"color": "#6366f1", "thickness": 0.25},
                        "bgcolor": "rgba(15,20,50,0.0)",
                        "borderwidth": 0,
                        "steps": [
                            {"range": [-1,   -0.2], "color": "rgba(239,68,68,0.12)"},
                            {"range": [-0.2,  0.2], "color": "rgba(245,158,11,0.10)"},
                            {"range": [0.2,     1], "color": "rgba(16,185,129,0.12)"},
                        ],
                        "threshold": {
                            "line": {"color": "#a5b4fc", "width": 2},
                            "thickness": 0.7,
                            "value": polarity,
                        },
                    },
                ))
                fig_gauge.update_layout(
                    height=180,
                    margin=dict(l=20, r=20, t=20, b=10),
                    paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(family="DM Sans"),
                )
                st.plotly_chart(fig_gauge, use_container_width=True, config={"displayModeBar": False})

                # ── DETECTED KEYWORDS
                if keywords:
                    st.markdown(
                        '<div style="margin-top:4px"><b style="font-size:13px;color:#c7d2fe">Keywords Detected</b></div>',
                        unsafe_allow_html=True
                    )
                    st.markdown(render_kw_pills(keywords), unsafe_allow_html=True)

                # ── FEATURE 8: AI STAR RATING PREDICTOR
                if use_ai_predictor:
                    with st.spinner("Predicting star rating with AI…"):
                        prediction     = predict_star_with_ai(processed, polarity, sentiment)
                        stars_filled   = "⭐" * prediction["stars"]
                        stars_empty    = "☆"  * (5 - prediction["stars"])
                        confidence_pct = int(prediction["confidence"] * 100)

                    st.markdown(f"""
                    <div class="star-predictor-card">
                        <div class="star-predictor-label">⭐ AI Predicted Star Rating</div>
                        <div class="star-predictor-stars">{stars_filled}{stars_empty}</div>
                        <div class="star-predictor-meta">{prediction['stars']}/5 &nbsp;·&nbsp; {confidence_pct}% confidence</div>
                        <div class="star-predictor-reason">"{prediction['reasoning']}"</div>
                    </div>
                    """, unsafe_allow_html=True)

                # ── FALLBACK INSIGHT BOX
                st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
                ins_kind = {"Positive": "success", "Negative": "danger", "Neutral": "warning"}.get(sentiment, "info")
                st.markdown(insight_box(suggestion, ins_kind), unsafe_allow_html=True)

        else:
            # ── EMPTY STATE (before any analysis)
            st.markdown("""
            <div style="text-align:center;padding:40px 24px;">
                <div style="font-size:40px;margin-bottom:12px">🔍</div>
                <div style="color:rgba(148,163,220,0.6);font-size:14px">
                    Enter a review on the left and click
                    <strong style="color:#a5b4fc">Analyze Review</strong>
                    to see results here.
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)
