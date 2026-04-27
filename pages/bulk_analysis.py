"""
pages/bulk_analysis.py
───────────────────────
Bulk Dataset Analysis page for DeepCSAT+.
Handles CSV upload, full dataset analysis, charts, keywords,
and the AI Recommendations panel (Feature 1).
All logic is imported from utils.py.
"""

import streamlit as st
import pandas as pd
import chardet
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
    generate_ai_recommendations,
    sentiment_badge,
    render_kw_pills,
    insight_box,
)


def render():
    """
    Main render function called by app.py for the Bulk Analysis page.
    No arguments needed — all settings come from session state or are self-contained.
    """

    # Load models (cached)
    model, vectorizer = load_ml_model()
    kw_model          = load_keybert()

    # ── FILE UPLOAD ────────────────────────────────────────────────
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">Upload Review Dataset</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Upload CSV",
        type=["csv"],
        label_visibility="collapsed",
    )
    st.markdown('</div>', unsafe_allow_html=True)

    if uploaded_file is None:
        return  # Nothing to show yet

    # ── READ CSV ───────────────────────────────────────────────────
    raw_data = uploaded_file.read()
    encoding = chardet.detect(raw_data).get("encoding", "utf-8") or "utf-8"
    uploaded_file.seek(0)
    try:
        df = pd.read_csv(uploaded_file, encoding=encoding)
    except Exception:
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file, encoding="latin1")

    # Use "review" column if it exists, otherwise fall back to first column
    if "review" not in df.columns:
        df["review"] = df.iloc[:, 0].astype(str)
    else:
        df["review"] = df["review"].astype(str)

    # ── ANALYZE ALL ROWS ───────────────────────────────────────────
    with st.spinner("Analyzing your dataset… this may take a moment ⏳"):
        sentiments, polarities, csats = [], [], []
        for text in df["review"]:
            s, p = analyze_sentiment(text, model, vectorizer)
            sentiments.append(s)
            polarities.append(p)
            csats.append(calculate_csat(p))

        df["Sentiment"] = sentiments
        df["Polarity"]  = polarities
        df["CSAT"]      = csats

    # ── SUMMARY STATS ──────────────────────────────────────────────
    total    = len(df)
    pos      = (df["Sentiment"] == "Positive").sum()
    neg      = (df["Sentiment"] == "Negative").sum()
    neu      = (df["Sentiment"] == "Neutral").sum()
    avg_csat = df["CSAT"].mean()
    avg_pol  = df["Polarity"].mean()

    # ── METRIC CARDS ───────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)

    def mk(col, label, val, unit, color):
        with col:
            st.markdown(f"""
            <div class="metric-card {color}">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{val}<span class="metric-unit">{unit}</span></div>
            </div>
            """, unsafe_allow_html=True)

    mk(c1, "Total Reviews", f"{total:,}",                   "",   "blue")
    mk(c2, "Positive",      f"{round(pos/total*100, 1)}",   "%",  "green")
    mk(c3, "Negative",      f"{round(neg/total*100, 1)}",   "%",  "red")
    mk(c4, "Avg Polarity",  f"{avg_pol:.2f}",               "",   "amber")
    mk(c5, "Avg CSAT",      f"{avg_csat:.1f}",              "/5", "purple")

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    # ── CHARTS ROW ─────────────────────────────────────────────────
    ch1, ch2 = st.columns(2)

    with ch1:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">Sentiment Breakdown</div>', unsafe_allow_html=True)
        fig_pie = go.Figure(go.Pie(
            labels=["Positive", "Neutral", "Negative"],
            values=[int(pos), int(neu), int(neg)],
            marker=dict(
                colors=["#10b981", "#f59e0b", "#ef4444"],
                line=dict(color="rgba(0,0,0,0)", width=0)
            ),
            hole=0.60,
            textfont=dict(family="DM Sans", size=12, color="#c7d2fe"),
            textinfo="percent+label",
        ))
        fig_pie.update_layout(
            height=250,
            margin=dict(l=0, r=0, t=0, b=0),
            paper_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
            font=dict(family="DM Sans", color="#c7d2fe"),
        )
        st.plotly_chart(fig_pie, use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)

    with ch2:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">CSAT Distribution</div>', unsafe_allow_html=True)
        csat_dist = df["CSAT"].value_counts().sort_index()
        fig_csat = go.Figure(go.Bar(
            x=[f"{i} ★" for i in csat_dist.index],
            y=csat_dist.values,
            marker=dict(
                color=["#ef4444","#f87171","#f59e0b","#10b981","#059669"][:len(csat_dist)],
                line=dict(width=0),
                opacity=0.85,
            ),
            text=csat_dist.values,
            textposition="outside",
            textfont=dict(family="DM Sans", size=11, color="#a5b4fc"),
        ))
        fig_csat.update_layout(
            height=250,
            margin=dict(l=0, r=10, t=4, b=4),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(showgrid=False, tickfont=dict(family="DM Sans", size=12, color="#a5b4fc")),
            yaxis=dict(showgrid=True, gridcolor="rgba(99,102,241,0.10)", tickfont=dict(family="DM Sans", size=11, color="rgba(148,163,220,0.5)")),
            showlegend=False,
        )
        st.plotly_chart(fig_csat, use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)

    # ── POLARITY TREND ─────────────────────────────────────────────
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">Polarity Trend <span class="panel-sub">Smoothed rolling average (window=50)</span></div>', unsafe_allow_html=True)
    df["Smooth"] = df["Polarity"].rolling(min(50, max(1, len(df) // 20))).mean()
    fig_trend = go.Figure()
    fig_trend.add_trace(go.Scatter(
        y=df["Polarity"], mode="lines", name="Raw Polarity",
        line=dict(color="rgba(99,102,241,0.25)", width=1),
    ))
    fig_trend.add_trace(go.Scatter(
        y=df["Smooth"], mode="lines", name="Smoothed",
        line=dict(color="#818cf8", width=2.5),
    ))
    fig_trend.add_hline(y=0, line_dash="dash", line_color="#ef4444", line_width=1, opacity=0.4)
    fig_trend.update_layout(
        height=220,
        margin=dict(l=0, r=10, t=4, b=4),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=False, tickfont=dict(family="DM Sans", size=11, color="rgba(148,163,220,0.5)")),
        yaxis=dict(showgrid=True, gridcolor="rgba(99,102,241,0.08)", tickfont=dict(family="DM Sans", size=11, color="rgba(148,163,220,0.5)"), range=[-1.2, 1.2]),
        legend=dict(orientation="h", yanchor="bottom", y=1, font=dict(family="DM Sans", size=12, color="#a5b4fc")),
        font=dict(family="DM Sans"),
    )
    st.plotly_chart(fig_trend, use_container_width=True, config={"displayModeBar": False})
    st.markdown('</div>', unsafe_allow_html=True)

    # ── KEYWORD EXTRACTION ─────────────────────────────────────────
    # Done OUTSIDE column blocks so variables are available to AI panel below
    with st.spinner("Extracting keywords…"):
        all_text = " ".join(df["review"].head(500).tolist())
        top_kws  = get_keywords(all_text, kw_model, top_n=10)

    neg_reviews = df[df["Sentiment"] == "Negative"]["review"]
    neg_kws     = []
    if len(neg_reviews) > 0:
        with st.spinner("Extracting issue keywords…"):
            neg_text = " ".join(neg_reviews.head(300).tolist())
            neg_kws  = get_keywords(neg_text, kw_model, top_n=8)

    # ── KEYWORDS + ISSUES SIDE BY SIDE ────────────────────────────
    kw_col, iss_col = st.columns(2)

    with kw_col:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">Top Keywords <span class="panel-sub">All reviews</span></div>', unsafe_allow_html=True)
        if top_kws:
            st.markdown(render_kw_pills(top_kws), unsafe_allow_html=True)
            st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
            kw_df = pd.DataFrame(top_kws, columns=["Keyword", "Score"])
            kw_df["Score"] = kw_df["Score"].round(3)
            rows = "".join(
                f'<tr><td>{r["Keyword"]}</td><td>{r["Score"]}</td></tr>'
                for _, r in kw_df.iterrows()
            )
            st.markdown(
                f'<table class="styled-table"><thead><tr><th>Keyword</th><th>Relevance Score</th></tr></thead><tbody>{rows}</tbody></table>',
                unsafe_allow_html=True
            )
        else:
            st.info("Not enough text to extract keywords.")
        st.markdown('</div>', unsafe_allow_html=True)

    with iss_col:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">Issue Analysis <span class="panel-sub">Negative reviews</span></div>', unsafe_allow_html=True)
        if neg_kws:
            st.markdown(render_kw_pills(neg_kws, issue=True), unsafe_allow_html=True)
            st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
            iss_df = pd.DataFrame(neg_kws, columns=["Issue", "Severity"])
            iss_df["Severity"] = iss_df["Severity"].round(3)
            rows = "".join(
                f'<tr><td>{r["Issue"]}</td><td>{r["Severity"]}</td></tr>'
                for _, r in iss_df.iterrows()
            )
            st.markdown(
                f'<table class="styled-table"><thead><tr><th>Issue</th><th>Severity</th></tr></thead><tbody>{rows}</tbody></table>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(insight_box("✅ No significant issues detected in the dataset.", "success"), unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ── SAMPLE TABLE ───────────────────────────────────────────────
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">Sample Results <span class="panel-sub">First 20 rows</span></div>', unsafe_allow_html=True)
    sample = df[["review", "Sentiment", "Polarity", "CSAT"]].head(20).copy()
    sample["review"]   = sample["review"].str[:80] + "…"
    sample["Polarity"] = sample["Polarity"].round(2)

    rows_html = ""
    for _, r in sample.iterrows():
        rows_html += f"""
        <tr>
            <td style="max-width:420px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">{r['review']}</td>
            <td>{sentiment_badge(r['Sentiment'])}</td>
            <td style="text-align:center">{r['Polarity']}</td>
            <td style="text-align:center">{'⭐' * int(r['CSAT'])}</td>
        </tr>"""

    st.markdown(f"""
    <table class="styled-table">
        <thead><tr>
            <th>Review</th><th>Sentiment</th>
            <th style="text-align:center">Polarity</th>
            <th style="text-align:center">CSAT</th>
        </tr></thead>
        <tbody>{rows_html}</tbody>
    </table>""", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ── DOWNLOAD BUTTON ────────────────────────────────────────────
    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
    csv_out = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️  Download Full Analyzed CSV",
        data=csv_out,
        file_name="deepcsat_results.csv",
        mime="text/csv",
    )

    # ── FEATURE 1: AI RECOMMENDATIONS ─────────────────────────────
    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown(
        '<div class="panel-title">🤖 AI Recommendations <span class="panel-sub">Powered by Claude · based on your dataset</span></div>',
        unsafe_allow_html=True
    )

    with st.spinner("Generating AI recommendations…"):
        analysis_data = {
            "total":        total,
            "pos_pct":      round(pos / total * 100, 1),
            "neg_pct":      round(neg / total * 100, 1),
            "avg_pol":      round(float(avg_pol), 2),
            "avg_csat":     round(float(avg_csat), 1),
            "top_keywords": [(kw, round(sc, 3)) for kw, sc in top_kws],
            "neg_keywords": [(kw, round(sc, 3)) for kw, sc in neg_kws],
            "csat_dist":    df["CSAT"].value_counts().sort_index().to_dict(),
        }
        recs = generate_ai_recommendations(analysis_data)

    priority_colors = {"High": "#ef4444", "Medium": "#f59e0b", "Low": "#10b981"}

    for rec in recs:
        color = priority_colors.get(rec.get("priority", "Medium"), "#818cf8")
        st.markdown(f"""
        <div class="rec-card" style="border-left: 3px solid {color};">
            <div class="rec-title">
                {rec['title']}
                <span class="rec-badge" style="color:{color};">{rec['priority']}</span>
            </div>
            <div class="rec-detail">{rec['detail']}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)
