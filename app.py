import streamlit as st

from utils import load_css
from pages import single_review, bulk_analysis


#  PAGE CONFIG  (must be first Streamlit call)

st.set_page_config(
    page_title="DeepCSAT+",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)



load_css()   # reads style.css and injects it


with st.sidebar:

    # Brand header
    st.markdown(
        '<div class="sidebar-brand">Deep<span class="brand-accent">CSAT</span>+</div>',
        unsafe_allow_html=True
    )
    st.markdown("*AI-powered CSAT Analysis*")
    st.markdown("---")

    # Page navigation
    page = st.radio(
        "Navigation",
        ["🔍  Single Review", "📂  Bulk Analysis"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("**Language Settings**")

    lang_display = ["Auto Detect", "Hindi", "Marathi", "English"]
    lang_map     = {"Auto Detect": "auto", "Hindi": "hi", "Marathi": "mr", "English": "en"}
    selected_lang    = st.selectbox("Source Language", lang_display, label_visibility="collapsed")
    source_lang      = lang_map[selected_lang]
    translate_toggle = st.checkbox("Translate to English before analysis")

    st.markdown("---")
    st.markdown("**About**")
    st.caption(
        "DeepCSAT+ analyzes customer feedback for any product or service using "
        "ML sentiment analysis, KeyBERT keyword extraction, and Claude AI for "
        "smart recommendations and star rating prediction."
    )


page_titles = {
    "🔍  Single Review": ("Single Review Analyzer", "Paste one review and get instant analysis"),
    "📂  Bulk Analysis": ("Bulk Dataset Analysis",  "Upload a CSV and analyze thousands of reviews"),
}
title, subtitle = page_titles[page]
st.markdown(f'<p class="page-title">{title}</p>',  unsafe_allow_html=True)
st.markdown(f'<p class="page-sub">{subtitle}</p>', unsafe_allow_html=True)
st.markdown("<div style='height:24px'></div>",     unsafe_allow_html=True)


if page == "🔍  Single Review":
    single_review.render(source_lang, translate_toggle)

elif page == "📂  Bulk Analysis":
    bulk_analysis.render()
