# streamlit_app.py

import streamlit as st
from prototype import (
    initialize_state_from_user, generate_question, evaluate_question,
    State  # TypedDict
)

st.set_page_config(layout="centered")

if "stage" not in st.session_state:
    st.session_state.stage = "input"  # input / generated / resubmitted
    st.session_state.logs = []

st.title("🧠 CritiQ - Socratic Assistant")

# === Stage 1: Input ===
if st.session_state.stage == "input":
    statement = st.text_area("📝 Statement", height=100)
    reasoning = st.text_area("💭 Your Reasoning", height=150)
    judgement = st.radio("🧪 Your Judgement", ["valid", "invalid"])

    if st.button("Submit"):
        state = initialize_state_from_user(statement, reasoning, judgement)
        state = generate_question(state)
        state = evaluate_question(state)

        st.session_state.state = state
        st.session_state.stage = "generated"

# === Stage 2: Show generated question ===
if st.session_state.stage == "generated":
    state = st.session_state.state

    st.markdown("### 🧠 Statement")
    st.markdown(f'<div style="border:1px solid #ccc; border-radius:10px; padding:10px;">{state["statement"]}</div>', unsafe_allow_html=True)

    st.markdown("### 💡 Socratic Question")
    st.markdown(f'<div style="background-color:#FFF8DA; border-radius:10px; padding:10px;">{state["best_question"]}</div>', unsafe_allow_html=True)

    revised_reasoning = st.text_area("✏️ Update Your Reasoning", height=150)
    revised_judgement = st.radio("🔁 Update Your Judgement", ["valid", "invalid"])

    if st.button("Resubmit"):
        st.session_state.stage = "resubmitted"
        st.session_state.logs.append({
            "statement": state["statement"],
            "question": state["best_question"],
            "original_reasoning": state["user_reasoning"],
            "original_judgement": state["user_judgement"],
            "revised_reasoning": revised_reasoning,
            "revised_judgement": revised_judgement
        })

# === Stage 3: Show logs ===
if st.session_state.stage == "resubmitted":
    st.markdown("## 📜 Response Log")

    for i, log in enumerate(st.session_state.logs):
        with st.expander(f"Log {i+1}"):
            st.markdown(f"**Statement:** {log['statement']}")
            st.markdown(f"**Socratic Question:** {log['question']}")
            st.markdown(f"- Original Reasoning: {log['original_reasoning']}")
            st.markdown(f"- Original Judgement: **{log['original_judgement']}**")
            st.markdown(f"- Revised Reasoning: {log['revised_reasoning']}")
            st.markdown(f"- Revised Judgement: **{log['revised_judgement']}**")

    if st.button("🔁 Start New"):
        st.session_state.stage = "input"
