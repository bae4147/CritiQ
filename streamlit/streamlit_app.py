# streamlit_app.py

import streamlit as st
from your_module import QG_chain  # 기존 QG_chain 가져오기

st.set_page_config(page_title="CritiQ Demo", layout="centered")

# 상태 저장 (세션 state)
if "submitted" not in st.session_state:
    st.session_state.submitted = False
if "resubmitted" not in st.session_state:
    st.session_state.resubmitted = False
if "logs" not in st.session_state:
    st.session_state.logs = []

### 화면 1: 초기 입력 화면
if not st.session_state.submitted:
    st.markdown("#### 🧠 Statement")
    st.markdown(
        '<div style="border:1px solid #ccc; border-radius:10px; padding:10px;">'
        "In the United States, racial stratification still occurs. "
        "The racial wealth gap between African Americans and White Americans for the same job is found to be a factor of twenty."
        "</div>",
        unsafe_allow_html=True,
    )
    reasoning = st.text_area("Your Reasoning", height=150)
    judgement = st.radio("Your Judgement", ["valid", "invalid"])
    
    if st.button("Submit"):
        st.session_state.submitted = True
        st.session_state.reasoning = reasoning
        st.session_state.judgement = judgement

### action 1: 질문 생성
if st.session_state.submitted and not st.session_state.resubmitted:
    st.markdown("#### 🧠 Statement")
    st.markdown(
        '<div style="border:1px solid #ccc; border-radius:10px; padding:10px;">'
        "In the United States, racial stratification still occurs. "
        "The racial wealth gap between African Americans and White Americans for the same job is found to be a factor of twenty."
        "</div>",
        unsafe_allow_html=True,
    )

    with st.spinner("Generating Socratic Question..."):
        question = QG_chain.invoke({
            "statement": "In the United States, racial stratification still occurs. The racial wealth gap ...",
            "user_reasoning": st.session_state.reasoning,
            "user_judgement": st.session_state.judgement
        }).content.strip()
        st.session_state.generated_question = question

    st.markdown(
        f'<div style="background-color:#FFF8DA; border-radius:10px; padding:10px;">💡 <b>Socratic Question:</b><br>{question}</div>',
        unsafe_allow_html=True,
    )

    updated_reasoning = st.text_area("Revise Your Reasoning", value=st.session_state.reasoning, height=150, key="revised")
    updated_judgement = st.radio("Update Your Judgement", ["valid", "invalid"], index=["valid", "invalid"].index(st.session_state.judgement), key="revised_j")

    if st.button("Resubmit"):
        st.session_state.resubmitted = True
        st.session_state.logs.append({
            "original_reasoning": st.session_state.reasoning,
            "original_judgement": st.session_state.judgement,
            "question": st.session_state.generated_question,
            "updated_reasoning": updated_reasoning,
            "updated_judgement": updated_judgement,
        })

### action2: 로그 출력
if st.session_state.resubmitted:
    st.markdown("### ✅ Your Previous Logs")
    for i, log in enumerate(st.session_state.logs):
        st.markdown(f"#### Log {i+1}")
        st.markdown(f"- Original Reasoning: {log['original_reasoning']}")
        st.markdown(f"- Original Judgement: **{log['original_judgement']}**")
        st.markdown(f"- Socratic Question: {log['question']}")
        st.markdown(f"- Revised Reasoning: {log['updated_reasoning']}")
        st.markdown(f"- Revised Judgement: **{log['updated_judgement']}**")
