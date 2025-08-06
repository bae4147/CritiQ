# streamlit_app.py (Refactored for sequential execution)

import streamlit as st
import copy
from prototype import (
    initialize_state,
    generate_and_evaluate_questions,
    refine_persona,
    finalize_iteration_log,
)
import json
import base64
from utils import create_iteration_tables, create_csv_download_link

st.set_page_config(page_title="CritiQ Iterative", layout="centered")
st.title("🧠 CritiQ - Iterative Reasoning with Persona Refinement")

# ✅ 초기 설정: 세션 상태가 없으면 초기화
if "state" not in st.session_state:
    statements = [
        "In the United States, racial stratification still occurs. The racial wealth gap between African Americans and White Americans for the same job is found to be a factor of twenty.",
        "Governments should build high-rise buildings to solve the housing crisis. Some observations indicate that people tend to prefer housing in low-density areas.",
        "Governments should not subsidize tobacco production.A recent study found that the risk of dying from lung cancer before age 85 is about 22 times higher for a male smoker than for a non-smoker, illustrating the massive public health cost of tobacco use.",
        "The use of performance-enhancing drugs (PEDs) should be permitted in professional sports.According to some nurse practitioners, stopping all PED use may not be realistic, and regulated use could improve athlete safety while reducing underground abuse.",
        "Unsustainable logging should be banned globally. Forest degradation has been consistently linked to biodiversity loss and long-term environmental disruption, which are difficult to reverse."
    ]
    st.session_state.state = initialize_state(statements)
    st.session_state.submitted = False
    st.session_state.resubmitted = False

state = st.session_state.state
idx = state["current_index"]

# ✅ Iteration 종료 여부
if idx >= len(state["statements"]):
    st.success("🎉 All 5 iterations completed.")
    st.markdown("### 🧾 Full Iteration Logs")
    # st.code(json.dumps(state["iteration_logs"], indent=2), language="json")

        # 모든 반복이 완료된 후, 테이블과 CSV 다운로드 링크를 표시합니다.
    if "iteration_logs" in st.session_state.state and st.session_state.state["iteration_logs"]:
        st.markdown("---")
        st.header("### 📈 Iteration Summary Tables")
        
        iteration_logs = st.session_state.state["iteration_logs"]
        all_questions_df, best_question_df = create_iteration_tables(iteration_logs)
        
        # Table 1: All Socratic Questions per Iteration
        st.subheader("✅ Table 1: All Socratic Questions per Iteration")
        st.dataframe(all_questions_df, use_container_width=True)
        st.markdown(create_csv_download_link(all_questions_df, "all_questions_log.csv"), unsafe_allow_html=True)
        
        st.markdown("---")

        # Table 2: Best Question & Persona Refinement per Iteration
        st.subheader("✅ Table 2: Best Question & Persona Refinement per Iteration")
        st.dataframe(best_question_df, use_container_width=True)
        st.markdown(create_csv_download_link(best_question_df, "best_question_refinement_log.csv"), unsafe_allow_html=True)
    
    st.stop()

current_statement = state["statements"][idx]


# ✅ 현재 상태에 따라 UI 렌더링
# case 1: 초기 답변 입력 단계
if not st.session_state.submitted:
    st.markdown(f"### 🧠 Statement #{idx+1}")
    st.markdown("""주어진 제시문은 주장 1문장과 이를 뒷받침하는 근거 1문장으로 이루어져있습니다. 제시된 근거가 논리적으로 타당한 근거인지 판단해주세요.
<중요>
주장에 대한 자신의 의견을 서술하는 것이 아니라 근거가 주장을 논리적으로 잘 뒷받침하는지 여부를 판단해주셔야합니다.
근거에 제시된 연구 결과 혹은 사례는 참이라는 것을 전제로 판단해주세요.
""")
    st.markdown(f"<div style='border:1px solid #ccc; border-radius:10px; padding:10px;'>{current_statement}</div>", unsafe_allow_html=True)

    user_judgement = st.radio("🔎 Your Judgement", ["valid", "invalid"])
    user_reasoning = st.text_area("💭 Your Initial Reasoning")
    

    if st.button("Submit"):
        if not user_reasoning:
            st.error("이유를 입력해주세요.")
        else:
            with st.spinner("질문을 생성하고 있습니다. 잠시만 기다려주세요..."):
                # 1. 상태에 사용자 입력 저장
                st.session_state.state["user_reasoning"] = user_reasoning
                st.session_state.state["user_judgement"] = user_judgement

                # 2. 소크라테스식 질문 생성 및 평가 (함수 호출로 대체)
                try:
                    st.session_state.state = generate_and_evaluate_questions(st.session_state.state)
                    st.session_state.submitted = True
                except Exception as e:
                    import traceback
                    st.error("❌ 질문 생성 및 평가 중 예외가 발생했습니다.")
                    st.error(f"오류 메시지: {e}")
                    st.code(traceback.format_exc())
                    st.stop()
            st.rerun()

# case 2: 질문 생성 후 수정 단계
elif st.session_state.submitted and not st.session_state.resubmitted:
    state = st.session_state.state
    q = state.get("best_question")
    if not q:
      st.error("best_question이 생성되지 않았습니다. 이전 단계에서 오류가 발생했을 수 있습니다.")
      st.stop()

    st.markdown(f"### 🧠 Statement #{idx+1}")
    st.markdown(f"<div style='border:1px solid #ccc; border-radius:10px; padding:10px;'>{current_statement}</div>", unsafe_allow_html=True)
    st.markdown("### 💡 Socratic Question")
    st.markdown(f"<div style='background-color:#FFF8DA; border-radius:10px; padding:10px;'>{q}</div>", unsafe_allow_html=True)

    q_without_persona = state.get("question_without_persona")
    with st.expander("비교용 : persona 없이 만들어진 질문"):
        st.write(q_without_persona)


    prev_reasoning = state["user_reasoning"]
    prev_judgement = state["user_judgement"]

    updated_judgement = st.radio("🔁 Revise Your Judgement", options=["valid", "invalid"], index=0 if prev_judgement == "valid" else 1)
    updated_reasoning = st.text_area("✏️ Revise Your Reasoning", value=prev_reasoning)
    

    if st.button("Resubmit"):
        if not updated_reasoning:
            st.error("수정된 이유를 입력해주세요.")
        else:
            with st.spinner("페르소나를 개선하고 로그를 기록하고 있습니다. 잠시만 기다려주세요..."):
                # 1. 최종 유저 응답을 state에 저장
                st.session_state.state["final_user_response"] = {
                    "reasoning": updated_reasoning,
                    "judgement": updated_judgement
                }
                
                # 2. 페르소나 개선 및 로그 기록 (함수 호출로 대체)
                try:
                    st.session_state.state = refine_persona(st.session_state.state)
                    st.session_state.state = finalize_iteration_log(st.session_state.state)
                    st.session_state.resubmitted = True
                except Exception as e:
                    import traceback
                    st.error("❌ 페르소나 개선 또는 로그 기록 중 예외가 발생했습니다.")
                    st.error(f"오류 메시지: {e}")
                    st.code(traceback.format_exc())
                    st.stop()
                st.rerun()

# case 3: 반복 완료 및 다음 단계로 이동
else:
    st.success("✅ Iteration Completed")
    st.markdown("### 📜 Raw Iteration Log")
    last_log = st.session_state.state["iteration_logs"][-1]
    st.markdown(f"### Iteration Done Count : {len(st.session_state.state['iteration_logs'])}")
    # st.code(json.dumps(last_log, indent=2), language="json")

    # 여기부터 테이블 표시 로직을 추가합니다.
    if "iteration_logs" in st.session_state.state and st.session_state.state["iteration_logs"]:
        st.markdown("---")
        st.header("### 📈 Iteration Summary Tables")
        
        iteration_logs = st.session_state.state["iteration_logs"]
        all_questions_df, best_question_df = create_iteration_tables(iteration_logs)
        
        # Table 1: All Socratic Questions per Iteration
        st.subheader("✅ Table 1: All Socratic Questions per Iteration")
        st.dataframe(all_questions_df, use_container_width=True)
        st.markdown(create_csv_download_link(all_questions_df, "all_questions_log.csv"), unsafe_allow_html=True)
        
        st.markdown("---")

        # Table 2: Best Question & Persona Refinement per Iteration
        st.subheader("✅ Table 2: Best Question & Persona Refinement per Iteration")
        st.dataframe(best_question_df, use_container_width=True)
        st.markdown(create_csv_download_link(best_question_df, "best_question_refinement_log.csv"), unsafe_allow_html=True)


    if st.session_state.state["current_index"] < len(st.session_state.state["statements"]):
        if st.button("Next Statement"):
            # 다음 Statement를 위해 상태 초기화
            st.session_state.state["statement"] = st.session_state.state["statements"][st.session_state.state["current_index"]]
            st.session_state.state["user_reasoning"] = ""
            st.session_state.state["user_judgement"] = ""
            st.session_state.state["socratic_questions"] = []
            st.session_state.state["persona_responses"] = []
            st.session_state.state["best_question"] = None
            st.session_state.state["best_response"] = None
            st.session_state.state["final_user_response"] = None
            st.session_state.state["textual_loss"] = None
            st.session_state.state["textual_gradient"] = None
            st.session_state.state["question_without_persona"] = None

            # 플래그 초기화
            st.session_state.submitted = False
            st.session_state.resubmitted = False
            
            st.rerun()

    else:
        st.success("🎉 모든 Statement에 대한 평가가 완료되었습니다!")
