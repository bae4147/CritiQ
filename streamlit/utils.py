# utils.py
import pandas as pd
import json
import base64

def create_iteration_tables(log_data):
    """
    이터레이션 로그 데이터를 기반으로 두 개의 Pandas DataFrame을 생성합니다.

    Args:
        log_data (list): 각 이터레이션의 로그가 담긴 JSON 리스트.
    
    Returns:
        tuple: (all_questions_df, best_question_df)
    """
    # Table 1: All Socratic Questions per Iteration
    questions_data = []
    for log in log_data:
        iteration_num = log.get("iteration")
        
        # 'all_persona_responses'가 없는 경우를 처리
        if "question_generation" not in log or "all_persona_responses" not in log["question_generation"]:
            continue

        all_questions = log["question_generation"]["all_persona_responses"]
        best_question_text = log["question_generation"].get("best_question")
        
        for i, question_info in enumerate(all_questions):
            score_result = question_info.get("score_result", {})
            questions_data.append({
                "iteration": iteration_num,
                "question_index": i + 1,
                "socratic_question": question_info.get("question"),
                "is_best": (question_info.get("question") == best_question_text),
                "judgement_score": score_result.get("judgement_score"),
                "reasoning_score": score_result.get("reasoning_path_score"),
                "confidence_score": score_result.get("certainty_score"),
                "total_score": score_result.get("total_score"),
                "explanation_judgement": score_result.get("explanation_judgement"),
                "explanation_reasoning": score_result.get("explanation_reasoning"),
                "explanation_confidence": score_result.get("explanation_certainty"),
            })
    
    all_questions_df = pd.DataFrame(questions_data)
    all_questions_df = all_questions_df.fillna("")

    # Table 2: Best Question & Persona Refinement per Iteration
    refinement_data = []
    for log in log_data:
        iteration_num = log.get("iteration")
        
        if "question_generation" not in log or "refinement" not in log:
            continue

        best_question_text = log["question_generation"].get("best_question", "")
        initial_user_response = log.get("initial_user_response", {})
        final_user_response = log.get("final_user_response", {})
        refinement = log.get("refinement", {})

        refinement_data.append({
            "iteration": iteration_num,
            "question_without_persona": log.get("statement"),
            "best_question": best_question_text,
            "user_reasoning": initial_user_response.get("reasoning"),
            "user_judgement": initial_user_response.get("judgement"),
            "updated_user_reasoning": final_user_response.get("reasoning"),
            "updated_user_judgement": final_user_response.get("judgement"),
            "loss": refinement.get("textual_loss"),
            "gradient": refinement.get("textual_gradient"),
            "updated_prompt": refinement.get("refined_persona_prompt"),
        })
    
    best_question_df = pd.DataFrame(refinement_data)
    best_question_df = best_question_df.fillna("")
    
    return all_questions_df, best_question_df


def create_csv_download_link(df, filename):
    """
    Pandas DataFrame을 CSV로 변환하고 Streamlit 다운로드 링크를 생성합니다.
    """
    csv = df.to_csv(index=False)
    # 한글 처리를 위해 utf-8-sig 인코딩 사용
    b64 = base64.b64encode(csv.encode('utf-8-sig')).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}" style="text-decoration:none; display:inline-block; padding: 10px 20px; background-color: #262730; color: #fff; border-radius: 5px;">CSV 파일 다운로드</a>'
    return href