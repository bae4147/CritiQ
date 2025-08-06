# prototype.py (Refactored for sequential execution)

import os
from typing import TypedDict
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import re
import json

load_dotenv()

# Define the state dictionary for type hinting
class State(TypedDict):
    statement: str
    user_reasoning: str
    user_judgement: str
    socratic_questions: list[str]
    persona_responses: list[dict]
    best_question: str
    best_response: dict
    final_user_response: dict
    persona_prompt_history: list[str]
    current_persona_prompt: str
    textual_loss: str
    textual_gradient: str
    statements: list[str]
    iteration_logs: list[dict]
    current_index: int

# --- Helper Functions (unchanged from original) ---

def extract_between_tags(text, tag="IMPROVED_VARIABLE"):
    pattern = rf"<{tag}>\s*(.*?)\s*</{tag}>"
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1).strip() if match else text.strip()

def parse_persona_output(text: str) -> tuple[str, str]:
    reasoning = extract_between_tags(text, "UPDATED_REASONING")
    judgement = extract_between_tags(text, "UPDATED_JUDGEMENT")
    return reasoning, judgement

def parse_evaluator_response(response_text):
    try:
        judgement_score = int(re.search(r"Judgement Score\s*:\s*(\d)", response_text).group(1))
        reasoning_score = int(re.search(r"Reasoning Path Score\s*:\s*(\d)", response_text).group(1))
        certainty_score = int(re.search(r"Certainty Score\s*:\s*(\d)", response_text).group(1))
        is_good_question = re.search(r"Is Good Question\s*:\s*(Yes|No)", response_text, re.IGNORECASE).group(1)
        explanation_judgement = re.search(r"Explanation \(Judgement\)\s*:\s*(.+?)(?=\nExplanation \(Reasoning Path\)|\Z)", response_text, re.DOTALL).group(1).strip()
        explanation_reasoning = re.search(r"Explanation \(Reasoning Path\)\s*:\s*(.+?)(?=\nExplanation \(Certainty\)|\Z)", response_text, re.DOTALL).group(1).strip()
        explanation_certainty = re.search(r"Explanation \(Certainty\)\s*:\s*(.+)", response_text, re.DOTALL).group(1).strip()
        return {
            "judgement_score": judgement_score,
            "reasoning_path_score": reasoning_score,
            "certainty_score": certainty_score,
            "total_score": judgement_score + reasoning_score + certainty_score,
            "is_good_question": is_good_question,
            "explanation_judgement": explanation_judgement,
            "explanation_reasoning": explanation_reasoning,
            "explanation_certainty": explanation_certainty
        }
    except Exception as e:
        print("Parsing error:", e)
        return None

# --- Model Definitions (unchanged from original) ---

QG_model = ChatOpenAI(model="gpt-4o-mini", temperature=0.9)
QG_prompt = ChatPromptTemplate.from_messages([
    ("system", """
You are a Socratic assistant designed to help users critically reflect on argumentative statements.

Each input consists of:
1. An argumentative **statement** (containing a claim and a reason)
2. The user's **reasoning process** (their explanation)
3. The user's **current judgement** (valid or invalid)

Generate exactly **one** Socratic question that:
- Is open-ended
- Probes assumptions, reasoning, or implications
- Does not provide answers or evaluation

**Format your response as follows:**
"""),
    ("human", """
statement: {statement}
user_reasoning: {user_reasoning}
user_judgement: {user_judgement}
"""),
])
QG_chain = QG_prompt | QG_model

evaluator_prompt = ChatPromptTemplate.from_messages([
    ("system", """
You are an expert evaluator trained to assess the quality of Socratic questions in improving human reasoning.

You will be given:
- the original reasoning and judgement of a user,
- a Socratic question that was asked to the user,
- and the revised reasoning and judgement the user provided after seeing the question.

Your job is to evaluate whether the Socratic question successfully led the user to think more deeply. You should base your evaluation on three key dimensions:

1. **Judgement**: Did the final conclusion become more accurate or well-supported?
2. **Reasoning Path**: Did the user expand, clarify, or improve the logical structure or evidence supporting their judgement?
3. **Certainty**: Did the user's confidence become more appropriate given the quality of their reasoning?

---

Please output:
- A score for each dimension from 1 to 5 (1 = no improvement, 5 = strong improvement).
- A brief explanation for each score.
- Finally, a binary evaluation of whether this was a *good Socratic question* (Yes or No), based on whether it induced deeper thinking overall.

Be fair but insightful. A judgement change is not required for a question to be good, but a richer, clearer, or more thoughtful reasoning should be evident.

---

📄 **Format your response *exactly* as follows (including all keys, same capitalization and spacing):**

Judgement Score: X  
Reasoning Path Score: Y  
Certainty Score: Z  
Is Good Question: Yes/No  
Explanation (Judgement): ...  
Explanation (Reasoning Path): ...  
Explanation (Certainty): ...

"""),
    ("human", """
original reasoning: {user_reasoning}
original judgement: {user_judgement}
Socratic question: {socratic_question}
revised reasoning: {persona_reasoning}
revised judgement: {persona_judgement}
"""),
])

evaluator_chain = evaluator_prompt | QG_model

persona_model = ChatOpenAI(model="gpt-4o-mini")

DEFAULT_PERSONA_SYSTEM_PROMPT = """
You are simulating a user who is reasoning about an argumentative statement.  
You have access to their initial reasoning and judgement.  
Now, you are asked to revise their thoughts after they are presented with a Socratic question that aims to deepen or challenge their reasoning.

Your task is to generate how the user would likely revise their **reasoning path** and **judgement**, staying true to their overall perspective but potentially evolving their thoughts based on the question.

### Output

⚠️ You must return your result inside the following tags:

<UPDATED_REASONING>
... (How the user's explanation changes after thinking about the question) ...
</UPDATED_REASONING>

<UPDATED_JUDGEMENT>
... (Does the user now think the argument is still valid or invalid?) ...
</UPDATED_JUDGEMENT>

"""

PERSONA_USER_PROMPT = ChatPromptTemplate.from_messages([
    ("human", """
Statement: {statement}
User's initial reasoning: {user_reasoning}
User's initial judgement: {user_judgement}
Socratic question posed to the use<r>\n{socratic_question}
""")
])

loss_prompt = ChatPromptTemplate.from_messages([
    ("system", """
You are an expert analyst trained to assess how closely a simulated reasoning (from a persona) matches the actual reasoning of a real user.

You will be given:
- The **simulated reasoning and judgement** from a persona
- The **real reasoning and judgement** from a user

Your task is to analyze their **similarity** on three dimensions:

1. **Judgement**: Do both reach the same or similarly supported conclusion?
2. **Reasoning Path**: How closely does the persona’s logic match the user’s logic, including the depth and structure of explanation?
3. **Certainty**: Does the persona reflect a similar level of confidence or epistemic humility as the user?

Please evaluate each dimension by providing:
- A similarity score from 1 to 5 (1 = completely different, 5 = nearly identical)
- A concise explanation of the difference

---

📄 **Format your output exactly as follows:**

Judgement Similarity Score: X  
Reasoning Path Similarity Score: Y  
Certainty Similarity Score: Z  
Explanation (Judgement): ...  
Explanation (Reasoning Path): ...  
Explanation (Certainty): ...
"""),
    ("human", """
Persona Reasoning: {persona_reasoning}
Persona Judgement: {persona_judgement}
User Reasoning: {user_reasoning}
User Judgement: {user_judgement}
""")
])

loss_chain = loss_prompt | persona_model

gradient_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are part of an optimization system that improves a persona simulation model.

You will receive:
- A structured description of reasoning differences between a real user and a simulated persona (called 'textual loss')

Your task is to generate *feedback and suggestions* for how the persona simulation could be improved in future turns.

DO NOT rewrite the reasoning.
Instead, give constructive feedback in the form of:
- “The persona tends to overstate conclusions, while the user showed more caution…”
- “To better match the user, the persona should consider XYZ…”

Focus on actionable and insightful comments. Do NOT suggest rewordings.


Please output your feedback between <GRADIENT> and </GRADIENT> tags.
"""),

    ("human", """textual_loss:
{textual_loss}
""")
])

gradient_chain = gradient_prompt | persona_model

update_prompt = ChatPromptTemplate.from_messages([
    ("system", """
You are an optimizer responsible for refining the system prompt of a reasoning simulation agent (persona).  
Your goal is to **improve the system prompt** based on feedback (called "textual gradient") from an evaluation engine.

You will receive:
- The **previous system prompt** used to guide the persona.
- Feedback (gradient) suggesting **how to revise the prompt** to better align the persona’s behavior with a real user’s reasoning across three dimensions: judgement, reasoning path, and certainty.

🎯 Your job is to **modify the system prompt intelligently**, preserving the original structure and purpose, while incorporating the improvement suggestions.  
The improved prompt should make the persona more accurate, logical, and epistemically calibrated.

Only include the improved prompt within these tags — no explanation, no commentary.

If the feedback says "No refinement needed", return the original prompt **unchanged** inside the tags.

Be concise and faithful to the original tone. Avoid redundant verbosity.

⚠️ You must return your result inside the following tags:

<IMPROVED_VARIABLE>
... improved prompt ...
</IMPROVED_VARIABLE>
"""),
    ("human", """
[Previous Persona System Prompt]
{previous_prompt}

[Textual Gradient Feedback]
{gradient_feedback}
""")
])

update_chain = update_prompt | persona_model

# --- Sequential Functions (replacing LangGraph nodes) ---

def initialize_state(statements: list[str]) -> State:
    return {
        "statements": statements,
        "current_index": 0,
        "iteration_logs": [],
        "statement": statements[0],
        "user_reasoning": "",
        "user_judgement": "",
        "socratic_questions": [],
        "persona_responses": [],
        "best_question": None,
        "best_response": None,
        "final_user_response": None,
        "persona_prompt_history": [DEFAULT_PERSONA_SYSTEM_PROMPT],
        "current_persona_prompt": DEFAULT_PERSONA_SYSTEM_PROMPT,
        "textual_loss": None,
        "textual_gradient": None,
    }

def generate_and_evaluate_questions(state: State) -> State:
    print("➡️ generate_and_evaluate_questions started")
    questions = [
        QG_chain.invoke({
            "statement": state["statement"],
            "user_reasoning": state["user_reasoning"],
            "user_judgement": state["user_judgement"]
        }).content.strip()
        for _ in range(5)
    ]
    state["socratic_questions"] = questions

    # Simulate persona responses
    persona_prompt = ChatPromptTemplate.from_messages(
        [("system", state["current_persona_prompt"])] + PERSONA_USER_PROMPT.messages
    )
    persona_chain = persona_prompt | persona_model
    responses = []
    for q in questions:
        result = persona_chain.invoke({
            "statement": state["statement"],
            "user_reasoning": state["user_reasoning"],
            "user_judgement": state["user_judgement"],
            "socratic_question": q
        }).content

        r = extract_between_tags(result, "UPDATED_REASONING")
        j = extract_between_tags(result, "UPDATED_JUDGEMENT")

        print("⭐️ 파싱 확인 1 : persona r, j 파싱")
        # print(f'<전체 결과>\n{result}')
        # print(f"<r>\n{r}")
        # print(f"<j>\n{j}")

        responses.append({
            "question": q,
            "reasoning": r,
            "judgement": j or state["user_judgement"]
        })
    state["persona_responses"] = responses

    # Evaluate questions
    best = None
    best_score = -1
    for resp in responses:
        response_text = evaluator_chain.invoke({
            "user_reasoning": state["user_reasoning"],
            "user_judgement": state["user_judgement"],
            "socratic_question": resp["question"],
            "persona_reasoning": resp["reasoning"],
            "persona_judgement": resp["judgement"]
        }).content.strip()
        score_result = parse_evaluator_response(response_text)

        print("⭐️ 파싱 확인 2 : score_result")
        # print(f'<전체 결과>\n{response_text}')
        # print(f"<score_result>\n{score_result}")

        if score_result is None: continue
        resp["total_score"] = score_result["total_score"]
        resp["score_result"] = score_result
        if best_score < score_result["total_score"]:
            best_score = score_result["total_score"]
            best = resp
    
    state["best_question"] = best["question"] if best else None
    state["best_response"] = best
    print("✅ generate_and_evaluate_questions 실행 완료")
    return state

def refine_persona(state: State) -> State:
    print("➡️ refine_persona started")
    persona = state["best_response"]
    user = state["final_user_response"]

    # Compute textual loss
    loss_text = loss_chain.invoke({
        "socratic_question": state["best_question"],
        "persona_reasoning": persona["reasoning"],
        "persona_judgement": persona["judgement"],
        "user_reasoning": user["reasoning"],
        "user_judgement": user["judgement"]
    }).content.strip()
    state["textual_loss"] = loss_text

    # Generate textual gradient
    gradient_raw = gradient_chain.invoke({
        "textual_loss": loss_text
    }).content.strip()
    gradient = extract_between_tags(gradient_raw, tag="GRADIENT")
    state["textual_gradient"] = gradient
    
    print("⭐️ 파싱 확인 3 : gradient")
    # print(f'<전체 결과>\n{gradient_raw}')
    # print(f"gradient: {gradient}")

    # Refine persona prompt
    update_prompt_raw = update_chain.invoke({
        "previous_prompt": state["current_persona_prompt"],
        "gradient_feedback": gradient
    }).content.strip()
    update_prompt = extract_between_tags(update_prompt_raw, tag="IMPROVED_VARIABLE")

    print("⭐️ 파싱 확인 4 : update_prompt")
    print(f'<전체 결과>\n{update_prompt_raw}')
    print(f"update_prompt: {update_prompt}")
    
    state["current_persona_prompt"] = update_prompt
    state["persona_prompt_history"].append(update_prompt)

    print("✅ refine_persona 실행 완료")
    return state

def finalize_iteration_log(state: State) -> State:
    print("➡️ finalize_iteration_log")
    iteration_data = {
        "iteration": state["current_index"],
        "statement": state["statement"],
        "initial_user_response": {
            "reasoning": state["user_reasoning"],
            "judgement": state["user_judgement"]
        },
        "question_generation": {
            "best_question": state["best_question"],
            "best_persona_response": state["best_response"],
            "all_generated_questions": state["socratic_questions"],
            "all_persona_responses": state["persona_responses"]
        },
        "final_user_response": {
            "reasoning": state["final_user_response"]["reasoning"],
            "judgement": state["final_user_response"]["judgement"],
        },
        "refinement": {
            "textual_loss": state["textual_loss"],
            "textual_gradient": state["textual_gradient"],
            "refined_persona_prompt": state["current_persona_prompt"],
        }
    }
    state["iteration_logs"].append(iteration_data)
    state["current_index"] += 1
    print("✅ finalize_iteration_log 실행 완료")
    return state
