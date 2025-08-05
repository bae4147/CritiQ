# graph_prototype_v1.py

import os
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatHuggingFace
from dotenv import load_dotenv

load_dotenv()

# LangSmith logging (optional)
from langchain_teddynote import logging
logging.langsmith("SocratiQ_demo")

##### STEP 1: Define State #####
class State(TypedDict):
    statement: str
    user_reasoning: str
    user_judgement: str
    socratic_questions: list[str]
    persona_responses: list[dict]
    best_question: str
    best_response: dict 
    final_user_response: dict
    # refinement_data: list[dict]
    messages: Annotated[list, add_messages]
    persona_prompt_history: list[str]  # iteration마다 사용된 system prompt 저장
    current_persona_prompt: str        # 항상 현재 prompt를 가리킴
    textual_loss: str
    textual_gradient: str
    statements: list[str]
    current_index: int
    iteration_logs: list[dict]
    user_reasonings: list[str]
    user_judgements: list[str]

    
def prepare_next_iteration(state):
    i = state["current_index"]
    state.update({
        "statement": state["statements"][i],
        "user_reasoning": state["user_reasonings"][i],
        "user_judgement": state["user_judgements"][i],
        "socratic_questions": [],
        "persona_responses": [],
        "best_question": None,
        "best_response": None,
        "final_user_response": None,
        # "refinement_data": [],
        "iteration_logs": [],
        "textual_loss": None,
        "textual_gradient": None,
        "iteration": i,
        "current_persona_prompt": DEFAULT_PERSONA_SYSTEM_PROMPT,
        "persona_prompt_history": [DEFAULT_PERSONA_SYSTEM_PROMPT]
    })
    return state


##### STEP 2: Define Models #####
QG_model = ChatOpenAI(model="gpt-4o-mini", temperature=0.9)  # diversified output

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

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
from langchain_huggingface import HuggingFacePipeline
# Persona Model (Hugging Face)
# model_id = "meta-llama/Llama-3.1-8B"  # 원하는 모델로 교체 가능 openai-community/gpt2-large meta-llama/Llama-3.1-8B

# tokenizer = AutoTokenizer.from_pretrained(model_id)
# tokenizer.pad_token = tokenizer.eos_token

# model = AutoModelForCausalLM.from_pretrained(
#     model_id,
#     device_map="auto",
#     low_cpu_mem_usage=True,
#     quantization_config=BitsAndBytesConfig(load_in_4bit=True)
#     )

# pipe = pipeline(
#     "text-generation", # "question-answering" https://huggingface.co/docs/transformers/v4.53.2/en/main_classes/pipelines#transformers.QuestionAnsweringPipeline
#     model=model,
#     # torch_dtype=torch.float16,
#     tokenizer=tokenizer,
#     max_new_tokens=512,
#     do_sample=False,
#     temperature=None,   # ⚠️ 이렇게 명시적으로 제거
#     top_p=None
# )

# persona_model = HuggingFacePipeline(pipeline=pipe)
persona_model = ChatOpenAI(model="gpt-4o-mini")

DEFAULT_PERSONA_SYSTEM_PROMPT = """
You are simulating a user who is reasoning about an argumentative statement.  
You have access to their initial reasoning and judgement.  
Now, you are asked to revise their thoughts after they are presented with a Socratic question that aims to deepen or challenge their reasoning.

Your task is to generate how the user would likely revise their **reasoning path** and **judgement**, staying true to their overall perspective but potentially evolving their thoughts based on the question.

### Output

You must output the user's likely updated thoughts as two parts:

1. **Revised Reasoning Path**  
(How the user's explanation changes after thinking about the question)

2. **Updated Judgement**  
(Does the user now think the argument is still valid or invalid?)

"""

PERSONA_USER_PROMPT = ChatPromptTemplate.from_messages([
    ("human", """
Statement: {statement}
User's initial reasoning: {user_reasoning}
User's initial judgement: {user_judgement}
Socratic question posed to the user: {socratic_question}
""")
])

### CHAINS FOR PERSONA REFINEMENT ###
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



import re

def extract_after_header(text, header="Gradient:"):
    parts = text.split(header, maxsplit=1)
    return parts[1].strip() if len(parts) > 1 else text.strip()

def extract_between_tags(text, tag="IMPROVED_VARIABLE"):
    pattern = rf"<{tag}>\s*(.*?)\s*</{tag}>"
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1).strip() if match else text.strip()

def parse_persona_output(text: str) -> tuple[str, str]:
    """
    Parse the persona model output to extract:
    - Revised Reasoning Path
    - Updated Judgement

    Returns:
        (reasoning: str, judgement: str)
    """
    # 정규식 기반으로 Revised Reasoning Path와 Updated Judgement 추출
    reasoning_match = re.search(
        r"Revised Reasoning Path:\s*(.*?)(?=Updated Judgement:|$)", 
        text, 
        re.DOTALL
    )
    judgement_match = re.search(
        r"Updated Judgement:\s*(.*)", 
        text, 
        re.DOTALL
    )

    reasoning = reasoning_match.group(1).strip() if reasoning_match else ""
    judgement = judgement_match.group(1).strip() if judgement_match else ""

    return reasoning, judgement


def parse_evaluator_response(response_text):
    try:
        # 점수 추출
        judgement_score = int(re.search(r"Judgement Score\s*:\s*(\d)", response_text).group(1))
        reasoning_score = int(re.search(r"Reasoning Path Score\s*:\s*(\d)", response_text).group(1))
        certainty_score = int(re.search(r"Certainty Score\s*:\s*(\d)", response_text).group(1))

        # Yes/No 추출
        is_good_question = re.search(r"Is Good Question\s*:\s*(Yes|No)", response_text, re.IGNORECASE).group(1)

        # 각 explanation 추출
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


##### STEP 3: Node Functions #####

def should_continue(state):
    return "continue" if state["current_index"] < len(state["statements"]) else "end"

def initialize_state(statements: list[str], reasonings: list[str], judgements: list[str]) -> State:
    return {
        "statements": statements, # "I think it is valid. because a factor of 20 is a large gap between the two groups.", # I think violent video games can make people more aggressive. In this case, someone stabbed another person after losing in a game, so the argument seems valid. I've also seen news stories where people who play violent games act violently in real life, so it feels like the conclusion makes sense.
        "user_reasonings": reasonings, # "In the United States, racial stratification still occurs. The racial wealth gap between African Americans and White Americans for the same job is found to be a factor of twenty.", # Violent video games causes people to be aggressive in the real world. A gamer stabbed another after being beaten in the online game Counter-Strike.
        "user_judgements": judgements,
        "current_index": 0,
        "iteration_logs": [],
        # 첫 statement를 위한 초기화도 바로 세팅
        "statement": statements[0],
        "user_reasoning": reasonings[0],
        "user_judgement": judgements[0],
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

def generate_question(state: State):
    questions = [
        QG_chain.invoke({
            "statement": state["statement"],
            "user_reasoning": state["user_reasoning"],
            "user_judgement": state["user_judgement"]
        }).content.strip()
        for _ in range(5)
    ]
    state["socratic_questions"] = questions
    return state

def simulate_persona_response(state: State):

    persona_prompt = ChatPromptTemplate.from_messages(
        [("system", state["current_persona_prompt"])] + PERSONA_USER_PROMPT.messages
    )

    persona_chain = persona_prompt | persona_model

    responses = []
    for q in state["socratic_questions"]:
        result = persona_chain.invoke({
            "statement": state["statement"],
            "user_reasoning": state["user_reasoning"],
            "user_judgement": state["user_judgement"],
            "socratic_question": q
        }).content
        # })

        r, j = parse_persona_output(result)

        responses.append({
            "question": q,
            "reasoning": r,
            "judgement": j or state["user_judgement"]
        })

    state["persona_responses"] = responses
    return state

def evaluate_question(state: State):
    
    # use judgement change as a proxy for impact
    best = None
    best_score = -1
    for resp in state["persona_responses"]:
        response_text = evaluator_chain.invoke({
            "user_reasoning": state["user_reasoning"],
            "user_judgement": state["user_judgement"],
            "socratic_question": resp["question"],
            "persona_reasoning": resp["reasoning"],
            "persona_judgement": resp["judgement"]
        }).content.strip()


        score_result = parse_evaluator_response(response_text)

        resp["total_score"] = score_result["total_score"] # state["persona_responses"][i]["total_score"]
        resp["score_result"] = score_result
        
        if best_score < score_result["total_score"]:
            best_score = score_result["total_score"]
            best = resp

    state["best_question"] = best["question"]
    state["best_response"] = best
    return state

def ask_user_response(state: State):
    print("\n[User] Please respond to the Socratic Question:")
    print("Statement:", state["statement"])
    print("Question:", state["best_question"])
    new_reasoning = input("Updated Reasoning: ")
    new_judgement = input("Updated Judgement (valid/invalid): ")
    state["final_user_response"] = {
        "reasoning": new_reasoning,
        "judgement": new_judgement
    }
    return state

def compute_textual_loss(state: State):
    persona = state["best_response"]
    user = state["final_user_response"]

    loss_text = loss_chain.invoke({
        "socratic_question": state["best_question"],
        "persona_reasoning": persona["reasoning"],
        "persona_judgement": persona["judgement"],
        "user_reasoning": user["reasoning"],
        "user_judgement": user["judgement"]
    }).content.strip()
    # })

    state["textual_loss"] = loss_text
    return state

def generate_textual_gradient(state: State):
    gradient_raw = gradient_chain.invoke({
        "textual_loss": state["textual_loss"]
    }).content.strip()
    # })

    gradient = extract_between_tags(gradient_raw, tag="GRADIENT")

    state["textual_gradient"] = gradient
    return state

def refine_persona_prompt(state: State):
    update_prompt_raw = update_chain.invoke({
        "previous_prompt": state["current_persona_prompt"],
        "gradient_feedback": state["textual_gradient"]
    }).content.strip()
    # })

    update_prompt = extract_between_tags(update_prompt_raw, tag="IMPROVED_VARIABLE")

    state["current_persona_prompt"] = update_prompt
    return state

def finalize_iteration_log(state):
    iteration_data = {
        "iteration": state["current_index"],
        "statement": state["statement"],
        "question_generation": {
            "questions": state["socratic_questions"],
            "responses": state["persona_responses"],
            "best_question": state["best_question"],
            "best_response": state["best_response"],
        },
        "final_user_response": {
            "reasoning": state["final_user_response"]["reasoning"],
            "judgement": state["final_user_response"]["judgement"],
        },
        "refinement": {
            "textual_loss": state["textual_loss"],
            "textual_gradient": state["textual_gradient"],
            "update_prompt": state["current_persona_prompt"],
        }
    }
    
    state["iteration_logs"].append(iteration_data)
    state["current_index"] += 1
    return state


##### STEP 4: Build Graph #####
# 1. 그래프 생성
workflow = StateGraph(State)

# 2. 노드 등록
# workflow.add_node("initialize_state", initialize_state)  # optional if used externally
workflow.add_node("prepare_next_iteration", prepare_next_iteration)
workflow.add_node("generate_question", generate_question)
workflow.add_node("simulate_persona_response", simulate_persona_response)
workflow.add_node("evaluate_question", evaluate_question)
workflow.add_node("ask_user_response", ask_user_response)
workflow.add_node("compute_textual_loss", compute_textual_loss)
workflow.add_node("generate_textual_gradient", generate_textual_gradient)
workflow.add_node("refine_persona_prompt", refine_persona_prompt)
workflow.add_node("finalize_iteration_log", finalize_iteration_log)

# 3. entry point
workflow.set_entry_point("generate_question")  # initialize_state is external

# 4. 노드 간 연결
workflow.add_edge("prepare_next_iteration", "generate_question")
workflow.add_edge("generate_question", "simulate_persona_response")
workflow.add_edge("simulate_persona_response", "evaluate_question")
workflow.add_edge("evaluate_question", "ask_user_response")
workflow.add_edge("ask_user_response", "compute_textual_loss")
workflow.add_edge("compute_textual_loss", "generate_textual_gradient")
workflow.add_edge("generate_textual_gradient", "refine_persona_prompt")
workflow.add_edge("refine_persona_prompt", "finalize_iteration_log")

# 5. 반복 여부 조건 분기
workflow.add_conditional_edges(
    "finalize_iteration_log",
    should_continue,      # -> returns "continue" or "end"
    # 조건에 따라 다음 흐름 결정
    {
        "continue": "prepare_next_iteration",  # 다음 statement 반복
        "end": END                        # 종료
    }
)

# 6. 그래프 빌드
graph = workflow.compile()

from langchain_teddynote.graphs import visualize_graph

# 그래프 시각화
visualize_graph(graph)

# 7. 실행
statements = ["In the United States, racial stratification still occurs. The racial wealth gap between African Americans and White Americans for the same job is found to be a factor of twenty.", "Violent video games causes people to be aggressive in the real world. A gamer stabbed another after being beaten in the online game Counter-Strike."]
reasonings = ["I think it is valid. because a factor of 20 is a large gap between the two groups.", "I think violent video games can make people more aggressive. In this case, someone stabbed another person after losing in a game, so the argument seems valid. I've also seen news stories where people who play violent games act violently in real life, so it feels like the conclusion makes sense."]
judgements = ["valid", "invalid"]
initial_state = initialize_state(statements, reasonings, judgements)
final_state = graph.invoke(initial_state)