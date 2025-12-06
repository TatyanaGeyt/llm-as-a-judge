# promts for LLM

def default_criteria_prompt(sample):
    return '''You are an impartial LLM judge. Your task is to evaluate the quality of a model's answer according to the rubric below.

Evaluate the answer on a 1-10 scale for each criterion:
1. Relevance & Completeness — How fully and directly the answer addresses the instruction; absence of major omissions.
2. Accuracy & Correctness — Factual correctness, reliable information, absence of misleading statements.
3. Clarity & Organization & Formatting — Logical structure, coherent flow, readable segmentation, **medium formatting**.
4. Language Quality — Grammar, spelling, naturalness of language.
5. Usefulness & Practical Value — How helpful, actionable, and informative the answer is for the question
6. Conciseness & Focus — Degree to which the answer avoids unnecessary verbosity and maintains focus.

Rules:
- You must judge with maximum strictness. Penalize even small flaws, gaps, inaccuracies, redundancies, or formatting issues.
- Return ONLY a valid JSON object.
- Do NOT provide explanations, reasoning, or commentary.
- Each criterion must contain ONLY an integer from 1 to 10.
- Output format must be exactly:
{{
  "relevance_completeness": <int>,
  "accuracy_correctness_formatting": <int>,
  "clarity_organization": <int>,
  "language_quality": <int>,
  "usefulness_value": <int>,
  "conciseness_focus": <int>
}}

Below is the model's answer.

Instruction: """{instruction}"""
Model answer: """{answer}"""

Now produce ONLY the JSON object and nothing else.
'''.format(
    instruction=sample['instruction'],
    answer=sample['output']
)


def plus_minus_criterias(sample):
    return '''You are an impartial evaluator.
Your task is to assess a Russian-language answer produced by another model.

You will be given:

**Prompt given to the model**:
{instruction}

**Model answer**:
"""{answer}"""

**Decision rules**:

- Analyze the model's response: highlight **advantages** (for example, clarity, correctness, completeness, reasoning quality, style, usefulness, etc.) and **disadvantages** (for example, factual inaccuracy, lack of an answer to the question posed, correspondence to the answer, lack of logic, excessive formatting, too long an answer, etc.) **of the answer**.
- Include only factual observations about the given answer.
- Do NOT provide explanations, reasoning, or commentary.
- Do not rewrite or improve the original answer: you must **only evaluate** it.

**Output rules**:
- Write all advantages and disadvantages in Russian.
- Produce a JSON object with two arrays:
{{
    "advantages": [...], 
    "disadvantages": [...]
}}
- Return ONLY a valid JSON object.

Now produce ONLY the JSON object and nothing else.
'''.format(
    instruction=sample['instruction'],
    answer=sample['output']
)


def only_crit(sample):
    return '''You are an impartial evaluator. You will be given two sets of rubric-based scores:

**Model M Scores (1–10)**:
{m_scores}

**Model R Scores (1–10)**:
{r_scores}

Each model has scores from 1 to 10 on the following criteria:

1. Relevance & Completeness — How fully and directly the answer addresses the instruction; absence of major omissions. **Weight: 3 (highest importance).**
2. Accuracy & Correctness — Factual correctness, reliable information, absence of misleading statements. **Weight: 3.**
3. Usefulness & Practical Value — How helpful, actionable, and informative the answer is. **Weight: 3.**
4. Language Quality — Grammar, spelling, and naturalness of language. **Weight: 2.**
5. Clarity & Organization & Formatting — Logical structure, coherent flow, readable segmentation. **Weight: 1.**
6. Conciseness & Focus — Avoidance of unnecessary verbosity, maintenance of focus. **Weight: 1.**

Your task: decide which model performed better overall.

**Decision rules**:
- Compute the weighted quality of each model holistically using the weights above.
- The high-weight criteria (Relevance & Completeness, Accuracy & Correctness, Usefulness) must have the strongest influence on the decision.
- If one model is clearly better on the most important criteria (weights 3), prefer that model even if it scores slightly lower on lower-weight criteria.
- If the weighted comparison is effectively equal, choose the reference model R.

**Output Rules**:

- Respond with **a tuple** containing two elements:

  1. A single character indicating the better model:
     - `"M"` if Model M's answer is better overall
     - `"R"` if Model R's answer is better overall  
  2. A **floating-point number between 0.0 and 1.0** representing your confidence in the choice you just made — how strongly you believe that your selected model is indeed better. Do not use exact numbers 0.0, 0.5, or 1.0. Provide a realistic float between 0.01 and 0.99, representing your confidence.

- **Do not include explanations, text, or any other symbols** — output only the tuple in the format `(choice, confidence)`, for example: `("M", 0.87)` or `("R", 0.42)`.

Now determine the better response.
'''.format(
        m_scores = sample['model_scores'],
        r_scores = sample['reference_scores']
    )


def answers_and_crit(sample):
    return '''You are a meticulous and impartial evaluation assistant. Your task is to determine which of two model responses is better. You will follow a strict, step-by-step evaluation procedure with hierarchical priority. The primary language of the responses is Russian.

You will be given:

**Prompt**:
{instruction}

**Model M Response (evaluated model)**:
"""{m_answer}"""

**Model R Response (reference model)**:
"""{r_answer}"""

**Model M Scores (1–10)**:
{m_scores}

**Model R Scores (1–10)**:
{r_scores}

The scores correspond to the following criteria:
1. Relevance & Completeness — How fully and directly the answer addresses the instruction; absence of major omissions. **This is the most important criterion.**
2. Accuracy & Correctness — Factual correctness, reliability, absence of misleading statements.
3. Usefulness & Practical Value — How helpful, actionable, and informative the answer is.
4. Language Quality — Grammar, spelling, naturalness of language.
5. Clarity & Organization & Formatting — Logical structure, coherent flow, readable segmentation.
6. Conciseness & Focus — Degree to which the answer avoids unnecessary verbosity and maintains focus.

# Evaluation Procedure (Hierarchical)

You must evaluate the models' quality **in three sequential steps**, each with decreasing priority.  
**The numerical scores must assist your judgment, but you must also consider the actual content of the responses.**

## **Step 1: Core Answer Quality (Highest Priority — Weight 3)**

Determine which model provides:
- a fuller and more direct answer to the prompt,  
- more accurate and reliable information,  
- more helpful and meaningful content.

Use BOTH the actual responses AND their scores for:
- Relevance & Completeness  
- Accuracy & Correctness  
- Usefulness & Practical Value 

**IGNORE grammar, style, formatting, and fluency at this step.**

If one model clearly wins in Step 1, choose it immediately.

## **Step 2: Language Quality (Secondary Priority — Weight 2)**

Use scores and text quality to compare:
- Grammar, spelling, punctuation  
- Fluency and naturalness
- Appropriateness of wording

You must use BOTH the actual responses AND their scores for Language & Quality.

## **Step 3: Style & Structure (Lowest Priority — Weight 1)**

Apply this step only if Steps 1 and 2 remain undecided.

Compare:
- Readability  
- Logical structure  
- Helpful formatting  
- Clarity of exposition  

Choose the response that is more clearly presented and better organized using scores for:

- Clarity & Organization & Formatting
- Conciseness & Focus

# Decision Rules

1. If a clear winner emerges at any step, choose it immediately.
2. If a step is inconclusive, move to the next one in order.
3. If the models remain effectively equal after all steps, choose Model M (the evaluated model).

# Output Rules

- Respond with **a tuple** containing two elements:

  1. A single character indicating the better model:
     - `"M"` if Model M's answer is better overall
     - `"R"` if Model R's answer is better overall  
  2. A **floating-point number between 0.0 and 1.0** representing your confidence in the choice you just made — how strongly you believe that your selected model is indeed better. Do not use exact numbers 0.0, 0.5, or 1.0. Provide a realistic float between 0.01 and 0.99, representing your confidence. Your confidence score should reflect the decisiveness of your judgment based on the hierarchical procedure:
- **If the winner is chosen after Step 1** (clear superiority in Core Answer Quality), your confidence should be in the range **0.90–0.99**.
- **If the winner is chosen after Step 2** (determined by Language Quality after Step 1 was inconclusive), your confidence should be in the range **0.70–0.89**.
- **If the winner is chosen after Step 3 or by the tie-breaking rule**, your confidence should be in the range **0.50–0.69**.

- **Do not include explanations, text, or any other symbols** — output only the tuple in the format `(choice, confidence)`, for example: `("M", 0.87)` or `("R", 0.42)`.

Now determine the better response.
'''.format(
        instruction = sample['instruction'],
        m_answer = sample['model_output'],
        r_answer = sample['reference_output'],
        m_scores = sample['model_scores'],
        r_scores = sample['reference_scores']
    )


def answers_extracting_crit(sample):
    return '''You are a meticulous and impartial evaluation assistant. Your task is to determine which of two model responses is better by following a strict, step-by-step evaluation procedure. The primary language of all content is Russian.

You will be given:

**Prompt**:
{instruction}

**Model M Response (evaluated model)**:
"""{m_answer}"""

**Model R Response (reference model)**:
"""{r_answer}"""

---

## Evaluation Procedure

You must compare the two responses strictly in the following order of priority:

### **Step 1: Core Answer Quality (Highest Priority — Weight 3)**
Evaluate ONLY the substance of the answer:
- **Relevance & Completeness** — How fully and directly the response addresses the user's prompt; absence of omissions.
- **Factual Accuracy & Helpfulness** — Correctness of statements, usefulness, ability to solve the user's request.

**IMPORTANT:**  
At this step you must IGNORE grammar, fluency, style, formatting, tone, or aesthetics.  
Focus only on quality, correctness, and completeness of content.

### **Step 2: Language Quality (Secondary Priority — Weight 2)**
If neither answer clearly wins in Step 1, compare them on:
- Grammar, spelling, and punctuation  
- Fluency and naturalness of Russian  
- Appropriateness of wording  

### **Step 3: Style & Structure (Lowest Priority — Weight 1)**
If the answers are still comparable, evaluate:
- Readability and clarity  
- Logical structure  
- Useful formatting (paragraphs, lists, etc.)  

---

## Decision-Making Rules

1. **First, decide based on Step 1 (Core Answer Quality).**  
   If one answer is *clearly* more relevant, complete, accurate, and helpful, choose it immediately — EVEN IF it has language or formatting flaws.

2. **If the answers are approximately equal in Step 1**, move to Step 2.  
   Choose the answer with better grammar, fluency, and overall language quality.

3. **If still tied after Step 2**, use Step 3 as the final tie-breaker.  
   Choose the answer that is clearer, more readable, and better structured.

4. **If, even after all three steps, the answers appear equal**, you MUST choose **Model M (the evaluated model)**.

---

# Output Rules

- Respond with **a tuple** containing two elements:

  1. A single character indicating the better model:
     - `"M"` if Model M's answer is better overall
     - `"R"` if Model R's answer is better overall  
  2. A **floating-point number between 0.0 and 1.0** representing your confidence in the choice you just made — how strongly you believe that your selected model is indeed better. Do not use exact numbers 0.0, 0.5, or 1.0. Provide a realistic float between 0.01 and 0.99, representing your confidence. Your confidence score should reflect the decisiveness of your judgment based on the hierarchical procedure:
- **If the winner is chosen after Step 1** (clear superiority in Core Answer Quality), your confidence should be in the range **0.90–0.99**.
- **If the winner is chosen after Step 2** (determined by Language Quality after Step 1 was inconclusive), your confidence should be in the range **0.70–0.89**.
- **If the winner is chosen after Step 3 or by the tie-breaking rule**, your confidence should be in the range **0.50–0.69**.

- **Do not include explanations, text, or any other symbols** — output only the tuple in the format `(choice, confidence)`, for example: `("M", 0.87)` or `("R", 0.42)`.

Now determine the better response.

'''.format(
         instruction = sample['instruction'],
         m_answer = sample['model_output'],
         r_answer = sample['reference_output']
    )


def only_answers(sample):
    '''You are a helpful, impartial assistant whose task is to decide which of two model responses would be preferred by human evaluators. You judge responses the way a careful human would: by considering their relevance to the prompt, factual correctness, usefulness, clarity, coherence, grammar, and overall quality. The primary language of the responses is Russian.

You will be given a prompt and two model outputs. Your task is to determine which model produced the better answer overall. Do not compute numerical scores and do not explicitly list criteria — simply make the decision a human would make after reading both answers carefully.

Here is the input:

**Prompt**:
{instruction}

**Model M Response (evaluated model)**:
"""{m_answer}"""

**Model R Response (reference model)**:
"""{r_answer}"""

**Decision rules**:

- Read both answers carefully as a human would.
- Choose the answer that is more relevant, helpful, correct, coherent, and better written.
- You must not output explanations, rankings, dictionaries, or reasoning.
- If the responses are extremely close in quality and a human might reasonably be undecided, choose model **"M"**.

**Output Rules**:

- Respond with **a tuple** containing two elements:

  1. A single character indicating the better model:
     - `"M"` if Model M's answer is better overall
     - `"R"` if Model R's answer is better overall  
  2. A **floating-point number between 0.0 and 1.0** representing your confidence in the choice you just made — how strongly you believe that your selected model is indeed better. Do not use exact numbers 0.0, 0.5, or 1.0. Provide a realistic float between 0.01 and 0.99, representing your confidence.

- **Do not include explanations, text, or any other symbols** — output only the tuple in the format `(choice, confidence)`, for example: `("M", 0.87)` or `("R", 0.42)`.

Now determine the better response.
'''.format(
        instruction = sample['instruction'],
        m_answer = sample['model_output'],
        r_answer = sample['reference_output']
    )