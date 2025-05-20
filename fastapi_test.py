from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse
from typing import Optional
import io
from PIL import Image
from langgraph.graph import StateGraph, END
from typing import Optional, List, TypedDict
import json
import google.generativeai as genai
from pydantic import BaseModel  # pydantic v2 compatible
import os
from dotenv import load_dotenv
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse
from typing import Optional
import io
from PIL import Image
from langgraph.graph import StateGraph, END
from typing import Optional, List, TypedDict
import json
import google.generativeai as genai
from pydantic import BaseModel  # pydantic v2 compatible
import os
from dotenv import load_dotenv



# Load environment variables from .env file
load_dotenv()

# Retrieve the Gemini API key
gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    gemini_api_key = "YOUR_GEMINI_API_KEY"  # Replace with your actual API key or environment variable

# Configure Google Gemini API
genai.configure(api_key=gemini_api_key)

# 1. Define the State for the Chat Workflow
class ChatState(TypedDict):
    """
    Represents the state of the conversation at any point in the workflow.
    """
    image: Optional[bytes]  # Optional image data (for issue detection)
    message: str  # The current user message
    location: Optional[str]  # Optional user location for context
    stage: Optional[str]  # Current stage of the conversation (not actively used)
    category: Optional[str]  # Predicted category of the user's query ("skin_care" or "nutrition")
    response: Optional[str]  # The AI's response to the user
    conversation_history: List[dict]  # History of the conversation (not actively used)
    intermediate_responses: List[dict]  # List of responses from intermediate steps (now dicts)
    requires_human_input: Optional[bool]  # Flag indicating if human input is needed
    clarification_question: Optional[str]  # The question to ask the user for clarification
    clarification_attempts: int  # Counter for the number of clarification attempts
    exit_requested: Optional[bool]
    agent_1_output: Optional[str]
    agent_2_output: Optional[str]
    aggregated_response: Optional[str]

# 2. Node 1: Classifier Agent
def classifier(state: ChatState) -> ChatState:
    # ... (rest of the classifier function remains the same)
    prompt = f"""
You are a **Classifier Agent** responsible for routing user queries to the correct downstream agent based on the nature of the query.

There are **two agents** you can classify into:

---

### 🧴 **Agent1: Skin Care Agent**
**Responsibilities:**
- Handles queries related to **skin issues**, **daily skin routines**, **product suggestions**, and **remedies**.
- Capable of responding to concerns such as:
    - Acne, pimples, dark spots, pigmentation
    - Dry/oily skin
    - Product compatibility and side effects
    - Skin care routines based on **skin type**
- Can accept **optional skin images** to assist better in issue detection.
- Can recommend **products, remedies, or lifestyle tips**.
- May ask for **clarifying information** (e.g., skin type, allergies) if missing.

---

### 🥗 **Agent2: Nutrition Agent**
**Responsibilities:**
- Responds to queries regarding **food habits**, **diet plans**, **weight management**, **vitamins**, and **nutritional deficiencies**.
- Examples of handled queries:
    - “What should I eat to gain healthy weight?”
    - “Best diet for glowing skin?”
    - “I feel tired all the time—could it be due to lack of nutrients?”
    - “Is intermittent fasting safe?”
- Capable of giving **personalized nutrition advice** if **age, weight, or goals** are provided.
- Will request missing context if necessary.

---

### 🧠 **Chain-of-Thought (CoT) Reasoning:**
1. **Skin-related Queries**:
    - If the query mentions **visible skin issues**, **skin products**, or **routine questions**, classify as `"skin_care"`.
        - If an image is **attached**, set `missing_image: false`.
        - If not, set `missing_image: true`.
2. **Nutrition-related Queries**:
    - If the query is **about food, diet, supplements, or weight**, classify as `"nutrition"`.
    - If an image is **attached**, set `missing_image: false`.
        - If not, set `missing_image: true`.
3. **Requires Clarification**:
    - If the query is **ambiguous** or lacks **necessary personal details** for precise guidance (e.g., age, condition), set `requires_human_input: true`.

---

### 📦 **Few-shot Examples:**

**Example 1:**
User Query: "What can I apply to reduce dark spots on my face?"
→ `predicted_category`: `"skin_care"`, `missing_image`: true, `requires_human_input`: true, `clarification_question`: true

**Example 2:**
User Query: "Suggest a meal plan to gain weight for a 20-year-old."
→ `predicted_category`: `"nutrition"`, `requires_human_input`: false, `clarification_question`: false

**Example 3:**
User Query: "Is whey protein good for acne?"
→ `predicted_category`: `"nutrition"`, `requires_human_input`: false, `clarification_question`: false

**Example 4:**
User Query: "Check this rash and tell me if I need a dermatologist." (with image)
→ `predicted_category`: `"skin_care"`, `missing_image`: false, `requires_human_input`: false, `clarification_question`: false

**Example 5:**
User Query: "What's the best food for glowing skin?"
→ `predicted_category`: `"nutrition"`, `requires_human_input`: false, `clarification_question`: false

**Example 6:**
User Query: "My skin gets dry after washing, what should I do?"
→ `predicted_category`: `"skin_care"`, `missing_image`: true, `requires_human_input`: false, `clarification_question`: false

**Example 7:**
User Query: "What should I eat?"
→ `predicted_category`: `"nutrition"`, `requires_human_input`: true, `clarification_question`: true

---

### ✅ **Output Format**:
Respond in **raw JSON only**, no extra explanation.

Example:
{{
    "title": "Analyzing",
    "content": "Detecting the nature of the query...",
    "next_action": "continue",
    "predicted_category": "skin_care",
    "requires_human_input": false,
    "missing_image": true, // if no image is included
    "User_Query": "{state['message']}",
    "clarification_question": false
}}

User Query:
"{state['message']}"
"""

    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt, stream=False)

    raw = response.text.strip()
    if raw.startswith("```"):
        raw = raw.strip("`").strip("json").strip()

    print("🔍 Cleaned Classifier Response:\n", raw)

    try:
        parsed_result = json.loads(raw)
    except json.JSONDecodeError:
        raise ValueError(f"❌ Gemini Classifier still didn't return valid JSON. Got:\n{raw}")

    state["category"] = parsed_result["predicted_category"]
    state["intermediate_responses"].append(parsed_result)
    return state

# 3. Node 2: Critic Agent
def critic(state: ChatState) -> ChatState:
    # ... (rest of the critic function remains the same)
    prompt = f"""
You are Agent2 (Critic).

Your role is to critically review and validate the classification made by Agent1.
You must check if Agent1 correctly categorized the user's query based on the rules below.

---

### Guidelines for Decision-Making:

1. If the query involves a **visible skin issue** (e.g., acne, rashes, pigmentation, dryness), it should be classified as `"skin_care"`.
    - If **no image** is provided, set `"missing_image": true`.
    - If an image **is present**, set `"missing_image": false`.

2. If the query is a **text-based question related to food, diet, weight, or nutrition**, classify as `"nutrition"`.
    - IMPORTANT: For nutrition queries, if the query lacks **key context** like age, health conditions, or goals (e.g., weight loss), you MUST set `"requires_human_input": true` and `"clarification_question": true`.
    - Only mark nutrition queries as not requiring clarification if they:
        a) Provide enough personal context, or
        b) Are general questions that don’t need personalization

3. If the query **does not provide enough details** to decide the category or offer advice, set `"requires_human_input": true` and `"clarification_question": true`.

You must **correct** any misclassification by Agent1 if needed.

---

Agent1 prediction: "{state['category']}"
User Query: "{state['message']}"

Respond only in **raw JSON**, no extra explanation.

### Output Format:

{{
    "title": "Final Review",
    "content": "Verified category.",
    "next_action": "final_answer",
    "predicted_category": "skin_care",  # Choose one: "skin_care" or "nutrition"
    "requires_human_input": false,  # Set true if more info is needed
    "missing_image": false,          # Only for skin_care queries
    "clarification_question": false,  # Ask clarification if needed
    "User_Query": "{{state['message']}}"
}}
"""

    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt, stream=False)

    raw = response.text.strip()
    if raw.startswith("```"):
        raw = raw.strip("`").strip("json").strip()

    print("🔍 Cleaned Critic Response:\n", raw)

    try:
        parsed_result = json.loads(raw)
        final_category = parsed_result["predicted_category"]
        state["category"] = final_category
        state["intermediate_responses"].append(parsed_result)
    except json.JSONDecodeError:
        print(f"⚠️ Critic returned invalid JSON: {raw}. Using classifier's prediction.")
        state["category"] = state.get('intermediate_responses', [{}])[-1].get('predicted_category') # Use last classifier prediction
        state["intermediate_responses"].append({"error": "Invalid Critic JSON", "raw_output": raw})

    return state

# 4. Node 3: Skin Care Agent
def ask_skin_care_agent(state: ChatState) -> ChatState:
    # ... (rest of the ask_skin_care_agent function remains the same)
    image_bytes = state.get("image")
    user_text = state.get("message", "")

    if not image_bytes and state["category"] == "skin_care":
        state["response"] = "Please upload an image of the issue so I can help you diagnose it better."
        return state

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB") if image_bytes else None
    model = genai.GenerativeModel("gemini-1.5-flash")

    prompt = f'''
You are Agent1 (Skin Issue Detection & Troubleshooting Agent).

Your role is to analyze user-submitted **images of their skin**, optionally accompanied by **textual descriptions**, to detect **visible skin concerns** and offer basic guidance.

---

💡 **Key Rules:**
1. An **image is required** to analyze the query. If no image is provided and the category is "skin_care", kindly ask the user to upload one.
2. **Textual context** (e.g., “This rash appeared yesterday” or “It’s itchy around my cheeks”) is helpful but optional.
3. You are encouraged to ask **clarifying questions** if the image is unclear or more information is needed.
4. Focus on **common visible skin concerns**, such as:
    - Acne or pimples
    - Rashes or irritation
    - Redness or inflammation
    - Dryness or flaky skin
    - Pigmentation or dark spots
    - Scars or blemishes
    - Unusual bumps or lesions

---

🧠 **Responsibilities:**
- Detect any visible skin issues in the uploaded image.
- Provide a **brief explanation** of the visible issue.
- Offer **basic skincare advice** or suggest seeing a dermatologist if necessary.
- If unsure, ask for **a clearer image or more context**.

---

📌 **Example Interactions (Few-Shot):**

**Example 1:**
User: “What’s happening on my forehead?” *(Image uploaded)*
Agent1: “You appear to have mild acne with some inflammation. Try using a gentle cleanser twice daily and avoid touching the affected area. If it persists, consider consulting a dermatologist.”

**Example 2:**
User: *(Image of dry, flaky skin, no text)*
Agent1: “It looks like your skin is experiencing dryness. I recommend using a fragrance-free moisturizer and avoiding harsh soaps.”

**Example 3:**
User: “This red patch feels itchy.” *(No image)*
Agent1: “Could you please upload a photo of the red patch? That would help me identify the issue more accurately.”

---

✅ **Action Rules**:
- If image is missing and category is "skin_care" → Ask the user to upload an image.
- If image is unclear → Ask clarifying questions.
- If issue is visible → Describe the issue and suggest basic guidance.

Now respond to:
    User: {state['message']} Optional Image: { 'present' if state.get('image') else 'missing' }
'''

    if image:
        response = model.generate_content([image, user_text] if user_text else [image], stream=False)
        state["response"] = response.text.strip()
        state["agent_1_output"] = response.text.strip()
    elif state["category"] == "skin_care" and not state.get("response"):
        state["response"] = "Please upload an image so I can assist you."
    elif state["category"] == "skin_care":
        state["agent_1_output"] = state.get("response") # Keep the "upload image" message
    return state

# 5. Node 4: Nutrition Agent
def ask_nutrition_agent(state: ChatState) -> ChatState:
    # ... (rest of the ask_nutrition_agent function remains the same)
    prompt = f'''
    You are Agent 2 (Nutrition Support Agent), responsible for providing **dietary and nutritional guidance** based on the user’s skin condition or concerns.

Responsibilities:
● Analyze the skin concern described by the user (e.g., acne, dryness, redness, pigmentation).
● Provide evidence-based nutritional recommendations to support skin health and healing.
● Suggest helpful foods, hydration tips, vitamins, and daily calorie needs if relevant.
● Encourage lifestyle habits that promote better skin from within.

Expected Behavior:
1. Identify the skin-related issue from the user’s message or context (passed from the image diagnosis or directly asked).
2. Recommend a supportive **diet plan**, including key foods and nutrients that address the issue.
3. If needed, ask clarifying follow-up questions (e.g., if severity, age, or medical history is relevant).
4. Avoid providing medical diagnoses—focus on general nutrition advice unless specified otherwise.

---

**Few-shot Examples:**

**User:** "What should I eat to help reduce acne?"
**Agent 2:** "To help reduce acne, consider lowering sugar and dairy intake. Include foods rich in omega-3s like flaxseeds and walnuts, zinc-rich foods like pumpkin seeds, and drink plenty of water. Would you like a simple meal plan?"

**User:** "I have dry skin—any food tips?"
**Agent 2:** "For dry skin, focus on healthy fats like avocado, nuts, and olive oil. Also include hydrating foods like cucumbers, watermelon, and drink 2-3 liters of water daily. Vitamin E and omega-3s are especially helpful."

**User:** "My skin has red patches and feels inflamed."
**Agent 2:** "That could be linked to inflammation. Avoid processed foods, and eat anti-inflammatory foods like berries, leafy greens, turmeric, and fatty fish. Would you like a sample anti-inflammatory diet?"

---

    User input: {state["message"]}
If the context is unclear or more details are needed to give accurate nutrition advice, ask a gentle clarifying question.
Respond with practical, food-based support for better skin health.
    '''

    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt, stream=False)
    state["response"] = response.text.strip()
    state["agent_2_output"] = response.text.strip()
    return state

# 6. Node 5: Human Intervention Handler
def handle_human_input(state: ChatState, ask_func=input) -> ChatState:
    # ... (rest of the handle_human_input function remains the same)
    if state.get("requires_human_input", False):
        clarification = state.get("clarification_question")
        if not clarification:
            clarification = "Can you provide more details to proceed further?"

        print(f"\n🔍 System: Clarification required (Attempt {state.get('clarification_attempts', 0) + 1}).")
        print(f"🧠 Asking user: {clarification}")

        user_input = ask_func(f"{clarification}\n→ ")
        if user_input.lower()== "exit":
            print("Exiting clarification loop.")
            state["response"] = "User chose to exit the conversation."
            state["requires_human_input"] = False
            state["exit_requested"] = True  # Flag for explicit exit
            state["clarification_attempts"] = state.get("clarification_attempts", 0) + 1
            print(f"DEBUG (handle_human_input - exit): exit_requested={state.get('exit_requested')}, requires_human_input={state.get('requires_human_input')}")
            return state

        state["message"] = f"{state['message']} {user_input}"
        state["requires_human_input"] = False  # Reset the flag
        state["clarification_question"] = None
        state["clarification_attempts"] = state.get("clarification_attempts", 0) + 1
        print(f"DEBUG (handle_human_input - input): exit_requested={state.get('exit_requested')}, requires_human_input={state.get('requires_human_input')}")
        return state
    return state

# 7. Node 6: Aggregator Agent
def aggregator_agent(state: ChatState) -> ChatState:
    # ... (rest of the aggregator_agent function remains the same)
    prompt = f"""
You are the Aggregator Agent.

Your job is to combine insights from:
- Agent1: Skin care issue diagnosis (based on image and user input)
- Agent2: Nutrition advice (based on skin issue and health context)

---

🎯 Objective:
Provide a final, clear, and well-structured response that:
1. Briefly explains the diagnosed skin condition (from Agent1).
2. Gives 1–2 helpful skincare suggestions (from Agent1).
3. Offers targeted nutrition advice to support healing or improvement (from Agent2).
4. If applicable, recommend visiting a dermatologist or dietitian if the issue seems serious or unclear.

Keep the language friendly, actionable, and medically aware (but not overly technical).

---

🧾 Format your final output like this:

---
🔍 **Diagnosis Summary**:
[Brief summary from Agent1]

💡 **Skin Care Tips**:
- [Tip 1]
- [Tip 2]

🥗 **Nutrition Advice**:
[Advice from Agent2]

🧑‍⚕️ *If symptoms persist or worsen, please consult a certified dermatologist or nutritionist.*
---

---

# Agent Outputs:
## Skin Care Diagnosis (Agent1):
{state.get('agent_1_output', 'No skin care diagnosis available.')}

## Nutrition Support (Agent2):
{state.get('agent_2_output', 'No nutrition advice available.')}

Now generate the final combined output.
"""

    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt, stream=False)
    state["aggregated_response"] = response.text.strip()
    return state

# 8. Node 7: Answer Verification Agent
def answer_verification(state: ChatState) -> ChatState:
    # ... (rest of the answer_verification function remains the same)
    print(f"DEBUG (answer_verification - entry): exit_requested={state.get('exit_requested')}, requires_human_input={state.get('requires_human_input')}")
    if state.get("exit_requested", False):
        print("DEBUG (answer_verification): exit_requested is True, returning state to end flow.")
        return state

    response_text = state.get("aggregated_response") or state.get("response", "") # Check aggregated response first
    prompt = f'''
You are a helpful assistant reviewing an AI agent's response.
Your task is to determine if the user's original query has been adequately addressed.

Answer 'yes' if the response seems complete and doesn't require further user input, otherwise answer 'no'.

---
Response:
"{response_text}"
---
'''
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt, stream=False)
    model_output = response.text.strip().lower()

    if "no" in model_output and "yes" not in model_output:
        state["requires_human_input"] = True
        state["clarification_question"] = "The previous response might need more detail. How can I further assist you?"
    else:
        state["requires_human_input"] = False
        state["clarification_question"] = None

    print(f"DEBUG (answer_verification - exit): requires_human_input={state.get('requires_human_input')}, clarification_question={state.get('clarification_question')}")
    return state

# 9. Workflow Setup
workflow = StateGraph(ChatState)

# 10. Add Nodes to the Workflow
workflow.add_node("classifier", classifier)
workflow.add_node("critic", critic)
workflow.add_node("agent_1", ask_skin_care_agent)
workflow.add_node("agent_2", ask_nutrition_agent)
workflow.add_node("answer_verification", answer_verification)
workflow.add_node("human_intervention", handle_human_input)
workflow.add_node("aggregator", aggregator_agent)

# 11. Define Sequential Edges for Desired Flow
workflow.add_edge("classifier", "critic")
workflow.add_edge("agent_1", "agent_2")
workflow.add_edge("agent_2", "aggregator")
workflow.add_edge("aggregator", "answer_verification")

# 12. Define Conditional Edges for Critic Output
def get_routing_decision(state: ChatState) -> str:
    """Routes based on the critic's assessment and predicted category."""
    if state.get("requires_human_input", False):
        return "human_intervention"
    return f"agent_{'1' if state['category'] == 'skin_care' else '2'}"

workflow.add_conditional_edges("critic", get_routing_decision, {
    "agent_1": "agent_1",
    "agent_2": "agent_2",
    "human_intervention": "human_intervention"
})

# 13. Define Conditional Edges for Human Intervention
def route_from_human_intervention(state):
    """Routes based on whether the user requested to exit or needs further clarification."""
    if state.get("exit_requested", False):
        return "__end__"
    return "classifier"  # Go back to classifier for new input

workflow.add_conditional_edges(
    "human_intervention",
    route_from_human_intervention,
    {
        "classifier": "classifier",
        "__end__": END  # Allow direct exit from human intervention
    }
)

# 14. Define Conditional Edges for Answer Verification
def verify_answer_and_route(state):
    """Routes based on whether the answer is sufficient or requires human intervention, or if the user exited."""
    if state.get("exit_requested", False):
        return "__end__"
    elif state.get("requires_human_input", False):
        return "human_intervention"
    else:
        return "__end__"

workflow.add_conditional_edges("answer_verification", verify_answer_and_route, {
    "human_intervention": "human_intervention",
    "__end__": END
})

# 15. Set the Entry Point of the Workflow
workflow.set_entry_point("classifier")

# 16. Compile the Workflow
chain = workflow.compile()

# 17. FastAPI Integration
middleware = [
    Middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173"],  # Allow requests from your React app's origin
        allow_credentials=True,
        allow_methods=["*"],  # Allow all HTTP methods (POST, GET, PUT, DELETE, etc.)
        allow_headers=["*"],  # Allow all headers
    )
]

app = FastAPI(middleware=middleware)

@app.post("/chat")
async def chat_endpoint(message: str = Form(...), image_file: Optional[UploadFile] = File(None)):
    image_bytes = None
    if image_file:
        image_bytes = await image_file.read()

    result = await chain.ainvoke(ChatState(
        image=image_bytes,
        message=message,
        location=None,
        stage=None,
        category=None,
        response=None,
        conversation_history=[],
        intermediate_responses=[],
        requires_human_input=None,
        clarification_question=None,
        clarification_attempts=0,
        exit_requested=False,
        agent_1_output=None,
        agent_2_output=None,
        aggregated_response=None
    ))
    return JSONResponse(content={"response": result.get('aggregated_response', result.get('response', 'No response.'))})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)