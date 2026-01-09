import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


# --------------------------------
# Page config
# --------------------------------
st.set_page_config(
    page_title="IxA",
    page_icon="🔬",
    layout="wide"
)

st.title("🌍 Impact Story Interview Assistant")
st.caption(
    "A guided human–AI process for translating research into societal impact stories."
)


# --------------------------------
# Sidebar: API config
# --------------------------------
st.sidebar.header("🔑 LLM Configuration")

api_key = st.sidebar.text_input(
    "OpenAI API Key",
    type="password"
)

base_url = st.sidebar.text_input(
    "Base URL",
    value="https://ai-research-proxy.azurewebsites.net"
)

model_name = st.sidebar.text_input(
    "Model",
    value="gpt-4.1-mini"
)

if not api_key:
    st.warning("Please enter your API key to continue.")
    st.stop()

# --------------------------------
# Initialize LLM
# --------------------------------
llm = ChatOpenAI(
    model=model_name,
    api_key=api_key,
    base_url=base_url
)

# --------------------------------
# Session state initialization
# --------------------------------
# if "paper_context" not in st.session_state:
#     st.session_state.paper_context = None

# if "contribution_state" not in st.session_state:
#     st.session_state.contribution_state = {
#         "problem_gap": None,
#         "method_novelty": None,
#         "theoretical_significance": None,
#         "empirical_strength": None,
#         "real_world_impact": None
#     }

# if "interview_transcript" not in st.session_state:
#     st.session_state.interview_transcript = []

# if "current_question" not in st.session_state:
#     st.session_state.current_question = None

if "paper_text" not in st.session_state:
    st.session_state.paper_text = None

if "impact_state" not in st.session_state:
    st.session_state.impact_state = {
        "title_hook": {"content": None, "answers": [], "attempts": 0},
        "societal_problem": {"content": None, "answers": [], "attempts": 0},
        "societal_impact": {"content": None, "answers": [], "attempts": 0},
        "research_and_approach": {"content": None, "answers": [], "attempts": 0},
        "people_and_collaboration": {"content": None, "answers": [], "attempts": 0},
        "outlook": {"content": None, "answers": [], "attempts": 0}
    }

if "interview_transcript" not in st.session_state:
    st.session_state.interview_transcript = []

if "current_prompt" not in st.session_state:
    st.session_state.current_prompt = None

# --------------------------------
# Helper functions
# --------------------------------
# def next_missing_dimension(state):
#     for k, v in state.items():
#         if v is None:
#             return k
#     return None


# def ask_question(missing_dimension, paper_context):
#     prompt = (
#         "You are an academic interviewer chatbot conducting a structured "
#         "impact elicitation interview.\n\n"
#         "Use the following research paper as background context:\n\n"
#         f"{paper_context}\n\n"
#         f"The interview should help clarify the scientific dimension: "
#         f"'{missing_dimension}'.\n\n"
#         "Ask one short, open-ended question."
#     )
#     return llm.invoke(prompt).content


# def update_state(dimension, researcher_answer, paper_context):
#     prompt = (
#         "You are evaluating whether a researcher's response sufficiently "
#         "clarifies a contribution.\n\n"
#         f"Scientific dimension: {dimension}\n\n"
#         "Research paper context:\n"
#         f"{paper_context}\n\n"
#         "Researcher's answer:\n"
#         f"{researcher_answer}\n\n"
#         "If the answer sufficiently clarifies the dimension, summarize it "
#         "in 1–2 sentences. Otherwise, return ONLY the word 'INSUFFICIENT'."
#     )
#     result = llm.invoke(prompt).content.strip()
#     if result.upper() != "INSUFFICIENT":
#         st.session_state.contribution_state[dimension] = result

def summarize_paper_sections(full_text):
    prompt = (
        "You are an academic assistant.\n\n"
        "Summarize the following research paper into the following sections:\n"
        "1. Research problem and motivation\n"
        "2. Methodology and data\n"
        "3. Key findings\n"
        "4. Theoretical or scientific contribution\n"
        "5. Practical or real-world implications\n\n"
        "Paper text:\n"
        f"{full_text}"
    )
    return llm.invoke(prompt).content

MAX_ATTEMPTS_PER_SECTION = 2

def next_missing_section(state):
    for k, v in state.items():
        if v["content"] is None and v["attempts"] < MAX_ATTEMPTS_PER_SECTION:
            return k
    return None


def generate_interview_turn(section, paper_text, last_answer=None):
    prompt = (
        "You are facilitating an impact story interview for a broad, non-academic audience.\n\n"
        "First, briefly acknowledge and paraphrase the researcher's previous answer "
        "in plain language (if one exists).\n"
        "Then explain why this aspect matters from a societal or human perspective.\n"
        "Finally, ask ONE open-ended question to help develop the following impact story section:\n\n"
        f"{section.replace('_', ' ').title()}\n\n"
        "Avoid jargon. Keep the tone conversational and supportive.\n\n"
        "Research paper context:\n"
        f"{paper_text}\n\n"
        f"Previous answer:\n{last_answer if last_answer else 'N/A'}"
    )
    return llm.invoke(prompt).content

def evaluate_and_store(section, paper_text):
    section_data = st.session_state.impact_state[section]
    combined_answers = "\n".join(section_data["answers"])

    prompt = (
        f"You are helping synthesize an impact story section: "
        f"{section.replace('_', ' ').title()}.\n\n"
        "Below are multiple responses provided by the researcher over the interview.\n\n"
        f"{combined_answers}\n\n"
        "If this information is sufficient, synthesize a clear, human-friendly "
        "paragraph (3–4 sentences) suitable for a broad audience.\n"
        "If information is still incomplete, return ONLY the word 'INSUFFICIENT'."
    )

    result = llm.invoke(prompt).content.strip()
    
    if (
        section_data["content"] is None
        and section_data["attempts"] >= MAX_ATTEMPTS_PER_SECTION
    ):
        fallback_prompt = (
            f"Based on the partial information below, write the best possible "
            f"version of the impact story section '{missing_section}'. "
            "Be transparent but constructive.\n\n"
            f"{combined_answers}"
        )
    
        section_data["content"] = llm.invoke(fallback_prompt).content

    if result.upper() != "INSUFFICIENT":
        section_data["content"] = result
    

# def evaluate_and_store(section, answer, paper_text):
#     prompt = (
#         "Determine whether the following answer sufficiently contributes to the "
#         f"impact story section '{section}'.\n\n"
#         "Research paper context:\n"
#         f"{paper_text}\n\n"
#         f"Answer:\n{answer}"
#         "If sufficient, rewrite it in clear, human-friendly language (2–3 sentences).\n"
#         "If not sufficient, return ONLY the word 'INSUFFICIENT'.\n\n"
#     )
#     result = llm.invoke(prompt).content.strip()
#     if result.upper() != "INSUFFICIENT":
#         st.session_state.impact_state[section] = result
        
# --------------------------------
# Step 1: Upload PDF
# --------------------------------
st.header("📄 Upload Research Paper")

uploaded_file = st.file_uploader(
    "Upload a research paper PDF",
    type=["pdf"]
)

if uploaded_file and st.session_state.paper_text is None:
    with st.spinner("Reading and processing PDF..."):
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())

        loader = PyPDFLoader("temp.pdf")
        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=200
        )
        docs = splitter.split_documents(documents)

        paper_text = "\n\n".join(d.page_content for d in docs)
        st.session_state.paper_text = summarize_paper_sections(paper_text)

    st.success("Paper processed successfully.")

# --------------------------------
# Step 2: Interview Section
# --------------------------------
if st.session_state.paper_text:
    st.header("🗣️ Guided Impact Interview")

    missing_section = next_missing_section(st.session_state.impact_state)

    if missing_section:
        if st.session_state.current_prompt is None:
            last_answer = (
                st.session_state.interview_transcript[-1][1]
                if st.session_state.interview_transcript else None
            )

            with st.spinner("Preparing next interview turn..."):
                st.session_state.current_prompt = generate_interview_turn(
                    missing_section,
                    st.session_state.paper_text,
                    last_answer
                )

        st.markdown(f"**🤖 Interviewer:** {st.session_state.current_prompt}")

        user_answer = st.text_area(
            "👤 Your response",
            key=f"answer_{missing_section}"
        )

        if st.button("Submit response"):
            st.session_state.interview_transcript.append(
                (st.session_state.current_prompt, user_answer)
            )

            with st.spinner("Integrating response..."):
                section_data = st.session_state.impact_state[missing_section]
                section_data["answers"].append(user_answer)
                section_data["attempts"] += 1
                
                evaluate_and_store(missing_section, st.session_state.paper_text)
                # evaluate_and_store(missing_section, user_answer, st.session_state.paper_text)

            st.session_state.current_prompt = None
            st.rerun()
    else:
        st.success("All impact story elements have been collected.")

# --------------------------------
# Step 3: Show Contribution State
# --------------------------------
if st.session_state.paper_text:
    st.header("📊 Impact Story Coverage")

    for k, v in st.session_state.impact_state.items():
        with st.expander(k.replace("_", " ").title()):
            st.write(v["content"] if v["content"] else "Not yet completed")

# --------------------------------
# Step 4: Generate Impact Story
# --------------------------------
if all(st.session_state.impact_state.values()):
    st.header("📘 Final Impact Story")

    if st.button("Generate Impact Story"):
        with st.spinner("Writing impact story..."):
            prompt = (
                "Write a complete Impact Story for a broad audience using the structure below.\n\n"
                "Structure and word limits:\n"
                "- Title (short, active)\n"
                "- Introduction (~100 words)\n"
                "- Societal Impact (~150 words)\n"
                "- Research and Approach (~150 words)\n"
                "- People and Collaboration (~100 words)\n"
                "- Conclusion / Outlook (~50 words)\n\n"
                "Keep the story concrete and human. Avoid jargon.\n"
                "Focus on the connection between science and everyday life.\n\n"
                "Use the following elicited content as your primary source:\n\n"
            )

            for k, v in st.session_state.impact_state.items():
                prompt += f"\n{k.replace('_', ' ').title()}:\n{v}\n"

            impact_story = llm.invoke(prompt).content

        st.markdown(impact_story)