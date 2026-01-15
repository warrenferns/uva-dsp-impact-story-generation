import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from datetime import datetime, timedelta
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from docx import Document
from docx.shared import Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH
import io
import re


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
# Initialize API key in session state if not present
if "api_key" not in st.session_state:
    st.session_state.api_key = ""

# Create expander that collapses when API key is entered
with st.sidebar.expander("🔑 LLM Configuration", expanded=not bool(st.session_state.api_key)):
    api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        value=st.session_state.api_key,
        key="api_key_input"
    )
    
    base_url = st.text_input(
        "Base URL",
        value="https://ai-research-proxy.azurewebsites.net"
    )
    
    model_name = st.text_input(
        "Model",
        value="gpt-4.1-mini"
    )
    
    # Update session state when API key changes
    st.session_state.api_key = api_key

if not st.session_state.api_key:
    st.warning("Please enter your API key to continue.")
    st.stop()

# Use session state API key for LLM initialization
api_key = st.session_state.api_key

# --------------------------------
# Sidebar: Session info
# --------------------------------
with st.sidebar.expander("📋 Interview Session", expanded=True):
    if st.session_state.get("paper_text"):
        st.markdown("✓ Paper uploaded")
        
        # Calculate elapsed time
        if st.session_state.get("session_start_time"):
            elapsed = datetime.now() - st.session_state.session_start_time
            minutes = int(elapsed.total_seconds() / 60)
            st.markdown(f"Started {minutes} mins ago")
        else:
            st.markdown("Started 0 mins ago")
    else:
        st.markdown("Paper not uploaded")

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

if "session_start_time" not in st.session_state:
    st.session_state.session_start_time = None

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

if "generated_story" not in st.session_state:
    st.session_state.generated_story = None

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


def extract_title(story_text):
    """Extract and sanitize the title from the story text for use in filenames."""
    paragraphs = story_text.split('\n\n')
    if paragraphs:
        title = paragraphs[0].strip()
        # Remove markdown formatting
        title = re.sub(r'\*\*(.+?)\*\*', r'\1', title)
        title = re.sub(r'\*([^*]+?)\*', r'\1', title)
        # Sanitize for filename (remove invalid characters)
        title = re.sub(r'[<>:"/\\|?*]', '', title)
        # Limit length and clean up
        title = title[:100].strip()
        return title if title else "impact_story"
    return "impact_story"


def generate_pdf(story_text):
    """Generate a PDF document from the impact story text."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter,
                            rightMargin=72, leftMargin=72,
                            topMargin=72, bottomMargin=18)
    
    styles = getSampleStyleSheet()
    story = []
    
    # Title style
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=18,
        textColor='#1f77b4',
        spaceAfter=30,
        alignment=TA_CENTER
    )
    
    # Normal style with justified text
    normal_style = ParagraphStyle(
        'CustomNormal',
        parent=styles['Normal'],
        fontSize=11,
        leading=14,
        alignment=TA_JUSTIFY,
        spaceAfter=12
    )
    
    # Split story into paragraphs
    paragraphs = story_text.split('\n\n')
    
    for i, para in enumerate(paragraphs):
        para = para.strip()
        if not para:
            continue
        
        # Check if it's a title (usually first paragraph or all caps/short)
        is_title = (len(para) < 100 and para.isupper()) or (i == 0)
        
        if is_title:
            # Remove markdown formatting from title
            title_clean = re.sub(r'\*\*(.+?)\*\*', r'\1', para)
            title_clean = re.sub(r'\*([^*]+?)\*', r'\1', title_clean)
            story.append(Paragraph(title_clean, title_style))
        else:
            # Replace markdown formatting for PDF (ReportLab uses HTML-like tags)
            # Handle bold: **text** -> <b>text</b>
            para = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', para)
            # Handle italic: *text* -> <i>text</i> (but not if it's part of **)
            para = re.sub(r'(?<!\*)\*([^*]+?)\*(?!\*)', r'<i>\1</i>', para)
            story.append(Paragraph(para, normal_style))
        
        story.append(Spacer(1, 0.2*inch))
    
    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()


def generate_word(story_text):
    """Generate a Word document from the impact story text."""
    doc = Document()
    
    # Set default font
    style = doc.styles['Normal']
    font = style.font
    font.name = 'Calibri'
    font.size = Pt(11)
    
    # Split story into paragraphs
    paragraphs = story_text.split('\n\n')
    
    for i, para in enumerate(paragraphs):
        para = para.strip()
        if not para:
            continue
        
        # Check if it's a title
        is_title = (len(para) < 100 and para.isupper()) or (i == 0)
        
        if is_title:
            # Remove markdown formatting from title
            title_clean = re.sub(r'\*\*(.+?)\*\*', r'\1', para)
            title_clean = re.sub(r'\*([^*]+?)\*', r'\1', title_clean)
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            run = p.add_run(title_clean)
            run.font.size = Pt(18)
            run.font.bold = True
        else:
            # Handle markdown formatting for Word
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            
            # Process text with markdown formatting
            text_parts = re.split(r'(\*\*.*?\*\*|\*.*?\*)', para)
            for part in text_parts:
                if not part:
                    continue
                if part.startswith('**') and part.endswith('**'):
                    # Bold text
                    run = p.add_run(part[2:-2])
                    run.bold = True
                elif part.startswith('*') and part.endswith('*') and not part.startswith('**'):
                    # Italic text
                    run = p.add_run(part[1:-1])
                    run.italic = True
                else:
                    # Regular text
                    p.add_run(part)
        
        doc.add_paragraph()  # Add spacing between paragraphs
    
    buffer = io.BytesIO()
    doc.save(buffer)
    buffer.seek(0)
    return buffer.getvalue()
    

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
        # Set session start time when paper is uploaded
        if st.session_state.session_start_time is None:
            st.session_state.session_start_time = datetime.now()
        
        st.success("Paper processed successfully.")
        st.rerun()

# --------------------------------
# Step 2: Interview Section
# --------------------------------
if st.session_state.paper_text:
    st.header("🗣️ Guided Impact Interview")

    missing_section = next_missing_section(st.session_state.impact_state)
    all_answered = all(v["content"] is not None for v in st.session_state.impact_state.values())

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

    # Display conversation history
    for question, answer in st.session_state.interview_transcript:
        with st.chat_message("assistant"):
            st.write(question)
        with st.chat_message("user"):
            st.write(answer)

    # Display current pending question
    if missing_section and st.session_state.current_prompt:
        with st.chat_message("assistant"):
            st.write(st.session_state.current_prompt)

    # Chat input for user response (disabled when all answered)
    user_answer = st.chat_input("Type your answer here...", disabled=all_answered)

    if user_answer and missing_section:
        # Append to transcript immediately so it appears in chat
        st.session_state.interview_transcript.append(
            (st.session_state.current_prompt, user_answer)
        )

        with st.spinner("Integrating response..."):
            section_data = st.session_state.impact_state[missing_section]
            section_data["answers"].append(user_answer)
            section_data["attempts"] += 1
            
            evaluate_and_store(missing_section, st.session_state.paper_text)

        st.session_state.current_prompt = None
        st.rerun()
    
    if all_answered:
        st.success("All impact story elements have been collected.")
# --------------------------------
# Step 3: Show Contribution State
# --------------------------------
# if st.session_state.paper_text:
#     st.header("📊 Impact Story Coverage")

#     for k, v in st.session_state.impact_state.items():
#         with st.expander(k.replace("_", " ").title()):
#             st.write(v["content"] if v["content"] else "Not yet completed")

# --------------------------------
# Step 4: Generate Impact Story
# --------------------------------
if st.session_state.paper_text:
    all_answered = all(v["content"] is not None for v in st.session_state.impact_state.values())
    
    if all_answered:
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
                    prompt += f"\n{k.replace('_', ' ').title()}:\n{v['content']}\n"

                impact_story = llm.invoke(prompt).content
                st.session_state.generated_story = impact_story

        # Display the story if it exists in session state
        if st.session_state.generated_story:
            st.markdown(st.session_state.generated_story)
        
        # Display export options if story is generated
        if st.session_state.generated_story:
            st.subheader("📥 Export Options")
            
            # Extract title for filename
            story_title = extract_title(st.session_state.generated_story)
            
            pdf_bytes = generate_pdf(st.session_state.generated_story)
            st.download_button(
                label="📄 Download as PDF",
                data=pdf_bytes,
                file_name=f"{story_title}.pdf",
                mime="application/pdf"
            )
            
            word_bytes = generate_word(st.session_state.generated_story)
            st.download_button(
                label="📝 Download as Word",
                data=word_bytes,
                file_name=f"{story_title}.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            )
