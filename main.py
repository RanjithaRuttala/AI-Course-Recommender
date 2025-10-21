import os
import json
import streamlit as st
import pandas as pd
import openai
import psycopg2
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_postgres import PGVector
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from pathlib import Path
import base64

# ---------------------------
# Load Config
# ---------------------------
with open("config.json", "r") as f:
    CONFIG = json.load(f)

os.environ["OPENAI_API_KEY"] = CONFIG["OPENAI_API_KEY"]
openai.api_key = CONFIG["OPENAI_API_KEY"]

DB = CONFIG["POSTGRES"]
embedding_model = CONFIG["EMBEDDING_MODEL"]
llm_model = CONFIG["LLM_MODEL"]
connection = f"postgresql+psycopg://{DB['USER']}:{DB['PASSWORD']}@{DB['HOST']}:{DB['PORT']}/{DB['DBNAME']}"
collection_name = "edtech_courses"

# ---------------------------
# Initialize Embeddings and Vector Store
# ---------------------------
embeddings = OpenAIEmbeddings(model=embedding_model)
try:
    vectorstore = PGVector.from_existing_index(
        embedding=embeddings,
        collection_name=collection_name,
        connection=connection,
        use_jsonb=True,
    )
except Exception as e:
    st.error(f"Error connecting to PGVector: {e}")
    st.stop()

# ---------------------------
# Helper Functions
# ---------------------------
def get_embedding(text: str):
    """Generate embeddings for a given text."""
    if not text or pd.isna(text):
        return None
    response = openai.embeddings.create(model=embedding_model, input=text)
    return response.data[0].embedding



def get_top_courses_from_postgres(query_embedding, top_k=3):
    """Retrieve top K most similar courses from PostgreSQL."""
    with psycopg2.connect(
        host=DB["HOST"],
        port=DB["PORT"],
        dbname=DB["DBNAME"],
        user=DB["USER"],
        password=DB["PASSWORD"]
    ) as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT course_id, courses_title, courses_description, category, sub_category
                FROM courses
                ORDER BY embedding <-> %s::vector
                LIMIT %s;
            """, (query_embedding, top_k))
            return cur.fetchall()


def generate_final_response(user_query, top_courses):
    """Generate a clean, structured recommendation from retrieved courses."""
    context = "\n\n".join([
        f"Course ID: {c[0]}\nTitle: {c[1]}\nDescription: {c[2]}\nCategory: {c[3]}\nSubcategory: {c[4]}"
        for c in top_courses
    ])

    prompt = f"""
    You are an AI course recommendation assistant.

    The user asked: "{user_query}"

    Below are the top 3 most relevant courses retrieved from our database:

    {context}

    Generate recommendations in this format:
    1. Course ID: <id> | Title: <title> | Category: <category> | Subcategory: <subcategory> | Reason: <why this course suits the user's query>
    2. Course ID: <id> | Title: <title> | Category: <category> | Subcategory: <subcategory> | Reason: <why this course suits the user's query>
    3. Course ID: <id> | Title: <title> | Category: <category> | Subcategory: <subcategory> | Reason: <why this course suits the user's query>

    Keep each reason short (2–3 lines).
    """
    response = openai.chat.completions.create(
        model=llm_model,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()

# ---------------------------
# Feedback Analysis Chains (Use Case 2)
# ---------------------------
llm = ChatOpenAI(model=llm_model, temperature=0.3)

feedback_prompt = PromptTemplate.from_template("""
You are an intelligent learning analyst.

Below is a learner's feedback or quiz performance summary:

"{feedback_text}"

Your task:
1. Parse all sections, their max marks, and marks obtained.
2. Identify **all strong areas** (>=75%) and **all weak areas** (<75%).
3. List every weak topic explicitly.
4. Suggest a concise **learning goal** describing what the learner should focus on next.

Return your answer in this exact format:
Strengths: ...
Weaknesses: ...
Learning Goal: ...
""")

feedback_chain = LLMChain(llm=llm, prompt=feedback_prompt)

recommend_prompt = PromptTemplate.from_template("""
You are an AI course recommendation assistant.

The learner's goal is: "{learning_goal}"

Below are the top {k} most relevant courses retrieved from our database:
{context}

Generate recommendations in this exact format:
1. Course ID: <id> | Title: <title> | Category: <category> | Subcategory: <subcategory> | Reason: <why this course suits the learner>
2. Course ID: <id> | Title: <title> | Category: <category> | Subcategory: <subcategory> | Reason: <why this course suits the learner>
3. Course ID: <id> | Title: <title> | Category: <category> | Subcategory: <subcategory> | Reason: <why this course suits the learner>
""")

recommend_chain = LLMChain(llm=llm, prompt=recommend_prompt)

# ---------------------------
# Streamlit UI
# ---------------------------
assets_path = Path("assets")

def load_svg(svg_filename):
    """Load and return contents of an SVG file."""
    svg_path = assets_path / svg_filename
    if not svg_path.exists():
        st.error(f"SVG file not found: {svg_path}")
        return ""
    with open(svg_path, "r", encoding="utf-8") as f:
        return f.read()
    
def setup_page():
    """Setup Streamlit page config and render logo header."""
    st.set_page_config(
        page_title="AI Course Recommender",
        layout="wide",
        page_icon=f"{assets_path}/V2-favicon.png"
    )
    st.markdown(
        """
        <style>
        /* Hide default Streamlit top bar */
        header {display: none !important;}
        </style>
        """,
        unsafe_allow_html=True
    )
    logo_svg = load_svg("V2-logo.svg")
    bg_svg = load_svg("signing-background.svg")
    logo_encoded = base64.b64encode(logo_svg.encode()).decode()
    bg_encoded = base64.b64encode(bg_svg.encode()).decode()

    # Create the header section
    logo_html = f'''
            <div style="width: 100%; height: 100px; background-image: url('data:image/svg+xml;base64,{bg_encoded}');
                        background-size: cover; background-position: center; display: flex;margin-top:-1rem ">
                <img src="data:image/svg+xml;base64,{logo_encoded}" alt="Logo" style="width: 200px;margin-left:0;">
            </div>
        '''
    st.markdown(logo_html, unsafe_allow_html=True)

    # Adjust overall padding
    st.markdown(
        """
        <style>
                 .block-container {
                    
                    margin-top: 0rem !important;
                    padding: 0 !important;
                    margin-bottom: 5rem !important;
                    max-width: 100% !important;
                }
                [data-testid="stVerticalBlock"] [data-testid="stVerticalBlockBorderWrapper"] {
                    padding-left: 10rem !important;
                    padding-right: 10rem !important;
                    max-width: 100% !important;
                }
                </style>
        """,
        unsafe_allow_html=True
    )

# st.set_page_config(page_title="AI Course Recommender", layout="wide")
setup_page()

st.title("AI-Powered Course Recommendation System")
st.markdown("Find personalized course suggestions based on your **goals** or **feedback performance**.")

# Buttons to select mode
col1, col2 = st.columns(2)
with col1:
    goal_button = st.button("Recommend from Learning Goal")
with col2:
    feedback_button = st.button("Recommend from Feedback or Quiz Performance")

st.markdown("---")

# ---------------------------
# Mode 1: Based on User Input (Goal)
# ---------------------------
if "mode" not in st.session_state:
    st.session_state.mode = None

if goal_button:
    st.session_state.mode = "goal"
elif feedback_button:
    st.session_state.mode = "feedback"

if st.session_state.mode == "goal":
    st.subheader("Find Courses by Learning Goal or Interest")
    st.markdown("Enter what you want to learn or improve — for example:")
    st.markdown("- *I want to become better at public speaking.*  \n- *I want to improve my emotional intelligence at work.*  \n- *I want to learn conflict resolution skills.*")

    user_query = st.text_input("Enter your learning goal:",
                               placeholder=" Enter your query here.")

    if st.button("Search Courses"):
        if not user_query.strip():
            st.warning("Please enter your learning goal first.")
        else:
            with st.spinner("Generating embeddings and finding top matches..."):
                query_embedding = get_embedding(user_query)
                top_courses = get_top_courses_from_postgres(query_embedding, top_k=3)

            if not top_courses:
                st.error("No matching courses found.")
            else:
                with st.spinner("Generating final recommendations..."):
                    final_answer = generate_final_response(user_query, top_courses)

                st.subheader("Recommended Courses")
                st.write(final_answer)

# ---------------------------
# Mode 2: Based on Feedback or Quiz
# ---------------------------
elif st.session_state.mode == "feedback":
    st.subheader("Recommend Based on Feedback or Quiz Performance")
    st.markdown("Enter what you want to learn or improve — for example:")
    st.markdown("- *Based on the assesment you have good understanding of emotional cues and teamwork. Try to improve on assertiveness and handling tough feedback.*")
    st.markdown("Upload or paste the learner’s feedback summary")
    

    feedback_text = st.text_area("Enter feedback text here:",
                               placeholder=" Enter your feedback here.")
    uploaded_file = st.file_uploader("Or upload a .txt file", type=["txt"])

    if uploaded_file is not None:
        content = uploaded_file.read().decode("utf-8")
        if not feedback_text:
            feedback_text = content

    if st.button("Analyze & Recommend"):
        if not feedback_text.strip():
            st.warning("Please provide feedback text or upload a file.")
        else:
            with st.spinner("Analyzing feedback..."):
                feedback_summary = feedback_chain.run(feedback_text=feedback_text)
                st.subheader("Feedback Analysis")
                st.text(feedback_summary)

            # Extract learning goal
            goal_line = ""
            for line in feedback_summary.split("\n"):
                if line.lower().startswith("learning goal"):
                    goal_line = line.split(":", 1)[-1].strip()
                    break

            if not goal_line:
                st.warning("Couldn't identify a clear learning goal from feedback.")
                st.stop()

            st.subheader(" Extracted Learning Goal")
            st.text(goal_line)

            with st.spinner("Finding best matching courses..."):
                results = vectorstore.similarity_search(goal_line, k=3)
                if not results:
                    st.warning("No relevant courses found.")
                    st.stop()

                context = "\n\n".join([
                    f"Course ID: {d.metadata['course_id']}\nTitle: {d.metadata['courses_title']}\n"
                    f"Description: {d.page_content}\nCategory: {d.metadata['category']}\nSubcategory: {d.metadata['sub_category']}"
                    for d in results
                ])

                final_response = recommend_chain.run(learning_goal=goal_line, context=context, k=3)

                st.subheader(" Recommended Courses")
                st.text(final_response)
