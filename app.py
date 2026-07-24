from reportlab.lib.pagesizes import letter

from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.lib.styles import getSampleStyleSheet

from reportlab.lib import colors

from datetime import datetime
import os
import shutil
import plotly.express as px
import pandas as pd
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from core.loader import load_document
from core.splitter import split_documents
from core.vectordb import create_vector_db, load_vector_db
from core.rag import build_rag
from core.config import CHROMA_PATH

load_dotenv()

# =====================================================
# MULTI AGENT PROMPTS
# =====================================================

AGENTS = {

    "Standard Copilot":
        "You are BizPilot AI. Answer professionally using the retrieved documents.",

    "CEO Agent":
        """
        You are the CEO.
        Focus on:
        • Strategy
        • Business Growth
        • Risk
        • Opportunities
        • Long-term planning
        """,

    "Finance Agent":
        """
        You are a CFO.
        Focus on:
        • Revenue
        • Profit
        • Cost
        • Budget
        • Financial KPIs
        """,

    "Marketing Agent":
        """
        You are a Marketing Director.
        Focus on:
        • Customers
        • Branding
        • Campaigns
        • Sales Funnel
        • Market Analysis
        """,

    "Sales Agent":
        """
        You are Head of Sales.
        Focus on:
        • Revenue
        • Pipeline
        • Conversion
        • Customer Acquisition
        • Sales Forecast
        """,

    "HR Agent":
        """
        You are HR Director.
        Focus on:
        • Employees
        • Hiring
        • Retention
        • Skills
        • Organization
        """,

    "Operations Agent":
        """
        You are COO.
        Focus on:
        • Processes
        • Supply Chain
        • Productivity
        • Operational Risk
        • Efficiency
        """,

    "SQL Agent":
        """
        You are Data Analyst.
        Focus on:
        • Tables
        • KPIs
        • Statistics
        • Trends
        • SQL Thinking
        """
}

# ----------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------

st.set_page_config(
    page_title="BizPilot AI",
    page_icon="🤖",
    layout="wide",
)

# ----------------------------------------------------
# CSS
# ----------------------------------------------------

css = Path("assets/style.css")

if css.exists():
    with open(css) as f:
        st.markdown(
            f"<style>{f.read()}</style>",
            unsafe_allow_html=True,
        )

# ----------------------------------------------------
# SESSION
# ----------------------------------------------------

if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

if "documents" not in st.session_state:
    st.session_state.documents = []

if "messages" not in st.session_state:
    st.session_state.messages = []

if "indexed" not in st.session_state:
    st.session_state.indexed = False

if "selected_agent" not in st.session_state:
    st.session_state.selected_agent = "Standard Copilot"

# ----------------------------------------------------
# LOAD EXISTING DATABASE (ONLY ON FIRST APP LOAD)
# ----------------------------------------------------

if (
    st.session_state.get("qa_chain") is None
    and os.path.exists(CHROMA_PATH)
    and os.listdir(CHROMA_PATH)
):

    try:
        vectordb = load_vector_db()
        st.session_state.qa_chain = build_rag(vectordb)
        st.session_state.indexed = True

    except Exception as e:
        st.warning(f"Unable to load existing database: {e}")

# ----------------------------------------------------
# SIDEBAR
# ----------------------------------------------------

with st.sidebar:

    st.markdown("""
<div class="hero">

<h1>🤖 BizPilot AI</h1>

<p>Enterprise Business Intelligence Copilot</p>

<span>Analyze PDFs • CSV • DOCX with AI</span>

</div>
""", unsafe_allow_html=True)

    uploaded_files = st.file_uploader(

        "Upload Files",

        type=["pdf", "docx", "csv"],

        accept_multiple_files=True

    )

    if st.button("Create Knowledge Base"):

     if uploaded_files and len(uploaded_files) > 0:

        # Clear previous knowledge base
        st.session_state.qa_chain = None
        st.session_state.messages = []
        st.session_state.documents = []
        st.session_state.indexed = False

        import gc
        gc.collect()

        if os.path.exists(CHROMA_PATH):
            shutil.rmtree(CHROMA_PATH, ignore_errors=True)

        if os.path.exists("uploads"):
            shutil.rmtree("uploads", ignore_errors=True)

        docs = []

        os.makedirs("uploads", exist_ok=True)

        progress = st.progress(0)

        total = len(uploaded_files)

        for i, file in enumerate(uploaded_files):

            save_path = os.path.join("uploads", file.name)

            with open(save_path, "wb") as f:
                f.write(file.getbuffer())

            docs.extend(load_document(save_path))

            progress.progress((i + 1) / total)

        st.info("Splitting Documents...")

        chunks = split_documents(docs)

        with st.spinner("Creating Vector Database..."):
            vectordb = create_vector_db(chunks)

        with st.spinner("Building AI Knowledge Base..."):
            st.session_state.qa_chain = build_rag(vectordb)

        st.session_state.documents = uploaded_files
        st.session_state.indexed = True

        st.success("Knowledge Base Ready!")

# =====================================================
# MAIN HEADER
# =====================================================

st.title("🤖 BizPilot AI")
st.caption("Enterprise Multi-Agent Business Intelligence Copilot")

# =====================================================
# STATUS METRICS
# =====================================================

c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-title">Knowledge Base</div>
        <div class="metric-value">{}</div>
    </div>
    """.format(
        "Ready ✅" if st.session_state.indexed else "Not Ready ❌"
    ), unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Documents</div>
        <div class="metric-value">{len(st.session_state.documents)}</div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Messages</div>
        <div class="metric-value">{len(st.session_state.messages)}</div>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# =====================================================
# ACTIVE AGENT
# =====================================================

st.divider()

st.markdown("""
<div class="glass-card">

<h3>🧠 AI Business Agents</h3>

<p>Select an AI specialist to analyze your business data.</p>

</div>
""", unsafe_allow_html=True)

st.session_state.selected_agent = st.selectbox(

    "Choose an AI Expert",

    list(AGENTS.keys())

)

# =====================================================
# CHAT HISTORY
# =====================================================

for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):

        st.markdown(msg["content"])

        if (
            msg["role"] == "assistant"
            and "sources" in msg
            and msg["sources"]
        ):

            st.markdown("### 📄 Sources")

            for src in msg["sources"]:

                st.write(
                    "•",
                    src.metadata.get(
                        "source",
                        "Unknown"
                    )
                )

# =====================================================
# CHAT INPUT
# =====================================================

question = st.chat_input(
    "Ask anything about your documents..."
)

# ----------------------------------------------------
# BUILD CHAT HISTORY
# ----------------------------------------------------

history = ""

for msg in st.session_state.messages[-6:]:

    history += f"{msg['role'].upper()}: {msg['content']}\n"

# =====================================================
# CHAT INPUT PROCESSING (FIXED LOGIC FLOW & INDENTATION)
# =====================================================

if question:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message("user"):

        st.markdown(question)

    with st.chat_message("assistant"):

        if not st.session_state.indexed:

            answer = "Please create the Knowledge Base first."

            st.markdown(answer)

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer
                }
            )

        else:

            with st.spinner("Thinking..."):

                agent_prompt = AGENTS[
                    st.session_state.selected_agent
                ]

                final_prompt = f"""
{agent_prompt}

You are having a continuous conversation.

Previous Conversation:

{history}

Answer ONLY using the uploaded documents.

Current User Question:

{question}

If the answer is not available in the uploaded documents, clearly say so.
"""

                result = st.session_state.qa_chain.invoke(
                    {
                        "query": final_prompt
                    }
                )

                answer = result["result"]

                sources = result["source_documents"]

            st.markdown(answer)

            confidence = min(
                98,
                75 + (len(sources) * 6)
            )

            st.progress(confidence / 100)

            st.caption(
                f"Confidence : {confidence}%"
            )

            if sources:

                with st.expander("View Sources"):

                    for doc in sources:

                        source = doc.metadata.get(
                            "source",
                            "Unknown"
                        )

                        page = doc.metadata.get(
                            "page",
                            "-"
                        )

                        st.write(
                            f"📄 {os.path.basename(source)} | Page {page}"
                        )

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "sources": sources
                }
            )

# =====================================================
# ENTERPRISE KPI DASHBOARD
# =====================================================

st.divider()

st.markdown("""
<div class="glass-card">

<h3>📊 Enterprise KPI Dashboard</h3>

<p>Visualize key business metrics and performance indicators.</p>

</div>
""", unsafe_allow_html=True)

k1, k2, k3, k4 = st.columns(4)

documents = len(st.session_state.documents)

messages = len(st.session_state.messages)

status = "Online" if st.session_state.indexed else "🔴 Offline"

llm = "Ready" if st.session_state.qa_chain else "🔴 Waiting"

with k1:

    st.metric(
        "📄 Documents",
        documents
    )

with k2:

    st.metric(
        "💬 Conversations",
        messages
    )

with k3:

    st.metric(
        "🗄 Vector Database",
        status
    )

with k4:

    st.metric(
        "🤖 AI Engine",
        llm
    )

# FIXED: Moved row 2 definition outside of "with k4" block to fix alignment
k5, k6, k7, k8 = st.columns(4)

total_words = sum(
    len(msg["content"].split())
    for msg in st.session_state.messages
)

uploaded_size = sum(
    f.size
    for f in st.session_state.documents
)

with k5:

    st.metric(
        "📝 Total Words",
        total_words
    )

with k6:

    st.metric(
        "📦 Upload Size",
        f"{round(uploaded_size/1024,2)} KB"
    )

with k7:

    st.metric(
        "🧠 Active Agent",
         st.session_state.selected_agent
    )

with k8:

    st.metric(
        "⚡ AI Status",
        "Running" if st.session_state.indexed else "Idle"
    )

# FIXED: Removed divider from with k8 block indentation
st.divider()

progress = 0

if st.session_state.indexed:
    progress += 25

if documents > 0:
    progress += 25

if messages > 0:
    progress += 25

if st.session_state.qa_chain:
    progress += 25

st.subheader("🚀 Workspace Readiness")

st.progress(progress/100)

st.caption(f"Workspace Completion : {progress}%")

# =====================================================
# DOCUMENT LIST
# =====================================================

st.subheader("📂 Uploaded Documents")

if len(st.session_state.documents) == 0:

    st.info("No documents uploaded.")

else:

    for file in st.session_state.documents:

        size = round(file.size / 1024,2)

        c1,c2,c3 = st.columns([5,2,2])

        with c1:

            st.write(f"📄 {file.name}")

        with c2:

            st.write(f"{size} KB")

        with c3:
         st.write(file.name.split(".")[-1].upper())

st.divider()

# =====================================================
# SYSTEM INFO
# =====================================================

st.subheader("⚙ System Information")

sys1,sys2 = st.columns(2)

with sys1:

    st.info(
        f"""
Embedding Model

sentence-transformers/all-MiniLM-L6-v2
"""
    )

with sys2:

    st.info(
        f"""
LLM

llama-3.3-70b-versatile
"""
    )

# =====================================================
# CSV ANALYTICS
# =====================================================

st.markdown("""
<div class="glass-card">

<h3>📊 Analytics Dashboard</h3>

</div>
""", unsafe_allow_html=True)
csv_files = [
    file
    for file in st.session_state.documents
    if file.name.endswith(".csv")
]

if len(csv_files) == 0:

    st.info("Upload a CSV file to unlock analytics.")

else:

    selected_csv = st.selectbox(

        "Choose CSV",

        [f.name for f in csv_files]

    )

    current_csv = next(

        f for f in csv_files

        if f.name == selected_csv

    )

    current_csv.seek(0)

    df = pd.read_csv(current_csv)

    st.subheader("Preview")

    st.dataframe(df, use_container_width=True)

    c1,c2,c3,c4 = st.columns(4)

    with c1:

        st.metric("Rows", len(df))

    with c2:

        st.metric("Columns", len(df.columns))

    with c3:

        st.metric(

            "Missing",

            int(df.isna().sum().sum())

        )

    with c4:

        st.metric(

            "Duplicates",

            int(df.duplicated().sum())

        )

    st.subheader("Column Types")

    st.write(df.dtypes)

    # =====================================================
    # INTERACTIVE CHARTS
    # =====================================================

    # FIXED: Indented all charting logic by 4 spaces so it only runs safely inside the CSV "else:" block
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    categorical_cols = df.select_dtypes(exclude="number").columns.tolist()

    st.divider()
    st.subheader("📈 Interactive Analytics")

    # -----------------------------
    # BAR CHART
    # -----------------------------

    if len(numeric_cols) > 0 and len(categorical_cols) > 0:

        st.markdown("### 📊 Bar Chart")

        x_col = st.selectbox(
            "Category",
            categorical_cols,
            key="bar_x"
        )

        y_col = st.selectbox(
            "Value",
            numeric_cols,
            key="bar_y"
        )

        bar_df = df.groupby(x_col)[y_col].sum().reset_index()

        fig = px.bar(
            bar_df,
            x=x_col,
            y=y_col,
            color=y_col,
            title=f"{y_col} by {x_col}"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # -----------------------------
    # PIE CHART
    # -----------------------------

    if len(categorical_cols) > 0:

        st.markdown("### 🥧 Pie Chart")

        pie_col = st.selectbox(
            "Pie Category",
            categorical_cols,
            key="pie"
        )

        pie_df = df[pie_col].value_counts().reset_index()

        pie_df.columns = [pie_col, "Count"]

        fig = px.pie(
            pie_df,
            names=pie_col,
            values="Count",
            hole=0.45
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # -----------------------------
    # HISTOGRAM
    # -----------------------------

    if len(numeric_cols) > 0:

        st.markdown("### 📉 Histogram")

        hist_col = st.selectbox(
            "Histogram Column",
            numeric_cols,
            key="hist"
        )

        fig = px.histogram(
            df,
            x=hist_col,
            nbins=30,
            title=f"Distribution of {hist_col}"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # -----------------------------
    # LINE CHART
    # -----------------------------

    if len(numeric_cols) >= 2:

        st.markdown("### 📈 Line Chart")

        x_axis = st.selectbox(
            "X Axis",
            df.columns,
            key="line_x"
        )

        y_axis = st.selectbox(
            "Y Axis",
            numeric_cols,
            key="line_y"
        )

        fig = px.line(
            df,
            x=x_axis,
            y=y_axis,
            markers=True,
            title=f"{y_axis} Trend"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # -----------------------------
    # SCATTER PLOT
    # -----------------------------

    if len(numeric_cols) >= 2:

        st.markdown("### 🔵 Scatter Plot")

        x = st.selectbox(
            "Scatter X",
            numeric_cols,
            key="scatter_x"
        )

        y = st.selectbox(
            "Scatter Y",
            numeric_cols,
            key="scatter_y"
        )

        fig = px.scatter(
            df,
            x=x,
            y=y,
            color=y,
            title=f"{x} vs {y}"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

# =====================================================
# AI BUSINESS INSIGHTS
# =====================================================

st.divider()

st.markdown("""
<div class="glass-card">

<h3>🤖 AI Business Insights</h3>

<p>Generate AI-powered business insights, trends, risks, opportunities, and strategic recommendations from your uploaded data.</p>

</div>
""", unsafe_allow_html=True)

if st.button("Generate AI Insights", use_container_width=True):

    if not st.session_state.indexed:

        st.warning(
            "Please create the Knowledge Base first."
        )

    elif "df" not in locals():

        st.warning(
            "Please upload or select a CSV file first."
        )

    else:

        with st.spinner("Analyzing business data..."):

            summary = f"""
Dataset Summary

Rows : {len(df)}

Columns : {len(df.columns)}

Column Names :

{', '.join(df.columns)}

Statistics :

{df.describe(include='all').fillna('').to_string()}

Missing Values :

{df.isna().sum().to_string()}
"""

            agent_prompt = AGENTS[
                st.session_state.selected_agent
            ]

            final_prompt = f"""
{agent_prompt}

You are an experienced Business Intelligence Consultant.

Analyze the uploaded dataset.

Generate:

1. Executive Summary

2. Key KPIs

3. Business Trends

4. Risks

5. Opportunities

6. Recommendations

7. Important Insights

Use bullet points.

Dataset:

{summary}
"""

            result = st.session_state.qa_chain.invoke(
                {
                    "query": final_prompt
                }
            )

            insights = result["result"]

        st.success("Analysis Complete!")

        st.markdown(insights)

st.divider()

# =====================================================
# EXECUTIVE PDF REPORT
# =====================================================

def generate_pdf_report():

    filename = "BizPilot_Executive_Report.pdf"

    doc = SimpleDocTemplate(
        filename,
        pagesize=letter
    )

    styles = getSampleStyleSheet()

    story = []

    story.append(
        Paragraph(
            "<b><font size=20>BizPilot AI</font></b>",
            styles["Title"]
        )
    )

    story.append(
        Paragraph(
            "Enterprise Business Intelligence Report",
            styles["Heading2"]
        )
    )

    story.append(
        Paragraph(
            f"Generated : {datetime.now()}",
            styles["Normal"]
        )
    )

    story.append(Spacer(1,20))

    # ------------------------------------------

    story.append(
        Paragraph(
            "<b>Knowledge Base Summary</b>",
            styles["Heading2"]
        )
    )

    table = Table(
        [

            ["Documents",
             str(len(st.session_state.documents))],

            ["Messages",
             str(len(st.session_state.messages))],

            ["Vector Database",
             "Ready" if st.session_state.indexed else "Not Ready"],

            ["LLM",
             "Groq Llama-3.3"]

        ]
    )

    table.setStyle(

        TableStyle(

            [

                ("GRID",(0,0),(-1,-1),1,colors.grey),

                ("BACKGROUND",(0,0),(-1,0),colors.lightblue),

                ("BACKGROUND",(0,0),(0,-1),colors.whitesmoke),

                ("BOTTOMPADDING",(0,0),(-1,-1),8),

            ]

        )

    )

    story.append(table)

    story.append(Spacer(1,20))

    # ------------------------------------------

    story.append(
        Paragraph(
            "<b>Conversation History</b>",
            styles["Heading2"]
        )
    )

    for msg in st.session_state.messages:

        role = msg["role"].upper()

        content = msg["content"]

        story.append(

            Paragraph(

                f"<b>{role}</b><br/>{content}",

                styles["BodyText"]

            )

        )

        story.append(Spacer(1,8))

    doc.build(story)

    return filename

# =====================================================
# CHAT EXPORT
# =====================================================

import json

st.divider()

st.markdown("""
<div class="glass-card">

<h3>💾 Export Conversation</h3>

<p>Download your AI conversation for future reference.</p>

</div>
""", unsafe_allow_html=True)

# TXT Export
conversation_txt = ""

for msg in st.session_state.messages:

    conversation_txt += f"{msg['role'].upper()}\n"

    conversation_txt += f"{msg['content']}\n\n"

st.download_button(

    "⬇ Download Conversation (.txt)",

    conversation_txt,

    file_name="BizPilot_Conversation.txt",

    mime="text/plain",

    use_container_width=True

)

# JSON Export

conversation_json = []

for msg in st.session_state.messages:

    conversation_json.append({

        "role": msg["role"],

        "content": msg["content"]

    })

st.download_button(

    "⬇ Download Conversation (.json)",

    json.dumps(conversation_json, indent=4),

    file_name="BizPilot_Conversation.json",

    mime="application/json",

    use_container_width=True

)



# =====================================================
# EXPORT REPORT
# =====================================================

st.divider()

st.markdown("""
<div class="glass-card">

<h3>📄 Executive Report</h3>

<p>Generate a professional executive business report instantly.</p>

</div>
""", unsafe_allow_html=True)

# FIXED: Using a state key tracking mechanism so the download button doesn't disappear on click
if "pdf_generated" not in st.session_state:
    st.session_state.pdf_generated = False

if st.button(
    "Generate Executive PDF",
    use_container_width=True
):
    st.session_state.pdf_path = generate_pdf_report()
    st.session_state.pdf_generated = True

if st.session_state.pdf_generated:
    with open(st.session_state.pdf_path, "rb") as f:
        st.download_button(
            "⬇ Download PDF",
            f,
            file_name=st.session_state.pdf_path,
            mime="application/pdf",
            use_container_width=True
        )

# =====================================================
# DATASET STATISTICS
# =====================================================

if "df" in locals():

    st.divider()

    st.subheader("📈 Dataset Statistics")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            "Rows",
            df.shape[0]
        )

    with c2:
        st.metric(
            "Columns",
            df.shape[1]
        )

    with c3:
        st.metric(
            "Missing Values",
            int(df.isnull().sum().sum())
        )

    with c4:
        st.metric(
            "Duplicate Rows",
            int(df.duplicated().sum())
        )

    st.divider()

    st.subheader("🧾 Column Information")

    column_info = pd.DataFrame(
        {
            "Column": df.columns,
            "Data Type": df.dtypes.astype(str),
            "Missing": df.isnull().sum(),
            "Unique Values": df.nunique()
        }
    )

    st.dataframe(
        column_info,
        use_container_width=True
    )

    numeric_df = df.select_dtypes(include="number")

    if not numeric_df.empty:

        st.divider()

        st.subheader("📊 Correlation Matrix")

        corr = numeric_df.corr()

        st.dataframe(
            corr.round(2),
            use_container_width=True
        )

        st.divider()

        st.subheader("📉 Statistical Summary")

        st.dataframe(
            numeric_df.describe(),
            use_container_width=True
        )

# =====================================================
# FOOTER
# =====================================================

st.divider()

st.markdown(
"""
### 🤖 BizPilot AI

Enterprise Multi-Agent Business Intelligence Copilot

---
"""
)

c1, c2, c3 = st.columns(3)

with c1:
    st.markdown(
        '<div class="footer"><strong>Version:</strong> 3.0</div>',
        unsafe_allow_html=True
    )

with c2:
    st.markdown(
        '<div class="footer"><strong>Powered by</strong><br>Groq + LangChain + ChromaDB</div>',
        unsafe_allow_html=True
    )

with c3:
    st.markdown(
        '<div class="footer"><strong>Developed by</strong><br>Adil Shaikh</div>',
        unsafe_allow_html=True
    )
