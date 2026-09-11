import streamlit as st
from retriever import retrieve
from reranker import rerank

import pandas as pd
from pathlib import Path
from datetime import datetime
######Code for evaluator

if "success_count" not in st.session_state:
    st.session_state.success_count = 0

if "failure_count" not in st.session_state:
    st.session_state.failure_count = 0

if "feedback_submitted" not in st.session_state:
    st.session_state.feedback_submitted = False

def save_feedback(question, answer, was_satisfactory):
    row = pd.DataFrame([{
        "timestamp": datetime.now().isoformat(),
        "question": question,
        "answer": answer,
        "satisfactory": was_satisfactory
    }])

    if FEEDBACK_FILE.exists():
        row.to_csv(
            FEEDBACK_FILE,
            mode="a",
            header=False,
            index=False
        )
    else:
        row.to_csv(
            FEEDBACK_FILE,
            index=False
        )

FEEDBACK_FILE = Path("feedback.csv")

companies = [
    "Apple Inc. (AAPL)",
    "Microsoft Corporation (MSFT)",
    "Amazon.com Inc.(AMZN)",
    "NVIDIA Corporation(NVDA)",
    "Meta Platforms Inc.(META)",
    "Alphabet Inc.(GOOGL)",
    "Tesla Inc.(TSLA)",
    "Broadcom Inc.(AVGO)",
    "Costco Wholesale Corporation (COST)",
    "Netflix Inc. (NFLX)",
    "Advanced Micro Devices. (AMD)",
    "Applied Materials, Inc. (AMAT)",
    "Cisco Systems Inc. (CSCO)",
    "Qualcomm Incorporated (QCOM)",
    "Intuit Inc. (INTU)",
    "Comcast Corporation (CMCSA)",
    "T-Mobile US Inc. (TMUS)",
    "Texas Instruments Incorporated (TXN)",
    "Adobe Inc. (ADBE)",
    "Palo Alto Networks Inc. (PANW)",
    "Amgen Inc. (AMGN)",
    "Starbucks Corporation (SBUX)",
    "Intuitive Surgical Inc. (ISRG)",
    "Mondelez International Inc. (MDLZ)",
    "Gilead Sciences Inc.(GILD)",
    "Regeneron Pharmaceuticals, Inc. (REGN)",
    "Vertex Pharmaceuticals Incorporated (VRTX)",
    "Micron Technology, Inc. (MU)"
]

sections = {"1":"Business", "1A": "Risk Factors", "7": "Management's Discussion and Analysis of Financial Condition and Results of Operations (MD&A)", "8":"Financial Statements and Supplementary Data"}

with st.sidebar:
    st.header("Companies")
    for item in companies:
        st.markdown(f"- {item}")

st.title("AI Financial Research Assistant")
st.text("Ask any financial question for companies listed from SEC 10-K filings sections listed below for year 2025")

st.title("Sections")

for key in sections.keys():
    st.write(f"{key}:{sections[key]}")
question = st.chat_input("Ask a financial question")
st.write(question)

if question:
    print(question)
    retrieved_chunks = retrieve(question)
    print(retrieved_chunks)
    answer = rerank(question, retrieved_chunks)
    print(answer)
    st.chat_message("assistant").write(answer)

    st.session_state.current_question = question
    st.session_state.current_answer = answer
    st.session_state.feedback_submitted = False

if "current_answer" in st.session_state:

    st.subheader("Answer")

    st.write(
        st.session_state.current_answer
    )

    st.write("Was this answer satisfactory?")

    col1, col2 = st.columns(2)

    with col1:
        if st.button(
            "👍 Yes",
            disabled=st.session_state.feedback_submitted
        ):
            st.session_state.success_count += 1

            save_feedback(
                st.session_state.current_question,
                st.session_state.current_answer,
                True
            )

            st.session_state.feedback_submitted = True
            #st.session_state.successful_questions.append(st.session_state.current_question)
            st.success("Feedback recorded.")

    with col2:
        if st.button(
            "👎 No",
            disabled=st.session_state.feedback_submitted
        ):
            st.session_state.failure_count += 1

            save_feedback(
                st.session_state.current_question,
                st.session_state.current_answer,
                False
            )

            st.session_state.feedback_submitted = True
            #st.session_state.failed_questions.append(st.session_state.current_question)
            st.error("Feedback recorded.")


st.divider()

st.subheader("Model Performance")


col1, col2, col3 = st.columns(3)

total = (
    st.session_state.success_count
    + st.session_state.failure_count
)

with col1:
    st.metric(
        "Successful Answers",
        st.session_state.success_count
    )

with col2:
    st.metric(
        "Failed Answers",
        st.session_state.failure_count
    )

with col3:
    if total > 0:
        success_rate = (
            st.session_state.success_count
            / total
            * 100
        )
    else:
        success_rate = 0

    st.metric(
        "Success Rate",
        f"{success_rate:.1f}%"
    )