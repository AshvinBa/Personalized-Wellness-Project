import streamlit as st
from rag_pipeline import build_rag_new_prompt, run_rag

st.title("💬 Wellness Chat")

if "ml_output" not in st.session_state or st.session_state.ml_output is None:
    st.warning("Please complete the wellness check first.")
    st.page_link("pages/1_Wellness_Check.py", label="Go to Wellness Check")
    st.stop()

user_question = st.text_input("Ask a follow-up question")

if st.button("Ask") and user_question.strip():
    rag_prompt = build_rag_new_prompt(
        st.session_state.ml_output,
        st.session_state.chat_history,
        user_question
    )

    _, answer = run_rag(rag_prompt)
    st.session_state.chat_history.append((user_question, answer))

    st.write("### Answer")
    st.write(answer)

