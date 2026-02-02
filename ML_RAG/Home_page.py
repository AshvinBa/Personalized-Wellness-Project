import streamlit as st



st.set_page_config(page_title="AI Wellness Assistant")

st.title("🏥 AI Wellness Assistant")

st.write("""
This app helps you:
1. Analyze your wellness using ML
2. Ask follow-up questions using RAG
""")

if st.button("🧠 Wellness Check"):
    st.switch_page("pages/Wellness_Check.py")

if st.button("💬 Workout and Meal recommendation"):
    st.switch_page("pages/workout_meal.py")
