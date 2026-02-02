import streamlit as st
from ML_Logic import build_user, predict_wellness
from rag_pipeline import build_rag_new_prompt,run_rag



st.title("🧠 Wellness Check")

if "ml_output" not in st.session_state:
    st.session_state.ml_output = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
    
    
st.title(" WELLNESS LABEL SYSTEM")

# -----------------------------
# USER INPUT FORM
# -----------------------------
with st.form("user_form"):
    age = st.number_input("Age", 18, 90, 30)
    gender = st.selectbox("Gender", ["Female", "Male"])
    weight = st.number_input("Weight (kg)", 30.0, 200.0, 70.0)
    height = st.number_input("Height (m)", 1.2, 2.2, 1.65)
    water_intake=st.number_input("Water_Intake (liters)",1,5,3)
    workout_frequency=st.number_input("Workout_Frequency (days/week)",0,7,5)
    daily_meal_frequency=st.number_input("Daily meals frequency")
    calories=st.number_input("Calories",1500,4500,2200)
    
    
    submit_btn = st.form_submit_button("Generate my wellness score")



if submit_btn:

    user = {
        "Age": age,
        "Gender": 0 if gender == "Female" else 1,
        "Weight (kg)": weight,
        "Height (m)": height,
        "Water_Intake (liters)":water_intake,
        "Workout_Frequency (days/week)":workout_frequency,
        "Daily meals frequency":daily_meal_frequency,
        "Calories":calories
    }

    st.subheader("Wellness Result")
    x=build_user(user_input=user)
    ml_output=predict_wellness(x)
    
    st.session_state.ml_output = ml_output
    st.session_state.chat_history = []  # reset chat for new user

    
    st.success(f"Wellness Status: {ml_output['user_status']}  "
               f"Confidence:{ml_output['confidence']:.2f})"
    )
    
    
    rag_prompt=build_rag_new_prompt(ml_output,
            st.session_state.chat_history)
    context, answer =run_rag(rag_prompt)

    st.subheader("Wellness Advice")
    st.write(answer)
    
    st.page_link("pages/Wellness_Chat.py", label="💬 Continue to Chat")