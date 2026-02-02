import streamlit as st
import pandas as pd
import pickle

from rag_pipeline import build_rag_new_prompt,run_rag
with open("ML_model.pkl", "rb") as f:
    ml_model = pickle.load(f)



def activity_multiplier(workout_days_per_week):
    if workout_days_per_week == 0:
        return 1.2
    elif 1 <= workout_days_per_week <= 2:
        return 1.375
    elif 3 <= workout_days_per_week <= 4:
        return 1.55
    elif 5 <= workout_days_per_week <= 6:
        return 1.725
    else:  # 7 days
        return 1.9
   
def build_user(user_input):
    bmi=user_input["Weight (kg)"]/(user_input["Height (m)"]**2)
    height_cm=user_input["Height (m)"] * 100
    if user_input["Gender"]==1:
        bmr = (10 * user_input["Weight (kg)"]) + (6.25 *height_cm) - (5 * user_input["Age"]) + 5
    else:
        bmr = (10 * user_input["Weight (kg)"]) + (6.25 * height_cm) - (5 * user_input["Age"]) -161
        
        
    multiplier=activity_multiplier(user_input["Workout_Frequency (days/week)"])
    
    tdee=multiplier * bmr
    cal_balance=user_input["Calories"]-tdee
    
    return pd.DataFrame([{
        "Age": user_input["Age"],
        "Gender": user_input["Gender"],
        "Weight (kg)": user_input["Weight (kg)"],
        "Height (m)": user_input["Height (m)"],
        "BMI": bmi,
        "Water_Intake (liters)": user_input["Water_Intake (liters)"],
        "Workout_Frequency (days/week)": user_input["Workout_Frequency (days/week)"],
        "Daily meals frequency":user_input["Daily meals frequency"],
        "Calories":user_input["Calories"],
        'TDEE':tdee,
        'cal_balance':cal_balance
    }])
    
    
def predict_wellness(user_input):
    label=ml_model.predict(user_input)[0]
    proba=ml_model.predict_proba(user_input)[0]
    confidence=max(proba)
    label_map = {
                    0: "Unhealthy",
                    1: "At Risk",
                    2: "Healthy"
                }

    user_status = label_map[label]
    bmi = user_input["BMI"].iloc[0]
    workout_frequency=user_input["Workout_Frequency (days/week)"].iloc[0]
    water_intake = float(user_input["Water_Intake (liters)"].iloc[0])
    gender=user_input["Gender"].iloc[0]
    calories=user_input["Calories"].iloc[0]
    
    risk_factors=[]
    if bmi >= 25:
        risk_factors.append("high_bmi")
    if workout_frequency < 3:
        risk_factors.append("low_activity")
    if water_intake < 2:
        risk_factors.append("low_hydration")
    if gender==0:
        if calories>=2400:
            risk_factors.append("high_calories_intake")
    else:
        if calories>3000:
            risk_factors.append("high_calories_intake")
        

    return {
    "user_status": user_status,
    "confidence": confidence,
    "risk_factors": risk_factors
        
    }

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

if st.session_state.ml_output is not None:

    st.subheader("Ask a follow-up question")

    user_question = st.text_input("Your question")

    if st.button("Ask") and user_question.strip():

        rag_prompt = build_rag_new_prompt(
            st.session_state.ml_output,
            st.session_state.chat_history,
            user_question
        )

        context, answer = run_rag(rag_prompt)

        # save conversation
        st.session_state.chat_history.append((user_question, answer))

        st.write("### Answer")
        st.write(answer)
