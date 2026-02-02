import streamlit as st
import pandas as pd

from pipeline import recommend_workouts,recommend_meals,load_loss_model,load_gain_model,load_meal_model

# -----------------------------
# LOAD DATASET
# -----------------------------
@st.cache_data
def load_workout_data():
    return pd.read_csv("datasets/DL_workout.csv")

df_workout = load_workout_data()


st.title("🏋️ Workout and Meal Recommendation System")

# -----------------------------
# USER INPUT FORM
# -----------------------------
with st.form("user_form"):
    age = st.number_input("Age", 18, 90, 30)
    gender = st.selectbox("Gender", ["Female", "Male"])
    weight = st.number_input("Weight (kg)", 30.0, 200.0, 70.0)
    height = st.number_input("Height (m)", 1.2, 2.2, 1.65)
    experience_level=st.selectbox("Experience",[0,1,2])
    goal = st.selectbox("Goal", ["weight loss", "weight gain"])
    preferred_diet=st.multiselect("Preferred Diet Types",
    ["Vegan", "Vegetarian", "Paleo", "Low-Carb","Keto","Balanced"],
    default=["Vegetarian"])
    
    submit_btn = st.form_submit_button("Generate Workout and Meal Plan")


# -----------------------------
# RECOMMENDATION OUTPUT
# -----------------------------

@st.cache_data
def load_meal_data():
    return pd.read_csv("datasets/DL_Meal.csv")

df_meal = load_meal_data()

if submit_btn:

    user = {
        "Age": age,
        "Gender": 0 if gender == "Female" else 1,
        "Weight (kg)": weight,
        "Height (m)": height,
        "Experience": experience_level,
        "goal": goal,
        "preferred_diet":preferred_diet
    }

    st.subheader("🔍 Recommended Exercises")

    if goal == "weight loss":
        model,scaler,feature_cols=load_loss_model()
        model_meal, scaler_meal, feature_cols_meal = load_meal_model()

    elif goal == "weight gain":
        model,scaler,feature_cols=load_gain_model()
        model_meal, scaler_meal, feature_cols_meal = load_meal_model()

    
    # if model is None or scaler is None:
    #     st.error(" Model not loaded properly")
    #     st.stop()

    recs = recommend_workouts(
        user_input=user,
        exercise_df=df_workout,
        model=model,
        scaler=scaler
    )

    st.dataframe(recs)
    st.subheader("🔍 Recommended Meals")
    preferred_diet = user["preferred_diet"]
    
    filtered_meals = df_meal[df_meal["diet_type_raw"].isin(preferred_diet)]
    meal_recs = recommend_meals(
        user_input=user,
        meal_df=filtered_meals,
        model=model_meal,
        scaler=scaler_meal
    )

    st.dataframe(meal_recs)
