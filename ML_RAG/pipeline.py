import pickle
import torch
import torch.nn as nn
import pandas as pd
import numpy as np

# ------------------------------------------------
# MODEL DEFINITION  LOSS model
# ------------------------------------------------
class WorkoutRecommender_loss(nn.Module):
    def __init__(self, input_dim):
        super(WorkoutRecommender_loss, self).__init__()

        self.fc1 = nn.Linear(input_dim, 32)
        self.fc2 = nn.Linear(32, 16)
        self.out = nn.Linear(16, 1)
    #fc1 ,fc2=hidden layers
        
        self.relu = nn.ReLU() #for learning non linear patterns
        self.sigmoid = nn.Sigmoid() #for output between 0 and 1

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.out(x))
        return x
    
   
# ------------------------------------------------
# MODEL DEFINITION  gain model
# ------------------------------------------------



class WorkoutRecommender_gain(nn.Module):
    def __init__(self, input_dim):
        super(WorkoutRecommender_gain, self).__init__()

        self.fc1 = nn.Linear(input_dim, 32)
        self.fc2 = nn.Linear(32, 16)
        self.out = nn.Linear(16, 1)
    #fc1 ,fc2=hidden layers
        
        self.relu = nn.ReLU() #for learning non linear patterns
        self.sigmoid = nn.Sigmoid() #for output between 0 and 1

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.out(x))
        return x

# -----------------------------
# LOAD MODEL + SCALER
# -----------------------------

def load_loss_model():
    checkpoint = torch.load("workout_recommender(loss).pth", map_location="cpu",weights_only=False)

    model = WorkoutRecommender_loss(checkpoint["input_dim"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, checkpoint["scaler"], checkpoint["feature_cols"]





def load_gain_model():
    checkpoint = torch.load("workout_recommender(gain).pth", map_location="cpu",weights_only=False)

    model = WorkoutRecommender_gain(checkpoint["input_dim"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, checkpoint["scaler"], checkpoint["feature_cols"]



# -----------------------------
# RECOMMENDATION FUNCTION
# -----------------------------
  

def recommend_workouts(user_input, exercise_df, model, scaler, top_n=10):

    bmi = user_input["Weight (kg)"] / (user_input["Height (m)"] ** 2)
    model.eval()
    recommendations = []

    for _, row in exercise_df.iterrows():

        X = [
            user_input["Age"],
            user_input["Gender"],
            bmi,
            user_input["Experience"],
            row["Session_Duration (hours)"],
            row["Difficulty Level"],
            row["Reps"],
            row["strength"],
            row["cardio"],
            row["hiit"],
            row["yoga"]
        ]

        X = scaler.transform([X])
        X_t = torch.tensor(X, dtype=torch.float32)

        with torch.no_grad():
            score = model(X_t).item()
        col=["Name of Exercise","target_muscles","focus_area"]
        recommendations.append({
            "Name of Exercise": row["Name of Exercise"],
            "target_muscles": row["target_muscles"],
            "focus_area": row["focus_area"],
            "score": score
        })
 
    recommendations.sort(key=lambda x: x["score"], reverse=True)
    
# keep top-N UNIQUE exercises
    unique_recs = []
    seen_exercises = set()

    for rec in recommendations:
        name = rec["Name of Exercise"]
        if name not in seen_exercises:
            unique_recs.append(rec)
            seen_exercises.add(name)

        if len(unique_recs) == top_n:
            break

    return unique_recs

#----------------------------------------------------------------------
class MealRecommenderDL(nn.Module):
    def __init__(self, input_dim, meal_c, diet_c, cook_c):
        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )

        self.meal_head = nn.Linear(64, meal_c)
        self.diet_head = nn.Linear(64, diet_c)
        self.cook_head = nn.Linear(64, cook_c)

    def forward(self, x):
        h = self.shared(x)
        return (
            self.meal_head(h),
            self.diet_head(h),
            self.cook_head(h)
        )


def load_meal_model():
    checkpoint = torch.load("meal_recommender.pth", map_location="cpu",weights_only=False)

    
    model = MealRecommenderDL(
        input_dim=checkpoint["input_dim"],
        meal_c=checkpoint["meal_c"],
        diet_c=checkpoint["diet_c"],
        cook_c=checkpoint["cook_c"]
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, checkpoint["scaler"], checkpoint["feature_cols"]


def recommend_meals(
    user_input,
    meal_df,
    model,
    scaler,
    top_n=3
):
    bmi = user_input["Weight (kg)"] / (user_input["Height (m)"] ** 2)
    model.eval()
    results = {}
    

    # encode goal ONCE
    goal_map = {
        "weight loss": 0,
        "weight gain": 1
        }
    goal_encoded = goal_map[user_input["goal"]]

    meal_types = ["Breakfast","Lunch","Dinner","Snack"]
    for meal_name in meal_types:

        meal_candidates = meal_df[meal_df["meal_type_raw"] == meal_name]
        recommendations = []

        for _, row in meal_candidates.iterrows():

            X = [
                user_input["Age"],
                user_input["Gender"],          # encoded
                user_input["Weight (kg)"],
                user_input["Height (m)"],
                bmi,
                row["Daily meals frequency"],
                row["Carbs"],
                row["Proteins"],
                row["Fats"],
                row["Calories"],
                row["Workout_Type"],           # encoded
                row["Calories_Burned"],
                row["cal_balance"],
                goal_encoded
            ]

            X = scaler.transform([X])
            X_t = torch.tensor(X, dtype=torch.float32)

            with torch.no_grad():
                meal_o, diet_o, cook_o = model(X_t)

                score = (
                    torch.softmax(meal_o, dim=1).max().item() +
                    torch.softmax(diet_o, dim=1).max().item() +
                    torch.softmax(cook_o, dim=1).max().item()
                )
               
                recommendations.append({
                    "meal_type": meal_name,  # already string
                    "meal_name":row["meal_name_raw"],
                    "diet_name" : row["diet_type_raw"],
                    "cooking_method" :row["cooking_method_raw"],
                    "score": score
                })

        recommendations.sort(key=lambda x: x["score"], reverse=True)
        results[meal_name] = recommendations[:top_n]
        rows = []

        for meal_type, meals in results.items():
            for meal in meals:
                rows.append(meal)
        
        results_df = pd.DataFrame(rows)
    return results_df
