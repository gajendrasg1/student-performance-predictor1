import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

# Load dataset
data = pd.read_csv("student_data.csv")

# Input features
X = data[
    [
        "study_hours",
        "attendance",
        "previous_marks",
        "assignment_score"
    ]
]

# Target
y = data["final_score"]

# Create model
model = LinearRegression()

# Train model
model.fit(X, y)

# Save model
with open("model.pkl", "wb") as file:
    pickle.dump(model, file)

print("Model trained successfully!")