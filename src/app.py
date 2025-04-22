import dash
from dash import html, dcc
from dash.dependencies import Input, Output, State
import numpy as np
from tensorflow.keras.models import load_model
import pandas as pd
import os
import joblib

# Construct paths relative to app.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "..", "artifacts")
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "student_grade_classifier.h5")
SCALER_PATH = os.path.join(ARTIFACTS_DIR, "scaler.joblib")

# Debug paths
print("Current Working Directory:", os.getcwd())
print("Artifacts directory:", ARTIFACTS_DIR)
print("Model file exists:", os.path.exists(MODEL_PATH))
print("Scaler file exists:", os.path.exists(SCALER_PATH))

# Load the trained model and scaler
try:
    model = load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print("Scaler feature names:", scaler.feature_names_in_)
    print("Model input shape:", model.input_shape)
except Exception as e:
    print(f"Error loading model or scaler: {e}")
    raise

# Initialize Dash app
app = dash.Dash(__name__)
server = app.server  # Needed for deployment on Render

# Define app layout
app.layout = html.Div([
    html.H1("Student Grade Predictor", style={'textAlign': 'center'}),
    html.Div([
        html.Label("Student ID (e.g., 1000-3392):"),
        dcc.Input(id='studentid', type='number', min=1000, max=3392, step=1, value=2000),
        html.Label("Age:"),
        dcc.Input(id='age', type='number', min=15, max=18, step=1, value=16),
        html.Label("Gender (0 = Male, 1 = Female):"),
        dcc.Input(id='gender', type='number', min=0, max=1, step=1, value=0),
        html.Label("Ethnicity (0=Caucasian, 1=AA, 2=Asian, 3=Other):"),
        dcc.Input(id='ethnicity', type='number', min=0, max=3, step=1, value=0),
        html.Label("Parental Education (0-4):"),
        dcc.Input(id='parentedu', type='number', min=0, max=4, step=1, value=2),
        html.Label("Study Time per Week (hours):"),
        dcc.Input(id='studytime', type='number', min=0, max=20, value=5),
        html.Label("Absences:"),
        dcc.Input(id='absences', type='number', min=0, max=30, value=2),
        html.Label("Tutoring (0=No, 1=Yes):"),
        dcc.Input(id='tutoring', type='number', min=0, max=1, step=1, value=0),
        html.Label("Parental Support (0-4):"),
        dcc.Input(id='parentsupport', type='number', min=0, max=4, step=1, value=3),
        html.Label("Extracurricular (0=No, 1=Yes):"),
        dcc.Input(id='extra', type='number', min=0, max=1, value=1),
        html.Label("Sports (0=No, 1=Yes):"),
        dcc.Input(id='sports', type='number', min=0, max=1, value=0),
        html.Label("Music (0=No, 1=Yes):"),
        dcc.Input(id='music', type='number', min=0, max=1, value=0),
        html.Label("Volunteering (0=No, 1=Yes):"),
        dcc.Input(id='volunteer', type='number', min=0, max=1, value=0),
        html.Label("GPA (0-4):"),
        dcc.Input(id='gpa', type='number', min=0, max=4, step=0.1, value=2.5),
        html.Br(),
        html.Button("Predict Grade", id='predict-button', n_clicks=0),
        html.H2(id='prediction-output', style={'color': 'red', 'fontSize': '24px', 'marginTop': '20px'})
    ], className='form-container')
])

# Prediction logic
@app.callback(
    Output('prediction-output', 'children'),
    Input('predict-button', 'n_clicks'),
    [
        State('studentid', 'value'),
        State('age', 'value'),
        State('gender', 'value'),
        State('ethnicity', 'value'),
        State('parentedu', 'value'),
        State('studytime', 'value'),
        State('absences', 'value'),
        State('tutoring', 'value'),
        State('parentsupport', 'value'),
        State('extra', 'value'),
        State('sports', 'value'),
        State('music', 'value'),
        State('volunteer', 'value'),
        State('gpa', 'value')
    ]
)
def predict_grade(n_clicks, *inputs):
    print(f"Button clicked: {n_clicks}")
    print(f"Inputs: {inputs}")
    if n_clicks == 0:
        print("No clicks yet, returning empty")
        return ""
    
    if any(x is None for x in inputs):
        print("Error: One or more inputs are missing")
        return "Error: Please fill in all fields"
    
    try:
        inputs = [float(x) for x in inputs]
        input_array = np.array(inputs).reshape(1, -1)
        print(f"Input array: {input_array}")
        dummy_df = pd.DataFrame([inputs], columns=[
            'StudentID', 'Age', 'Gender', 'Ethnicity', 'ParentalEducation', 'StudyTimeWeekly',
            'Absences', 'Tutoring', 'ParentalSupport', 'Extracurricular', 
            'Sports', 'Music', 'Volunteering', 'GPA'
        ])
        print(f"DataFrame: {dummy_df}")
        dummy_df_scaled = scaler.transform(dummy_df)
        print(f"Scaled Data: {dummy_df_scaled}")
        prediction = model.predict(dummy_df_scaled)
        print(f"Prediction: {prediction}")
        grade_class = np.argmax(prediction)
        print(f"Grade class: {grade_class}")
        grades = ['A', 'B', 'C', 'D', 'F']
        return f"Predicted Grade Class: {grades[grade_class]}"
    except Exception as e:
        print(f"Error in prediction: {e}")
        return f"Error: {str(e)}"

# Run the app
if __name__ == '__main__':
    port = int(os.getenv("PORT", 10000))
    app.run_server(debug=True, host="0.0.0.0", port=port)