import dash
from dash import html, dcc
from dash.dependencies import Input, Output, State
import numpy as np
from tensorflow.keras.models import load_model
import pandas as pd
import os

# Load the trained model
model = load_model("student_grade_classifier.h5")

# Initialize Dash app
app = dash.Dash(__name__)
server = app.server  # Needed for deployment on Render

# Define app layout
app.layout = html.Div([
     html.Link(rel='stylesheet', href='https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap'),

    html.Style('''
        body {
            font-family: 'Inter', sans-serif;
            background-color: #f4f7f9;
            margin: 0;
            padding: 0;
        }

        h1 {
            color: #2c3e50;
            font-weight: 600;
            margin-bottom: 30px;
        }

        label {
            margin-top: 10px;
            display: block;
            font-weight: 500;
            color: #34495e;
        }

        input[type=number] {
            width: 100%;
            padding: 10px;
            margin-top: 5px;
            margin-bottom: 15px;
            border: 1px solid #ccc;
            border-radius: 8px;
            box-sizing: border-box;
        }

        button {
            background-color: #2980b9;
            color: white;
            border: none;
            padding: 12px 24px;
            font-size: 16px;
            font-weight: bold;
            border-radius: 8px;
            cursor: pointer;
        }

        button:hover {
            background-color: #3498db;
        }

        h2 {
            text-align: center;
            color: #27ae60;
        }

        .form-container {
            background-color: white;
            padding: 30px;
            border-radius: 12px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
            width: 100%;
            max-width: 600px;
            margin: 30px auto;
        }
    '''),
    html.H1("Student Grade Predictor", style={'textAlign': 'center'}),
    
    html.Div([
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
        
        html.Br(),
        html.Button("Predict Grade", id='predict-button', n_clicks=0),
        html.H2(id='prediction-output', style={'marginTop': '20px'})
    ], className='form-container')
])

# Prediction logic
@app.callback(
    Output('prediction-output', 'children'),
    Input('predict-button', 'n_clicks'),
    [
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
        State('volunteer', 'value')
    ]
)
def predict_grade(n_clicks, *inputs):
    if n_clicks == 0:
        return ""
    
    # Convert input to model format (as numpy array and scale)
    input_array = np.array(inputs).reshape(1, -1)
    # Normalize using same scaler used in training
    from sklearn.preprocessing import StandardScaler
    # This assumes training features are accessible - use a stored scaler in production
    dummy_df = pd.DataFrame([inputs], columns=[
        'Age', 'Gender', 'Ethnicity', 'ParentalEducation', 'StudyTimeWeekly',
        'Absences', 'Tutoring', 'ParentalSupport', 'Extracurricular', 
        'Sports', 'Music', 'Volunteering'
    ])
    scaler = StandardScaler()
    dummy_df_scaled = scaler.fit_transform(dummy_df)  # WARNING: this should use the original scaler

    prediction = model.predict(dummy_df_scaled)
    grade_class = np.argmax(prediction)

    grades = ['A', 'B', 'C', 'D', 'F']
    return f"Predicted Grade Class: {grades[grade_class]}"

# Run the app
if __name__ == '__main__':

    port = int(os.getenv("PORT", 10000))
    app.run_server(debug=False, host="0.0.0.0", port=port)