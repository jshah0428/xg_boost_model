import dash
import base64
import io
from dash import dcc, html, Input, Output, State
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score
import plotly.express as px

# Initialize app
app = dash.Dash(__name__)

# Global variables
uploaded_data = None
model = None

# App layout
app.layout = html.Div([
    html.H1("Regression Prediction App"),

    # Upload Component
    html.Div([
        html.H2("1. Upload Dataset"),
        dcc.Upload(
            id='upload-data',
            children=html.Button('Upload CSV File'),
            multiple=False
        ),
        html.Div(id='upload-feedback')
    ], style={'margin-bottom': '20px'}),

    # Select Target Component
    html.Div([ #this is where target variables are selected.
        html.H2("2. Select Target Variable"),
        dcc.Dropdown(id='target-dropdown', placeholder='Select the target variable'),
        html.Div(id='target-feedback')
    ], style={'margin-bottom': '20px'}),

    # Bar Charts Component
    html.Div([
        html.H2("3. Analyze Data"),
        html.Div([
            dcc.RadioItems(id='categorical-radio', inline=True),
            dcc.Graph(id='category-barchart')
        ]),
        dcc.Graph(id='correlation-barchart')
    ], style={'margin-bottom': '20px'}),

    # Train Component
    html.Div([
        html.H2("4. Train Model"),
        html.Div([dcc.Checklist(
            id='features-checkboxes')]),
        html.Button('Train Model', id='train-button'),
        html.Div(id='model-feedback')
    ], style={'margin-bottom': '20px'}),

    # Predict Component
    html.Div([
        html.H2("5. Make Predictions"),
        dcc.Input(id='predict-input', placeholder='Enter feature values...', style={'width': '80%'}),
        html.Button('Predict', id='predict-button'),
        html.Div(id='prediction-output')
    ])
])

# _______________helper functions_____________________________
# def preprocessing(uploaded_da):
#     df = uploaded_da
#     print(df)
#     # #did this for data cleansing so that we can handle missing data better.
#     ##this doesn't do anything because we know that our data does nto have any null values in the numerical columns, but just for good practice.
#     numeric_columns = df.select_dtypes(include=[np.number]).columns
#     df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())
#
#     # use one hot encoding so that the machine learning model so that all values are treated equally.
#     encoder = OneHotEncoder(drop='first', sparse_output=False)
#     categorical_columns = df.select_dtypes(include=['object', 'category']).columns
#     encoded_df= encoder.fit_transform(categorical_columns)
#     encoded_df_columns = encoder.get_feature_names_out(categorical_columns)
#
#
#     # put the encoded features back into the data set.
#     df_encoded = pd.concat(
#         [df.drop(columns=categorical_columns),
#          pd.DataFrame(encoded_df, columns=encoded_df_columns)],
#         axis=1
#     )
#
#     # got rid out outliers so that predictions could be more accurate.
#     Q1 = df_encoded.quantile(0.25)
#     Q3 = df_encoded.quantile(0.75)
#     IQR = Q3 - Q1
#     df_encoded = df_encoded[~((df_encoded < (Q1 - 1.5 * IQR)) | (df_encoded > (Q3 + 1.5 * IQR))).any(axis=1)]
#
#     return df_encoded
# _______________Upload Component_____________________________

@app.callback(
    Output('upload-feedback', 'children'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename')
)
def handle_upload(contents, filename):
    global uploaded_data
    if contents:
        try:
            content_type, content_string = contents.split(',')
            decoded = base64.b64decode(content_string)
            uploaded_data = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
            return f"Uploaded file: {filename} (Rows: {uploaded_data.shape[0]}, Columns: {uploaded_data.shape[1]})"
        except Exception as e:
            return f"Error reading file: {e}"
    return "No file uploaded."


# _______________Target Selection Component_____________________________

@app.callback(
    Output('target-dropdown', 'options'),
    Input('upload-feedback', 'children')
)
def update_target_dropdown(upload_feedback):
    if uploaded_data is not None:
        numerical_cols = uploaded_data.select_dtypes(include=np.number).columns
        return [{'label': col,  'value': col} for col in numerical_cols]
    return []


# _______________Bar Charts_____________________________
@app.callback(
    [Output('categorical-radio', 'options'),
     Output('category-barchart', 'figure'),
     Output('correlation-barchart', 'figure')],
    [Input('target-dropdown', 'value'),
     Input('categorical-radio', 'value')]
)
def update_barcharts(target, categorical_var):
    if uploaded_data is None or target is None:
        return [], {}, {}

    # Filter numeric columns only for correlation computation
    numeric_data = uploaded_data.select_dtypes(include=[np.number])

    # Categorical columns for dropdown
    cat_cols = uploaded_data.select_dtypes(include=['object']).columns
    categorical_options = [{'label': col, 'value': col} for col in cat_cols]

    # First bar chart (categorical variable vs target)
    if categorical_var:
        avg_values = uploaded_data.groupby(categorical_var)[target].mean()
        category_chart = px.bar(avg_values, x=avg_values.index, y=avg_values.values, title='Average Target by Category')
    else:
        category_chart = {}

    # Second bar chart (correlation of numeric features with target)
    if numeric_data.empty:
        corr_chart = {}
    else:
        correlations = numeric_data.corr()[target].drop(target).abs().sort_values(ascending=False)
        corr_chart = px.bar(correlations, x=correlations.index, y=correlations.values, title='Correlation Strength')

    return categorical_options, category_chart, corr_chart

#---------get checkboxes
#input is the uploaded data,
#output should be the checkboxes of ALL the columns.
@app.callback(
    Output('features-checkboxes', 'options'),
    Input('upload-feedback', 'children')
    #https://chatgpt.com/share/67508cf8-e690-8002-8af3-df68537a651d
)

def get_options(uploaded_feedback):
    if uploaded_data is not None:
        return[{"label": col, "value": col} for col in uploaded_data.select_dtypes(include=np.number).columns ]
    return []


# _______________Train Component_____________________________
@app.callback(
    Output('model-feedback', 'children'),
    [Input('train-button', 'n_clicks')],
    [State('target-dropdown', 'value'),
     State('features-checkboxes', 'value')]
)
def train_model(n_clicks, target, selected_features):
    global model  # Ensuring the model is accessible globally for predictions
    
    if n_clicks is None:
        return "Click 'Train Model' to start training."

    if uploaded_data is None:
        return "Error: No data uploaded. Please upload a dataset."
    
    if not target:
        return "Error: No target variable selected."
    
    if not selected_features:
        return "Error: No features selected for training."
    
    try:
        # Extract features and target
        X = uploaded_data[selected_features]
        y = uploaded_data[target]

        # Handle missing values
        X = X.fillna(X.mean())
        y = y.fillna(y.median())

        # Split data into train/test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Define preprocessing pipeline
        numeric_features = X.select_dtypes(include=['number']).columns
        categorical_features = X.select_dtypes(include=['object']).columns

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', Pipeline([
                    ('imputer', SimpleImputer(strategy='mean')),
                    ('scaler', StandardScaler())
                ]), numeric_features),
                ('cat', Pipeline([
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('onehot', OneHotEncoder(handle_unknown='ignore'))
                ]), categorical_features)
            ])

        # Build pipeline with a regression model
        model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', XGBRegressor(n_estimators=100, max_depth=3, random_state=42))
        ])

        # Train the model
        model.fit(X_train, y_train)

        # Evaluate the model
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)

        return f"Model trained successfully! R^2 score: {r2:.4f}"

    except Exception as e:
        return f"Error during training: {e}"


# _______________Predict Component_____________________________
@app.callback(
    Output('prediction-output', 'children'),
    [Input('predict-button', 'n_clicks')],
    [State('predict-input', 'value'),  # Get input values
     State('features-checkboxes', 'value')]  # Get selected feature names
)
def make_prediction(n_clicks, input_values, selected_features):
    if n_clicks is None or not input_values or not selected_features:
        return "Please enter feature values and select features before predicting."

    try:
        # Convert input_values (string) to a list of floats
        input_data = [float(value) for value in input_values.split(',')]
        
        # Ensure input length matches the number of selected features
        if len(input_data) != len(selected_features):
            return f"Error: Expected {len(selected_features)} features, but got {len(input_data)}."

        # Convert input into a DataFrame with column names
        input_df = pd.DataFrame([input_data], columns=selected_features)

        # Make prediction using the trained model
        prediction = model.predict(input_df)
        return f"Prediction: {prediction[0]}"
    except ValueError as e:
        return f"Error: Invalid input format. Ensure all inputs are numeric and separated by commas. ({str(e)})"
    except Exception as e:
        return f"Error: {str(e)}"



# _______________Run App_____________________________

if __name__ == '__main__':
    app.run_server(debug=True)