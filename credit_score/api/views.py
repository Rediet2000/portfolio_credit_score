from h11 import Response
import joblib
import numpy as np
import os
import pandas as pd
from django.shortcuts import render

from credit_score.api.utils import encode_categorical_features, handle_missing_values, normalize_features
from .form import PredictionForm
import os, sys
sys.path.append(os.path.abspath(os.path.join('..')))
# Load the saved model

model_path = os.path.join(os.path.dirname(__file__), './model/best_credit_scoring_model.pkl')
model = joblib.load(model_path)
#credit_api\scoring\model\best_credit_scoring_model.pkl

# Define an API to serve predictions
@api_view(['POST']) # type: ignore
def predict_credit_risk(request):
    """
    API endpoint to make a prediction given a set of input features.

    Parameters:
        request (Request): A POST request containing the input features as a dictionary of lists

    Returns:
        Response: A JSON response containing the prediction as a list

    Raises:
        Response: A 400 error response if no data is provided
    """

    data = request.data.get('features', None)
    
    if data:
        # Create DataFrame from the input features (Assuming input as a dictionary of lists)
        df = pd.DataFrame([data])
        
        # Apply the preprocessing steps
        df = handle_missing_values(df)
        categorical_columns = ['ProductCategory', 'ChannelId', 'PricingStrategy', 'FraudResult']
        df = encode_categorical_features(df, categorical_columns)
        df = normalize_features(df)
        
        # Make prediction
        features = df.to_numpy()  # Convert preprocessed DataFrame to numpy array
        prediction = model.predict(features)
        return Response({'prediction': prediction.tolist()})
    else:
        return Response({'error': 'No data provided'}, status=400)

# UI for submitting data
def home(request):
    if request.method == 'POST':
        form = PredictionForm(request.POST)
        if form.is_valid():
            # Extract form values into a DataFrame
            data = {field: form.cleaned_data[field] for field in form.fields}
            df = pd.DataFrame([data])
            
            # Apply preprocessing
            df = handle_missing_values(df)
            categorical_columns = ['ProductCategory', 'ChannelId', 'PricingStrategy', 'FraudResult']
            df = encode_categorical_features(df, categorical_columns)
            #df = normalize_features(df)
            
            # Make prediction
            features = df.to_numpy()
            prediction = model.predict(features)
            return render(request, 'home.html', {'form': form, 'prediction': prediction[0]})
    else:
        form = PredictionForm()
        return render(request, 'home.html', {'form': form})