import sys
from django.http import JsonResponse
from django.shortcuts import render
import pandas as pd
import os, sys

from Scripts.analysis_script import load_data
from Scripts.future_engineering import aggregate_features, encode_categorical_data, extract_features
from Scripts.model_script import evaluate_model, predict_new_data, split_data, train_model
from Scripts.WoE import divide_good_bad, rfms_score, woe_binning
sys.path.append(os.path.abspath(os.path.join('..')))

# from credit_score.api.WoE import divide_good_bad, rfms_score, woe_binning
# from credit_score.api.future_engineering import aggregate_features, encode_categorical_data, extract_features
# from credit_score.api.model_script import evaluate_model, predict_new_data, split_data, train_model
def home(request):
    """
    Home view. Returns the rendered index.html page.

    :param request: WSGIRequest
    :return: HttpResponse
    """
    return render(request, 'index.html')

# Train model
def train_model_api(request):
    if request.method == 'POST':
        try:
            df = load_data('../Datas/data.csv')
            df = aggregate_features(df)
            df = extract_features(df)
            df = encode_categorical_data(df)

            X_train, X_test, y_train, y_test = split_data(df)
            model = train_model(X_train, y_train)
            accuracy = evaluate_model(model, X_test, y_test)

            # Save model
            import pickle
            with open('model.pkl', 'wb') as f:
                pickle.dump(model, f)

            return JsonResponse({"message": "Model trained successfully", "accuracy": accuracy})
        except Exception as e:
            return JsonResponse({"error": str(e)})
    return JsonResponse({"error": "Invalid request method"})

# Predict
def predict_api(request):
    if request.method == 'POST':
        try:
            import pickle
            with open('model.pkl', 'rb') as f:
                model = pickle.load(f)

            data = pd.DataFrame(request.POST.dict())
            data = aggregate_features(data)
            data = extract_features(data)
            data = encode_categorical_data(data)

            predictions = predict_new_data(model, data)
            return JsonResponse({"predictions": predictions.tolist()})
        except Exception as e:
            return JsonResponse({"error": str(e)})
    return JsonResponse({"error": "Invalid request method"})

# RFMS Scoring
def calculate_rfms_api(request):
    if request.method == 'POST':
        try:
            file = request.FILES['file']
            df = pd.read_csv(file)
            customer_metrics = rfms_score(df)
            return JsonResponse({"RFMS_Sample": customer_metrics.head(5).to_dict()})
        except Exception as e:
            return JsonResponse({"error": str(e)})
    return JsonResponse({"error": "Invalid request method"})

# Categorize customers
def categorize_customers_api(request):
    if request.method == 'POST':
        try:
            file = request.FILES['file']
            df = pd.read_csv(file)
            customer_metrics = rfms_score(df)
            categorized_df = divide_good_bad(df, customer_metrics)
            return JsonResponse({"Sample": categorized_df.head(5).to_dict()})
        except Exception as e:
            return JsonResponse({"error": str(e)})
    return JsonResponse({"error": "Invalid request method"})

# WoE Binning
def calculate_woe_api(request):
    if request.method == 'POST':
        try:
            file = request.FILES['file']
            df = pd.read_csv(file)
            target = request.POST['target']
            features = request.POST.getlist('features')
            bins, iv_values = woe_binning(df, target, features)
            return JsonResponse({"IV_Values": iv_values})
        except Exception as e:
            return JsonResponse({"error": str(e)})
    return JsonResponse({"error": "Invalid request method"})
