import sys
from flask import Flask, render_template, request, jsonify
import pandas as pd
import os
sys.path.append(os.path.abspath(os.path.join('..')))
from .utils import *



app = Flask(__name__)

# Train model
@app.route('/train', methods=['POST'])
def train_model_api():
    try:
        # Load dataset
        df = load_data('../Datas/data.csv')

        # Data preprocessing
        df = aggregate_features(df)
        df = extract_features(df)
        df = encode_categorical_data(df)

        # Split data
        X_train, X_test, y_train, y_test = split_data(df)

        # Train the model
        model = train_model(X_train, y_train)
        accuracy = evaluate_model(model, X_test, y_test)

        # Save model
        import pickle
        with open('model.pkl', 'wb') as f:
            pickle.dump(model, f)

        return jsonify({"message": "Model trained successfully", "accuracy": accuracy})
    except Exception as e:
        return jsonify({"error": str(e)})

# Predict using trained model
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Load model
        import pickle
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)

        # Get input data
        data = request.get_json()
        df = pd.DataFrame(data)

        # Preprocess input data
        df = aggregate_features(df)
        df = extract_features(df)
        df = encode_categorical_data(df)

        # Make predictions
        predictions = predict_new_data(model, df)
        return jsonify({"predictions": predictions.tolist()})
    except Exception as e:
        return jsonify({"error": str(e)})

# Calculate RFMS scores
@app.route('/rfms', methods=['POST'])
def calculate_rfms():
    try:
        # Load dataset
        file = request.files['file']
        df = pd.read_csv(file)

        # Calculate RFMS scores
        customer_metrics = rfms_score(df)

        # Save histograms for visualization
        plot_histogram(customer_metrics)

        return jsonify({
            "message": "RFMS calculation successful",
            "RFMS_Sample": customer_metrics.head(5).to_dict()
        })
    except Exception as e:
        return jsonify({"error": str(e)})

# Categorize customers into good and bad
@app.route('/good_bad', methods=['POST'])
def categorize_customers():
    try:
        # Load dataset
        file = request.files['file']
        df = pd.read_csv(file)

        # Calculate RFMS scores
        customer_metrics = rfms_score(df)

        # Categorize customers
        categorized_df = divide_good_bad(df, customer_metrics)

        return jsonify({
            "message": "Customers categorized successfully",
            "Sample": categorized_df.head(5).to_dict()
        })
    except Exception as e:
        return jsonify({"error": str(e)})

# Perform WoE binning
@app.route('/woe', methods=['POST'])
def calculate_woe():
    try:
        # Load dataset
        file = request.files['file']
        df = pd.read_csv(file)
        target_col = request.form.get('target')
        features = request.form.getlist('features')

        # Perform WoE binning
        bins, iv_values = woe_binning(df, target_col, features)

        # Plot the first feature's WoE bins
        if features:
            plot_woe_binning(bins, features[0])

        return jsonify({
            "message": "WoE calculation successful",
            "IV_Values": iv_values,
            "Sample": bins.head(5).to_dict()
        })
    except Exception as e:
        return jsonify({"error": str(e)})

# Web interface
@app.route('/')
def home():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
