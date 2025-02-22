import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# Load dataset
df = pd.read_csv("C:/Users/mkcha/Downloads/house_prices.csv")  # Replace with your dataset

# Preprocessing (Assuming relevant columns)
X = df[['area', 'bedrooms', 'bathrooms', 'location_score']]  # Example features
y = df['price']

# Splitting dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f"Model MAE: {mae}")

# Save model
with open("model.pkl", "wb") as file:
    pickle.dump(model, file)
  #app.py
from flask import Flask, request, jsonify
import pickle
import numpy as np

# Load trained model (Make sure you have 'model.pkl' in the same directory)
try:
    with open("model.pkl", "rb") as file:
        model = pickle.load(file)
except FileNotFoundError:
    model = None  # Handle missing model file

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return "House Price Prediction API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model file not found. Please train and save the model as model.pkl'})

    try:
        data = request.json
        features = np.array([
            data['area'], data['bedrooms'], data['bathrooms'], data['location_score']
        ]).reshape(1, -1)
        prediction = model.predict(features)[0]
        return jsonify({'predicted_price': prediction})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True,use_reloader=False)
