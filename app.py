from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# Load model
models = joblib.load("wine_models.pkl")
model = models[0]

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():

    # Get values from form
    features = [
        
        float(request.form['fixed_acidity']),
        float(request.form['volatile_acidity']),
        float(request.form['citric_acid']),
        float(request.form['residual_sugar']),
        float(request.form['chlorides']),
        float(request.form['free_sulfur_dioxide']),
        float(request.form['total_sulfur_dioxide']),
        float(request.form['density']),
        float(request.form['ph']),
        float(request.form['sulphates']),
        float(request.form['alcohol']),
        float(request.form['id'])
    ]

    # Convert to numpy array
    final_input = np.array([features])

    # Predict
    prediction = model.predict(final_input)

    quality_score = prediction[0]

    # Convert prediction into label
    if quality_score >= 7:
        result = "Excellent Wine"

    elif quality_score >= 5:
        result = "Good Wine"

    else:
        result = "Poor Wine"

    # Return result to HTML
    return render_template(
        'index.html',
        prediction_text=f'Predicted Quality: {quality_score} ({result})'
    )


if __name__ == "__main__":
    app.run(debug=True)
