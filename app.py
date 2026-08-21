import os
import pickle
import numpy as np
from flask import Flask, render_template_string, request, jsonify

app = Flask(__name__)

# Load the pickle model
MODEL_PATH = "random_model.pkl"
model = None

if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Student Performance Prediction</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-gradient: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
            --card-bg: rgba(30, 41, 59, 0.7);
            --accent-primary: #6366f1;
            --accent-hover: #4f46e5;
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            --pass-glow: rgba(34, 197, 94, 0.4);
            --fail-glow: rgba(239, 68, 68, 0.4);
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Inter', sans-serif;
        }

        body {
            background: var(--bg-gradient);
            color: var(--text-main);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 2rem 1rem;
        }

        .container {
            width: 100%;
            max-width: 800px;
            background: var(--card-bg);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 24px;
            padding: 2.5rem;
            box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
        }

        .header {
            text-align: center;
            margin-bottom: 2rem;
        }

        .header h1 {
            font-size: 2.2rem;
            font-weight: 700;
            background: linear-gradient(to right, #818cf8, #c084fc);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
        }

        .header p {
            color: var(--text-muted);
            font-size: 0.95rem;
        }

        .form-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
            gap: 1.25rem;
        }

        .form-group {
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
        }

        .form-group label {
            font-size: 0.85rem;
            font-weight: 600;
            color: #cbd5e1;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        .form-group input, .form-group select {
            background: rgba(15, 23, 42, 0.6);
            border: 1px solid rgba(255, 255, 255, 0.15);
            border-radius: 12px;
            padding: 0.8rem 1rem;
            color: #fff;
            font-size: 0.95rem;
            outline: none;
            transition: all 0.3s ease;
        }

        .form-group input:focus, .form-group select:focus {
            border-color: var(--accent-primary);
            box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.25);
        }

        .btn-wrapper {
            margin-top: 2rem;
            text-align: center;
        }

        .predict-btn {
            width: 100%;
            padding: 1rem 2rem;
            border: none;
            border-radius: 14px;
            background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%);
            color: #ffffff;
            font-size: 1.1rem;
            font-weight: 600;
            cursor: pointer;
            position: relative;
            overflow: hidden;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            box-shadow: 0 10px 20px -5px rgba(99, 102, 241, 0.4);
        }

        .predict-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 15px 25px -5px rgba(99, 102, 241, 0.6);
        }

        .predict-btn:active {
            transform: translateY(1px);
        }

        /* Ripple Effect */
        .ripple {
            position: absolute;
            border-radius: 50%;
            background: rgba(255, 255, 255, 0.4);
            transform: scale(0);
            animation: ripple-animation 0.6s linear;
            pointer-events: none;
        }

        @keyframes ripple-animation {
            to {
                transform: scale(4);
                opacity: 0;
            }
        }

        /* Loading Spinner */
        .spinner {
            display: none;
            width: 22px;
            height: 22px;
            border: 3px solid rgba(255, 255, 255, 0.3);
            border-radius: 50%;
            border-top-color: #fff;
            animation: spin 0.8s ease-in-out infinite;
            margin: 0 auto;
        }

        @keyframes spin {
            to { transform: rotate(360deg); }
        }

        /* Result Display Card */
        .result-card {
            display: none;
            margin-top: 2rem;
            padding: 1.5rem;
            border-radius: 16px;
            text-align: center;
            transition: all 0.5s cubic-bezier(0.4, 0, 0.2, 1);
            animation: fadeIn 0.5s ease-in-out;
        }

        .result-card.pass {
            background: rgba(34, 197, 94, 0.15);
            border: 1px solid rgba(34, 197, 94, 0.4);
            box-shadow: 0 0 30px var(--pass-glow);
            color: #4ade80;
        }

        .result-card.fail {
            background: rgba(239, 68, 68, 0.15);
            border: 1px solid rgba(239, 68, 68, 0.4);
            box-shadow: 0 0 30px var(--fail-glow);
            color: #f87171;
        }

        .result-card h2 {
            font-size: 1.8rem;
            margin-bottom: 0.3rem;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
    </style>
</head>
<body>

    <div class="container">
        <div class="header">
            <h1>Student Outcome Predictor</h1>
            <p>Input student performance metrics to predict expected result</p>
        </div>

        <form id="prediction-form">
            <div class="form-grid">
                <div class="form-group">
                    <label>Age</label>
                    <input type="number" name="Age" min="15" max="60" value="20" required>
                </div>
                <div class="form-group">
                    <label>Gender</label>
                    <select name="Gender" required>
                        <option value="0">Female</option>
                        <option value="1">Male</option>
                    </select>
                </div>
                <div class="form-group">
                    <label>Department</label>
                    <select name="Department" required>
                        <option value="0">Computer Science</option>
                        <option value="1">Electrical</option>
                        <option value="2">Mechanical</option>
                        <option value="3">Civil</option>
                    </select>
                </div>
                <div class="form-group">
                    <label>Study Hours / Day</label>
                    <input type="number" step="0.1" name="Study_Hours_Per_Day" min="0" max="24" value="5.5" required>
                </div>
                <div class="form-group">
                    <label>Attendance (%)</label>
                    <input type="number" step="0.1" name="Attendance_Percentage" min="0" max="100" value="85.0" required>
                </div>
                <div class="form-group">
                    <label>Assignments Completed</label>
                    <input type="number" name="Assignments_Completed" min="0" max="50" value="10" required>
                </div>
                <div class="form-group">
                    <label>Midterm Score</label>
                    <input type="number" step="0.1" name="Midterm_Score" min="0" max="100" value="72.5" required>
                </div>
                <div class="form-group">
                    <label>Final Score</label>
                    <input type="number" step="0.1" name="Final_Score" min="0" max="100" value="78.0" required>
                </div>
            </div>

            <div class="btn-wrapper">
                <button type="submit" class="predict-btn" id="submit-btn">
                    <span id="btn-text">Predict Outcome</span>
                    <div class="spinner" id="btn-spinner"></div>
                </button>
            </div>
        </form>

        <div class="result-card" id="result-card">
            <h2 id="result-status">---</h2>
            <p id="result-details">Prediction outcome details will appear here.</p>
        </div>
    </div>

    <script>
        const form = document.getElementById('prediction-form');
        const submitBtn = document.getElementById('submit-btn');
        const btnText = document.getElementById('btn-text');
        const btnSpinner = document.getElementById('btn-spinner');
        const resultCard = document.getElementById('result-card');
        const resultStatus = document.getElementById('result-status');
        const resultDetails = document.getElementById('result-details');

        // Click Ripple Effect
        submitBtn.addEventListener('click', function (e) {
            let x = e.clientX - e.target.getBoundingClientRect().left;
            let y = e.clientY - e.target.getBoundingClientRect().top;
            let ripples = document.createElement('span');
            ripples.className = 'ripple';
            ripples.style.left = x + 'px';
            ripples.style.top = y + 'px';
            this.appendChild(ripples);
            setTimeout(() => { ripples.remove() }, 600);
        });

        // Form Submit Handler
        form.addEventListener('submit', async function (e) {
            e.preventDefault();

            // UI Loading state
            btnText.style.display = 'none';
            btnSpinner.style.display = 'block';
            submitBtn.disabled = true;
            resultCard.style.display = 'none';

            const formData = new FormData(form);

            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    body: formData
                });
                const data = await response.json();

                // Artificial delay to show smooth loading transition
                setTimeout(() => {
                    btnText.style.display = 'inline';
                    btnSpinner.style.display = 'none';
                    submitBtn.disabled = false;

                    if (data.status === 'success') {
                        resultCard.className = 'result-card ' + data.prediction.toLowerCase();
                        resultStatus.innerText = 'Predicted Result: ' + data.prediction;
                        resultDetails.innerText = `Probability: Pass (${(data.probability[1] * 100).toFixed(1)}%) | Fail (${(data.probability[0] * 100).toFixed(1)}%)`;
                        resultCard.style.display = 'block';
                    } else {
                        alert('Error: ' + data.message);
                    }
                }, 400);

            } catch (error) {
                btnText.style.display = 'inline';
                btnSpinner.style.display = 'none';
                submitBtn.disabled = false;
                alert('An error occurred during prediction.');
            }
        });
    </script>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"status": "error", "message": "Model file not found on server."}), 500

    try:
        features = [
            float(request.form["Age"]),
            float(request.form["Gender"]),
            float(request.form["Department"]),
            float(request.form["Study_Hours_Per_Day"]),
            float(request.form["Attendance_Percentage"]),
            float(request.form["Assignments_Completed"]),
            float(request.form["Midterm_Score"]),
            float(request.form["Final_Score"])
        ]

        input_array = np.array([features])
        prediction = model.predict(input_array)[0]
        
        # Get probabilities if supported by the classifier
        probabilities = [0.0, 0.0]
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_array)[0].tolist()

        return jsonify({
            "status": "success",
            "prediction": str(prediction),
            "probability": probabilities
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
