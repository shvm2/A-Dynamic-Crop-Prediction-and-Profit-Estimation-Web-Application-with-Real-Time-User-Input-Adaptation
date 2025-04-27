Dynamic Crop Recommendation & Profit Estimation

An end-to-end web application that predicts optimal crops based on soil and environmental inputs, quantifies uncertainty, provides SHAP-based explanations, and simulates risk-adjusted profits using real commodity price data.
🚀 Features

    Real-time Crop Prediction: Multilayer Perceptron (MLP) served via FastAPI for low-latency inference

    Uncertainty Quantification: Monte Carlo Dropout at inference to estimate predictive entropy

    Explainability: SHAP (SHapley Additive exPlanations) for per-feature contribution insights

    Probability Calibration: Temperature scaling to align model confidences with real-world accuracies

    Profit Simulation: Monte Carlo sampling of historical modal prices for expected profit ± risk

    Responsive UI: Glassmorphic design with scroll animations; dynamic district loading

🔧 Tech Stack

Backend:

    Python 3.9+, FastAPI, Uvicorn

    PyTorch, scikit-learn, SHAP

Frontend:

    HTML5, CSS3 (Glassmorphism), Bootstrap 5, AOS animations, jQuery

Data:

    Soil parameters CSV (N, P, K, pH, rainfall, temperature, humidity)

    Government commodity prices CSV

⚙️ Getting Started
Prerequisites

    Python 3.8+

    pip or conda

    Node.js (optional, for front-end tooling)

Installation

    Clone the repo

text
git clone https://github.com/shvm2/A-Dynamic-Crop-Prediction-and-Profit-Estimation-Web-Application-with-Real-Time-User-Input-Adaptation.git
cd crop-prediction

Create & activate a virtual environment

text
python3 -m venv venv
source venv/bin/activate    # macOS/Linux
venv\Scripts\activate       # Windows

Install Python dependencies

    text
    pip install -r requirements.txt

🚀 Running the App

    Start the backend

text
uvicorn main:app --reload

FastAPI will run at http://127.0.0.1:8000

Open the frontend

    Simply open index.html in your browser, or serve via any static server:

        text
        npx serve .

    Interact!

        Enter soil N, P, K, pH, select State, District, Month → Click Predict

        View crop recommendation, confidence, SHAP explanations, & profit estimates

📘 Usage Examples
API Endpoints

    POST /predict

        Payload (JSON):

json
{
  "nitrogen": 90.0,
  "phosphorous": 42.0,
  "potassium": 43.0,
  "ph": 6.5,
  "state": "KARNATAKA",
  "district": "BANGALORE RURAL",
  "month": "JAN"
}

Response (JSON):

        json
        {
          "crop": "apple",
          "probabilities": { "apple": 0.85, "banana": 0.10, ... },
          "uncertainty": 0.12,
          "shap_values": { "N": 0.15, "pH": 0.05, ... },
          "profit_estimates": [
            { "crop": "apple", "expected_profit": 5320, "risk": 450 },
            ...
          ]
        }

📊 Notebooks & Visualization

    profit_visualization.ipynb

        Generates tables and plots of expected profit ± risk across all crops

    SHAP Summary

        Use the notebook to produce bar charts of mean |SHAP| per feature for publication figures

🤝 Contributing

    Fork this repository

    Create your feature branch (git checkout -b feature/YourFeature)

    Commit your changes (git commit -m 'Add feature')

    Push to the branch (git push origin feature/YourFeature)

    Open a Pull Request

Please fill out issues and PRs with clear descriptions, and adhere to the existing code style.
📝 License

This project is licensed under the MIT License. See LICENSE for details.
✉️ Contact

    Maintainer: Shivam

    Email: shivamsingh271104@gmail.com.com

    GitHub: shvm2

Feel free to raise issues or feature requests-happy farming! 🌱
