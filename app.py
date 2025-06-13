from flask import Flask, render_template, request, session
import numpy as np
from sklearn.discriminant_analysis import StandardScaler
import torch
import joblib
import os
import shap
from datetime import datetime
import matplotlib.pyplot as plt
import sqlite3


app = Flask(__name__)
app.secret_key = 'supersecretkey123'


# Load PyTorch model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TabTransformerGRU(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embeddings = torch.nn.ModuleList([
            torch.nn.Embedding(2, 10),   # sex
            torch.nn.Embedding(4, 10),   # cp
            torch.nn.Embedding(2, 10),   # fbs
            torch.nn.Embedding(3, 10),   # restecg
            torch.nn.Embedding(2, 10),   # exang
            torch.nn.Embedding(3, 10),   # slope
            torch.nn.Embedding(5, 10),   # ca
            torch.nn.Embedding(4, 10)    # thal
        ])
        self.gru = torch.nn.GRU(8*10 + 5, 64, batch_first=True)
        self.fc = torch.nn.Linear(64, 1)

    def forward(self, X_cat, X_num):
        embedded = [embed(X_cat[:, i]) for i, embed in enumerate(self.embeddings)]
        embedded = torch.cat(embedded, dim=-1)
        X_combined = torch.cat([embedded, X_num], dim=-1).unsqueeze(1)
        _, hidden = self.gru(X_combined)
        return self.fc(hidden.squeeze(0)).squeeze(1)

model = TabTransformerGRU().to(device)
model.load_state_dict(torch.load('Heart-Disease-Prediction-main/mymodel.pkl', map_location=device))
model.eval()

# Scaler configuration
scaler = StandardScaler()
scaler.mean_ = np.array([54.3, 131.6, 246.5, 149.6, 1.05])
scaler.scale_ = np.array([9.0, 17.6, 51.8, 22.9, 1.2])
scaler.n_samples_seen_ = 300

def model_wrapper(X):
    """Wrapper for SHAP explanation"""
    X_cat = torch.tensor(X[:, :8], dtype=torch.long).to(device)
    X_num = torch.tensor(X[:, 8:], dtype=torch.float32).to(device)
    with torch.no_grad():
        return model(X_cat, X_num).cpu().numpy()

@app.route("/")
def hello():
    return render_template("index.html")

@app.route("/detail", methods=["POST"])
def submit():
    session['name'] = request.form["Username"]
    return render_template("detail.html", n=session['name'])

@app.route('/predict', methods=["POST"])
def predict():
    inputs = {
        'age': float(request.form['age']),
        'sex': int(request.form['sex']),
        'cp': int(request.form['cp']),
        'trestbps': float(request.form['trestbps']),
        'chol': float(request.form['chol']),
        'fbs': int(request.form['fbs']),
        'restecg': int(request.form['restecg']),
        'thalach': float(request.form['thalach']),
        'exang': int(request.form['exang']),
        'oldpeak': float(request.form['oldpeak']),
        'slope': int(request.form['slope']),
        'ca': int(request.form['ca']),
        'thal': int(request.form['thal'])
    }
    session['inputs'] = inputs

    # Prepare data
    cat_data = [inputs[col] for col in ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]]
    num_data = [inputs[col] for col in ["age", "trestbps", "chol", "thalach", "oldpeak"]]
    num_scaled = scaler.transform(np.array(num_data).reshape(1, -1))

    # Prediction
    X_cat = torch.tensor([cat_data], dtype=torch.long).to(device)
    X_num = torch.tensor(num_scaled, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        probability = torch.sigmoid(model(X_cat, X_num)).item()
    
    session['prediction'] = 1 if probability >= 0.5 else 0
    return render_template('predict.html', prediction=session['prediction'])

def save_to_history(data_dict, prediction, report_path, username):
    conn = sqlite3.connect('user_history.db')
    cursor = conn.cursor()

    cursor.execute('''
        INSERT INTO history (
            username, age, sex, cp, trestbps, chol, fbs, restecg, thalach,
            exang, oldpeak, slope, ca, thal, prediction, report_path, timestamp
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        username,
        data_dict['age'],
        data_dict['sex'],
        data_dict['cp'],
        data_dict['trestbps'],
        data_dict['chol'],
        data_dict['fbs'],
        data_dict['restecg'],
        data_dict['thalach'],
        data_dict['exang'],
        data_dict['oldpeak'],
        data_dict['slope'],
        data_dict['ca'],
        data_dict['thal'],
        prediction,
        report_path,
        datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    ))

    conn.commit()
    conn.close()


@app.route("/generate_report", methods=["POST"])
def generate_report():
    import matplotlib.pyplot as plt
    import base64
    from io import BytesIO

    # Get session data
    inputs = session.get('inputs', {})
    prediction = session.get('prediction', 0)
    name = session.get('name', 'User')
    current_date = datetime.now().strftime("%d %b %Y %I:%M %p")

    # SHAP Explanation
    background = np.array([[1, 3, 0, 0, 0, 1, 0, 2] + list(scaler.transform([[54.3, 131.6, 246.5, 149.6, 1.05]])[0])])
    test_instance = np.array([[inputs['sex'], inputs['cp'], inputs['fbs'], inputs['restecg'],
                               inputs['exang'], inputs['slope'], inputs['ca'], inputs['thal'],
                               inputs['age'], inputs['trestbps'], inputs['chol'],
                               inputs['thalach'], inputs['oldpeak']]])

    explainer = shap.Explainer(model_wrapper, background)
    shap_values = explainer(test_instance)

    feature_names = [
        'Sex', 'Chest Pain', 'FBS', 'ECG',
        'Exercise Angina', 'Slope', 'Major Vessels',
        'Thalassemia', 'Age', 'Blood Pressure',
        'Cholesterol', 'Max HR', 'ST Depression'
    ]
    shap_data = list(zip(feature_names, shap_values.values[0]))

    # Generate SHAP Plot
    if not os.path.exists("static"):
        os.makedirs("static")

    plt.figure()
    shap.summary_plot(shap_values.values, test_instance, feature_names=feature_names, show=False)
    plt.tight_layout()
    shap_plot_path = os.path.join("static", "shap_plot.png")
    plt.savefig(shap_plot_path, bbox_inches='tight')
    plt.close()

    # 📌 Save user history
    # You need to define this function separately (already shared above)
    prediction_label = "High Risk" if prediction == 1 else "No Risk"
    report_path = shap_plot_path  # or a real PDF path if you generate one
    save_to_history(inputs, prediction_label, report_path, name)

    return render_template("generate_report.html",
                           name=name,
                           inputs=inputs,
                           prediction=prediction,
                           current_date=current_date,
                           shap_data=shap_data,
                           shap_plot_path=shap_plot_path)

    
    
    
    
    
@app.route("/history")
def history():
    username = session.get("name", "Guest")  # Get logged in user's name

    conn = sqlite3.connect('user_history.db')
    cursor = conn.cursor()

    cursor.execute("SELECT age, sex, cp, trestbps, chol, thalach, exang, prediction, report_path, timestamp FROM history WHERE username=?", (username,))
    rows = cursor.fetchall()
    conn.close()

    return render_template("history.html", rows=rows, username=username)

   
    

if __name__ == "__main__":
    app.run(debug=True)