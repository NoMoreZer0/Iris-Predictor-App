# 🌸 Iris Species Prediction App

A **Streamlit**-based interactive machine learning application for predicting the species of iris flowers using multiple classifiers. This app supports single predictions, batch predictions via CSV, visualizations, and SHAP-based interpretability.

test ci/cd

---

## 🚀 Features

* **Model Selection**: Random Forest, SVM, Logistic Regression
* **Single Prediction**: Enter feature values using sliders
* **Batch Prediction**: Upload CSV and get predictions with download option
* **Data Visualizations**: Pairplots, scatterplots
* **Model Evaluation**: Accuracy, confusion matrix, classification report
* **SHAP Explanation**: Visual force plot for Random Forest predictions
* **Custom Training**: Upload your own training dataset (with the same structure)

---

## 📂 Project Structure

```bash
Streamlit_ml_app/
├── classification.py         # Main Streamlit app
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
```

---

## ⚙️ Setup Instructions

1. **Clone the repository**

```bash
git clone https://github.com/yourusername/Streamlit_ml_app.git
cd Streamlit_ml_app
```

2. **Create and activate a virtual environment (optional but recommended)**

```bash
python -m venv .venv
# Activate it
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Run the app**

```bash
streamlit run classification.py
```

---

## 📌 How to Use

1. **Choose a model** from the sidebar.
2. **Set hyperparameters** and provide input for single prediction.
3. **Upload a CSV file** for batch prediction (same column format as iris dataset).
4. **View model metrics** and SHAP explanations if using Random Forest.
5. **(Optional)** Upload your own dataset for training under "Custom Training Data".

---

## 🧲 Example CSV Format

```csv
sepal length (cm),sepal width (cm),petal length (cm),petal width (cm)
5.1,3.5,1.4,0.2
6.2,3.4,5.4,2.3
...
```

---

## 🖼️ Screenshots

> <img width="1919" height="844" alt="image" src="https://github.com/user-attachments/assets/051dc198-8fe8-4495-9c30-2db10488c4bc" />


---

## 📄 License

MIT License
© 2025 Sameer017
