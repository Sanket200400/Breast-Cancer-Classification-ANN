# 🧬 Breast Cancer Classification using Artificial Neural Network (ANN)

This project applies machine learning and deep learning techniques to classify whether a breast tumor is **benign** or **malignant** based on clinical features. The model is built using an **Artificial Neural Network (ANN)** implemented with **TensorFlow/Keras**, achieving high prediction accuracy.

---

## 📌 Project Objective

Early diagnosis of breast cancer is crucial for effective treatment.  
This project aims to build an intelligent model capable of accurately predicting tumor type using structured medical data.

---

## 🚀 Features

✔ Data preprocessing and cleaning  
✔ Feature scaling using **StandardScaler**  
✔ Artificial Neural Network built with **TensorFlow/Keras**  
✔ Model evaluation using metrics such as Accuracy, Precision, Recall & F1 Score  
✔ Prediction support for new patient samples  

---

## 🧠 Model Architecture

| Layer | Type   | Units | Activation |
|-------|--------|--------|------------|
| 1     | Dense  | 16     | ReLU       |
| 2     | Dense  | 8      | ReLU       |
| Output| Dense  | 1      | Sigmoid    |

---

## 📊 Results

| Metric | Value |
|--------|-------|
| Accuracy | **XX%** |
| Precision | **XX%** |
| Recall | **XX%** |
| F1-score | **XX%** |

> Replace `XX` values with your actual results from the notebook.

---

## 🛠 Tech Stack

| Category | Tools |
|----------|--------|
| Programming Language | Python |
| Libraries | TensorFlow, Scikit-Learn, Pandas, NumPy, Matplotlib |
| Notebook Environment | Google Colab |

---

## 📁 Project Structure


Breast-Cancer-Classification-ANN
│
├── data/
│ └── data.csv
│
├── notebooks/
│ └── DL_Project_1_Breast_Cancer_Classification_with_NN.ipynb
│
├── src/
│ └── model.py
│ └── preprocess.py
│ └── predict.py (optional)
│
├── requirements.txt
└── README.md

 1️⃣ Clone the Repository
```bash
git clone https://github.com/Sanket200400/Breast-Cancer-Classification-ANN.git
cd Breast-Cancer-Classification-ANN

2️⃣ Install Requirements
pip install -r requirements.txt

3️⃣ Run Notebook

Open and execute:

notebooks/DL_Project_1_Breast_Cancer_Classification_with_NN.ipynb

4️⃣ (Optional) Predict New Data
python src/predict.py

📈 Confusion Matrix Example (Add optional)

🔥 Future Improvements

🔹 Add hyperparameter tuning (GridSearchCV / Optuna)

🔹 Convert project into a Flask or FastAPI web app

🔹 Deploy using Streamlit or HuggingFace Spaces

📄 License

This project is licensed under the MIT License.
