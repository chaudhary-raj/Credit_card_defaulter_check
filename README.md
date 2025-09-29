# 💳 Credit Card Default Prediction Dashboard  

This project is an interactive web application built with **Streamlit** that predicts the probability of a client defaulting on their credit card payment. The prediction is powered by a **K-Nearest Neighbors (KNN)** machine learning model trained on the **Default of Credit Card Clients** dataset from Taiwan.  

---

## ✨ Features  

- 🎛️ **Interactive UI** – Input client data via a clean and user-friendly interface  
- 🤖 **ML Integration** – Real-time predictions using a pre-trained Scikit-learn KNN model  
- 📊 **Probability Score** – Get not just a prediction, but also the probability of default  
- 🖼️ **Data Visualization** – Risk displayed with color-coded indicators & progress bar  
- 🔁 **Reproducibility** – Includes Jupyter Notebook with full training process  

---

## 🖥️ Demo  

📷 *(Add a screenshot of your running application here!)*  

---

## 🛠️ Tech Stack  

- **Language**: Python 3  
- **Web Framework**: Streamlit  
- **Machine Learning**: Scikit-learn  
- **Data Manipulation**: Pandas, NumPy  
- **Model Serialization**: Joblib  
- **Environment**: Jupyter Notebook / Google Colab  

---

## 🚀 Getting Started  

### Prerequisites  
- Python 3.8 or higher  
- `pip` package manager  

### Installation  

Clone the repository:  
```bash
git clone https://github.com/your-username/your-repository-name.git
cd your-repository-name
```
Create and activate a virtual environment:
```bash
python -m venv venv
.\venv\Scripts\activate
```
Install dependencies
``` bash
pip install -r requirements.txt
```
Run the Streamlit app
```bash
streamlit run app.py
```
## 📦 credit-card-default-dashboard
 - ┣ 📜 app.py                       -- Streamlit web application
 - ┣ 📜 knn_model.joblib             --Pre-trained KNN model
 - ┣ 📜 KNN_model_for_credit_card.ipynb  -- Jupyter Notebook (EDA + training)
 - ┣ 📜 requirements.txt             -- Python dependencies
 - ┣ 📜 Data.csv                     -- Credit card dataset
 - ┗ 📜 README.md                    -- Project documentation




