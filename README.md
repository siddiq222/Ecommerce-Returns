# 🛒 Ecommerce Returns Prediction

A machine learning model to predict whether an ecommerce order 
will be returned, using XGBoost classifier.

## 📁 Project Structure
ecommerce-returns/
│
├── run.py                  # Main script to run the model
├── xgb_model.pkl           # Trained XGBoost model
├── sample_from_return.csv  # Sample dataset
└── README.md               # Project documentation

## 📌 Problem Statement
Predict whether a customer will return a purchased product
based on order/customer features.

## 🔧 Tech Stack
- Python 3.11
- XGBoost
- Pandas
- Scikit-learn

## ⚙️ Installation
git clone https://github.com/siddiq222/Ecommerce-Returns.git
cd Ecommerce-Returns
pip install -r requirements.txt

## 🚀 Usage
python run.py

## 📊 Dataset
- File: `sample_from_return.csv`
- Contains order-level data with return labels

## 📈 Model
- Algorithm: XGBoost Classifier
- Saved as: `xgb_model.pkl`

## 📬 Contact
SHAIK HUSSAIN SIDDIK — skhussainsiddik235@email.com