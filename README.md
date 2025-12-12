# 📘 Gurgaon Real Estate ML Model 
Predicting house prices in Gurgaon using a machine-learning model trained on the California Housing dataset.

---

## 📌 Project Overview
This project builds a **machine learning pipeline** to predict house prices for Gurgaon city.  
Since Gurgaon housing data is limited, the model is trained on the **California Housing dataset** (a standard ML dataset) and later used to generate predictions for custom inputs.

The project includes:
- End-to-end ML **training pipeline**
- **Preprocessing** using `ColumnTransformer`  
- **Random Forest Regressor** model  
- **Model serialization** using `.joblib` / `.pkl`  
- **Prediction script** for new input files  
- Clean & scalable folder structure  

---
## 🚀 Features
- Complete ML training workflow  
- Auto preprocessing (numerical + categorical)  
- Missing value handling  
- Feature scaling + one-hot encoding  
- Random Forest regression model  
- Model saving using Joblib  
- Clean prediction pipeline for new inputs  

---

## 🛠 Technologies Used
- **Python 3.10+**
- **Pandas**
- **NumPy**
- **Scikit-Learn**
- **Joblib**
- **OpenPyXL**

---

## 📥 Setup and Installation
This guide explains how to set up the project on your local machine, install all dependencies, and run the training/prediction scripts.
                
* ### 1. Clone the Repository                                    
First, download the project from GitHub:
```bash
    git clone https://github.com/your-username/Gurgaon-Real-Estate-ML-Model.git
    cd Gurgaon-Real-Estate-ML-Model
```
(Replace your-username and your-repo-name with your actual GitHub details.)

* ### 2. Install Python(Required Version: 3.9+)
Check your Python version:
```bash
 python --version
```
If Python is not installed, download it from:
https://www.python.org/downloads/
  
* ### 3. Create a Virtual Environment (Recommended)
Creating a virtual environment keeps your project dependencies isolated.                
```bash
python -m venv .venv
```
Activate the virtual environment:
* **On Windows:**
```bash
.\.venv\Scripts\activate
```
* **On macOS/Linux:**
```bash
source .venv/bin/activate
 ```
Once activated, you should see `(venv)` before your terminal prompt.  

* ### 4. Install Required Dependencies
All project dependencies are listed in `requirements.txt.`
```bash
pip install -r requirements.txt
```

* ### 5. Add Dataset Files (If Not Included)
Place your dataset files (like housing.xlsx, input.xlsx) into the Datasets/ folder

* ### 6. Train the Model
:- Load dataset  
:- Build preprocessing pipeline   
:- Train the ML model    
:- Save the model to model.pkl   

* ### 7. Run Predictions
:- Load your trained model    
:- Preprocess your input data    
:- Save predictions in `Dataset/predictions.xlsx`

