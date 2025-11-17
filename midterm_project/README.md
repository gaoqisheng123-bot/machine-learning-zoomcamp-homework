# ❤️ Heart Disease Prediction API  
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
**Machine Learning • Flask API • Docker • Google Cloud Run**

# Heart Disease Prediction Service
This repository contains a complete, production-ready machine learning project that predicts the probability of heart disease. 

---

## 1. Project Background: The Problem and The Solution

**The Problem:** Heart disease is a leading cause of mortality worldwide. For clinicians and healthcare providers, the ability to quickly and accurately assess a patient's risk is crucial for early intervention and preventative care. Traditional methods can be time-consuming, and identifying high-risk individuals from complex patient data is a significant challenge.

**The Solution:** This project addresses this challenge by building a machine learning-powered API service. The core of the solution is a predictive model trained on a comprehensive health dataset. This model is then exposed via a web API, which can be used in several ways:
*   **Integration with Health Systems:** A hospital's electronic health record (EHR) system could call this API to provide doctors with an instant risk score for a patient during a consultation.
*   **Research Tool:** Researchers could use this service to analyze risk factors across large populations.
*   **Educational Tool:** It serves as a real-world example of how a machine learning model is taken from a research phase to a live, usable product.

---

## 2. Project Overview

This section shows the complete workflow of the project from experimentation until cloud deployment.

1.  **Phase 1: Experimentation:** This project starts in a Jupyter Notebook to interactively explore the data, handle missing values, and compare different models under different parameters to find the most accurate one.
2.  **Phase 2: Production Training:** This project creates a training script (`train.py`) that loads the raw data, applies the exact same cleaning steps, and trains the winning model to produce the final model artifact.
3.  **Phase 3: Serving & Packaging:** The trained model is loaded by a Flask web server (`predict.py`) that exposes an API. This entire service is then packaged into a portable Docker container.
4.  **Phase 4: Cloud Deployment:** The final Docker container is deployed to Google Cloud Run, a serverless platform, making the prediction service available globally.

---

## 3. The Dataset: Attributes and Information

*   **Source:** [Kaggle Heart Disease Dataset](https://www.kaggle.com/datasets/oktayrdeki/heart-disease/data)
*   **Description:** The dataset contains 10,000 patient records and 21 attributes. Below is a description of each attribute used in the model.

| Attribute | Description | Data Type |
| :--- | :--- | :--- |
| **Age** | The patient's age in years. | Numerical |
| **Gender** | The patient's gender. | Categorical |
| **Blood Pressure** | Systolic blood pressure in mmHg. | Numerical |
| **Cholesterol Level**| Total cholesterol level in mg/dL. | Numerical |
| **Exercise Habits**| Level of physical activity (Low, Medium, High). | Categorical |
| **Smoking** | The patient's smoking status.(Yes or No) | Categorical |
| **Family Heart Disease**| Whether the patient has a family history of heart disease. (Yes or No) | Categorical |
| **Diabetes** | The patient's diabetes status. (Yes or No) | Categorical |
| **BMI** | Body Mass Index. | Numerical |
| **High Blood Pressure**| The patient's high blood pressure status. (Yes or No) | Categorical |
| **Low HDL Cholesterol**| Status of low HDL ("good") cholesterol. (Yes or No) | Categorical |
| **High LDL Cholesterol**| Status of high LDL ("bad") cholesterol. (Yes or No) | Categorical |
| **Alcohol Consumption**| The individual's alcohol consumption level (None, Low, Medium, High) (This column is dropped during EDA) | Categorical |
| **Stress Level** | The patient's self-reported stress level. (Low, Medium, High) | Categorical |
| **Sleep Hours** | Average hours of sleep per night. | Numerical |
| **Sugar Consumption**| Level of sugar intake (Low, Medium, High). | Categorical |
| **Triglyceride Level**| Triglyceride level in mg/dL. | Numerical |
| **Fasting Blood Sugar**| Fasting blood sugar level in mg/dL. | Numerical |
| **CRP Level** | C-Reactive Protein level, an inflammation marker. | Numerical |
| **Homocysteine Level**| Homocysteine level, another cardiovascular risk marker. | Numerical |
| **Heart Disease Status**| **(Target Variable)** Whether the patient has heart disease. (Yes or No) | Categorical |

---

## 4. Project File & Directory Structure

| File / Directory | Purpose |
| :--- | :--- |
| **`notebook.ipynb`** | **The Research Notebook.** All initial data cleaning, visualization, and model comparison happens here. |
| **`train.py`** | **The Model Trainer.** An automated script that trains the final model and saves the `heart_disease_model.pkl` file. |
| **`predict.py`** | **The Prediction API.** A Flask app that loads the trained model and exposes a `/predict` endpoint. |
| **`Dockerfile`** | **The Container Blueprint.** Instructions to package the service into a portable, self-contained Linux environment with all dependencies. |
| **`pyproject.toml`** | **The Dependency List.** Defines the Python packages this project needs (e.g., `pandas`, `xgboost`). |
| **`uv.lock`** | **The Environment Lock File.** Guarantees the reproducible package installation by locking the exact versions of all dependencies. |
| **`deploy.sh`** | **The Cloud Deployment Script.** Automates the build and deployment process to Google Cloud Run. |

---

## 5. Reproducibility

This project tackles the challenge of reproducibility at three distinct levels to ensure maximum reproducibility.

1.  **Environment Reproducibility (`uv.lock`):** The `uv.lock` file captures the exact version of every single Python package used. When someone sets up the project, `uv pip sync` installs these precise versions, eliminating any chance of errors caused by library updates.

2.  **Application Reproducibility (`Dockerfile`):** The `Dockerfile` packages the entire application, which are the Python interpreter, the operating system libraries, and the coding into a single, immutable image. This guarantees that the application runs identically whether on a developer's laptop, a testing server, or in the cloud.

3.  **Workflow Reproducibility (`train.py`, `predict.py`):** By separating experimentation from production, the coding scripts (`train.py`, `predict.py`) create an repeatable workflow. Anyone can run `python train.py` and get the same model artifact, ensuring consistency from training to deployment.

---

## 6. Cloud Deployment with Google Cloud

A local service is great for development, but to make this model truly useful, it needs to be accessible on the internet. This section explains the steps taken to make this project cloud-ready.

### How the Project is made Deployable

Taking a local Flask app to the cloud requires three key steps, which have been implemented in this project:

1.  **Containerization with Docker (`Dockerfile`):**
    A Docker container packages the entire application, which is the Python interpreter, all the installed libraries (`pandas`, `flask`, etc.), and the `predict.py` script into a single, universal package. The `Dockerfile` in this repository is a blueprint that defines exactly how to build this package, ensuring that the application's environment is reproducible and will run identically anywhere.

2.  **Google Cloud Run:**
    **Google Cloud Run**, a modern **serverless** platform is chosen for this project. This is the perfect choice for an API like this project because:
    *   **No Server Management:** Only the Docker container is provided and Google handles everything else.
    *   **Scales to Zero:** If no one is using the API, it automatically scales down to zero instances.
    *   **Automatic Scaling:** If the API suddenly gets thousands of requests, Cloud Run automatically scales up the number of containers to handle the load.

3.  **Automating the Deployment (`deploy.sh`):**
    Manually uploading files and clicking buttons in a web console is slow and prone to errors. The `deploy.sh` script automates the entire deployment process. When the file is run, it performs a professional CI/CD (Continuous Integration/Continuous Deployment) workflow:
    *   It tells **Google Cloud Build** to read the `Dockerfile` and build the container image in the cloud.
    *   The new image is stored in **Google Container Registry**, a secure private storage.
    *   Finally, it instructs **Google Cloud Run** to pull this new image and deploy it as the latest version of the web service.
---

## 7. How to Run This Project on Your Own Machine

This guide will walk you through setting up and running the entire project locally.

### Prerequisites (What you need to install first)
*   **Git:** To download the project code through Git and test the model directly.
*   **Python (version 3.11+):** The programming language this project uses.

### Method 1

### Step A: Get the Project Code
Open your terminal (like Git Bash on Windows) and clone this repository:
```bash
git clone https://github.com/gaoqisheng123-bot/machine-learning-zoomcamp-homework-main.git
cd machine-learning-zoomcamp-homework-main/midterm_project
```
*(You will now be inside the project directory).*


### Step B: Set Up Your Private Python Workspace (Virtual Environment)
This creates an isolated environment for the project.

1.  **Create the environment:**
    ```bash
    python -m venv venv
    ```
2.  **Activate it:**
    *   **On macOS/Linux:** `source venv/bin/activate`
    *   **On Windows (Git Bash):** `source venv/Scripts/activate`
    > You will know it's working when you see `(venv)` at the start of your terminal prompt.

3.  **Install all the necessary packages:**
    ```bash
    pip install uv
    uv pip sync uv.lock
    ```
    You are now all set to run the code!

### Step C: Train the Model
Run the automated training script to produce the `heart_disease_model.pkl` file.
```bash
python train.py
```

### Step D: Run and Test the Local Prediction Service
1.  **Start the Flask server:**
    ```bash
    python predict.py
    ```
    > Your terminal will now be "busy" running the server. **Do not close this terminal.**

2.  **Open a NEW terminal window**, activate the `venv`, and send a test request using `curl`:
    ```bash
    curl -X POST "http://localhost:9696/predict" \
      -H "Content-Type: application/json" -d '{"Age":30,"Gender":"Male","Blood Pressure":160,"Cholesterol Level":250,"Exercise Habits":"High","Smoking":"No","Family Heart Disease":"No","Diabetes":"No","BMI":32.0,"High Blood Pressure":"No","Low HDL Cholesterol":"Yes","High LDL Cholesterol":"Yes","Stress Level":"Low","Sleep Hours":6,"Sugar Consumption":"Low","Triglyceride Level":200,"Fasting Blood Sugar":130,"CRP Level":4.0,"Homocysteine Level":15.0}'
    ```
You will get a JSON prediction back, confirming your local setup works.



###  Method 2: Dowload the zip file directly from GitHub and opened them with VS Code (Optional)
1.   **Install all the necessary packages**

2.   **Run** 
     ```powershell
     python train.py
     ```

3.   **Run**
     ```powershell
     python predict.py
     ```

4.   **Open a NEW terminal and test**
     ```powershell
     Invoke-WebRequest -Uri "http://localhost:9696/predict" -Method POST -Headers @{"Content-Type"="application/json"} -Body '{
            "Age": 30, "Gender": "Male", "Blood Pressure": 160, "Cholesterol Level": 250,
            "Exercise Habits": "High", "Smoking": "No", "Family Heart Disease": "Yes",
            "Diabetes": "No", "BMI": 32.0, "High Blood Pressure": "No",
            "Low HDL Cholesterol": "Yes", "High LDL Cholesterol": "Yes",
            "Stress Level": "Low", "Sleep Hours": 6, "Sugar Consumption": "Low",
            "Triglyceride Level": 200, "Fasting Blood Sugar": 130, "CRP Level": 4.0,
            "Homocysteine Level": 15.0
          }'
     ```



### Method 3 : Try it directly through Google Cloud using Git bash (!!! Try to make the command in one line only in Git bash !!!)
1. **Open Git bash**

2. **Run**
     ```bash
	curl -X POST "https://heart-disease-predictor-867184804970.us-central1.run.app/predict" -H "Content-Type: application/json" -d '{"Age":30,"Gender":"Male","Blood Pressure":160,"Cholesterol Level":250,"Exercise Habits":"High","Smoking":"No","Family Heart Disease":"Yes","Diabetes":"No","BMI":32.0,"High Blood Pressure":"No","Low HDL Cholesterol":"Yes","High LDL Cholesterol":"Yes","Stress Level":"Low","Sleep Hours":6,"Sugar Consumption":"Low","Triglyceride Level":200,"Fasting Blood Sugar":130,"CRP Level":4.0,"Homocysteine Level":15.0}'
     ```
3. **Example Output**
     ```bash
     {
       "has_heart_disease": false,
       "heart_disease_probability": 0.17344666614228477
     }
     ```


## Technologies Used
- Python 3.11  
- Pandas, scikit-learn, XGBoost  
- Flask (API)  
- Docker  
- Google Cloud Run  
- Git + GitHub

## Future Improvements
- Add a frontend dashboard  
- Add monitoring with Cloud Logging  
- Improve model using SHAP explainability  
- Implement CI/CD with GitHub Actions

