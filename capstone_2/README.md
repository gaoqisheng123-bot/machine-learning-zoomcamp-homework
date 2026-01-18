# Traffic Sign Recognition AI Project

This project implements a complete end-to-end Machine Learning pipeline to classify traffic signs from the German Traffic Sign Recognition Benchmark (GTSRB).

## Problem Description
Traffic sign recognition is a fundamental component and difficulty in autonomous vehicle systems.Therefore, a high accuracy system that can recognize the sign accurately and in a short time is required to ensure passenger safety. The challenge with this dataset is the high variability in real-world images:
*   **Varying Lighting:** Images range from very dark to overexposed.
*   **Motion Blur:** Captured from moving vehicles, causing low resolution.
*   **Class Imbalance:** Some signs appear significantly more frequently than others in the training data.

**Goal:** This capstone project aims to create a CNN model that handles these variabilities and deploy it as a production-ready web service.

---
## 1. Dataset
The dataset used is the **German Traffic Sign Recognition Benchmark (GTSRB)** obtained from Kaggle.

*   **Source:** [Kaggle - GTSRB - German Traffic Sign Recognition Benchmark]([(https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign)])
*   **Size:** ~50,000 images across 43 categories.
*   **Dataset Acquisition:** The data is pulled programmatically via `kagglehub`in training.py, manual zip file management is not needed.

### Label Enhancement (Feature Engineering)
The raw dataset provides only numeric identifiers. As part of the **Data Preparation** phase:
1.  A **Numeric-to-Text Mapping** is implemented for all 43 classes.
2.  A `ClassName` column is added to the dataframes to enable human-readable visualization during EDA and testing stage.
3.  The final deployed API will return both the `ClassId` and the `ClassName`.

---

## 2. Exploratory Data Analysis
Extensive EDA is performed in the `notebook.ipynb` to guide the modeling process:
*   **Target Variable Analysis:** The distribution of the 43 classes is visualized using Seaborn. This reveals a significant imbalance which is addressed using **Stratified Splitting** and **Balanced Class Weights**.
*   **Feature Mapping:** The numerical `ClassId` (0-42) is converted to descriptive `ClassName` (e.g., "Stop", "Yield") for better human interpretability.
*   **Image Content Analysis:** The pixel intensity distribution is analyzed and random samples are chosen to be visualized, identifying that many images suffer from poor contrast.
*   **Augmentation Strategy:** Based on EDA, data augmentation is applied (rotation, brightness shifts, and zooms) to make the model invariant to lighting and camera angles.

---

## 3. Model Training & Tuning
This project trains CNN under different parameter and the best model is chosen:
*   **Baseline Model:** A simple CNN without dropout to establish a performance floor.
*   **Variations:** This project experiments with **Inner Layer Size**, **Dropout Rate** and **Learning Rate** to find the optimal depth.
*   **Parameter Tuning:**
    *   **Inner Layer Size:** Tested 64, 128, and 256 nodes.
    *   **Dropout Rate:** Tested 0.2, 0.5, and 0.8 to find the best regularization balance.
    *   **Learning Rate:** Tuned the Adam optimizer using 0.01, 0.001, and 0.0001.
*   **Inheritance:** Each tuning phase inherited the "Best" parameters from the previous step to continuously evolve the model.
*   During parameter stage, all models are trained with 5 epochs, the best parameters obtained are then trained with 20 epochs to be used as the final model. The default "Best" parameter are 128 inner layers, dropout rate with 0.5 and learning rate of 0.001.

| Phase | Parameter Tuned | Values Tested | Best Value | Val Accuracy | Rationale & Observation |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **0** | **Baseline** | N/A | No Dropout | 94.13% | Establishing a floor. The model showes high accuracy even for base model. |
| **1** | **Inner Layer Size** | 64, 128, **256** | 256 | 89.53% | 256 nodes captures complex spatial features of 43 classes better than other number of nodes. |
| **2** | **Dropout Rate** | **0.2**, 0.5, 0.8 | 0.5 | 94.44% | 0.8 caused underfitting; 0.2 provided the best regularization against overfitting. |
| **3** | **Learning Rate** | 0.01, **0.001**, 0.0001 | 0.001 | 94.44% | 0.01 is too aggressive, 0.0001 is too slow 0.001 provided smooth convergence. |
| **Final**| **Grand Champion**| Combined Best | **Final Model** | **99.03%** | Final architecture: 2 Conv layers + 256 Dense + 0.2 Dropout + 0.001 LR (20 Epochs). |

---

## 4. Deployment and Run Locally
The model is served as a web service via **Flask**.
*   **Endpoint:** `/predict` (POST)
*   **Response:** JSON format including `class_id`, `class_name`, and `confidence`.

Run
*   **`train.py`**: This script contains the final architecture and logic to train the model and save it as `traffic_sign_model.h5`.
Run
*   **`predict.py`**: This script starts the Flask API locally and interact with the model via a web interface.

When predict.py is running, open a new terminal and 

Run
*   **`predict_test.py`**: This script run the selected testing picture in the testing dataset, edit the py script to change the tested picture.

or

Open on a browser and search to choose an own picture to test
```bash
http://localhost:9696/
```

---

## 5. Containerization
The application is fully containerized.
*   **Build command:**
   ```bash
   `docker build -t traffic-sign-ai .`
   ```
   ```bash
*   **Run command:** `docker run -it -p 9696:9696 traffic-sign-ai`
   ```



### Cloud Deployment
*   **Public URL:** [INSERT YOUR RENDER URL HERE]
*   **Note:** Render's Free Tier has a 512MB RAM limit. TensorFlow initialization may occasionally exceed this limit.
*   **Deployment Evidence:** A screenshot of a successful prediction response (JSON) from the deployed container is provided below.

*(INSERT_YOUR_SCREENSHOT_HERE)*

---

## 7. Project Structure
*   `notebook.ipynb`: Data cleaning, EDA, tuning experiments, and evaluation.
*   `train.py`: Script for training the final model.
*   `predict.py`: Flask web service for deployment.
*   `predict_test.py`: CLI tool for users to test local images.
*   `Dockerfile`: Multi-stage build for containerization.
*   `pyproject.toml` & `uv.lock`: Dependency management.
*   `traffic_sign_model.h5`: The serialized trained model.
