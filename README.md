# Crop-Recommendation

A **Machine Learning-powered Crop Recommendation System** that predicts the most suitable crop for cultivation based on soil nutrients and environmental conditions.

This repository includes:

* A Python application (`app.py`) to run the crop recommendation interface
* A Jupyter notebook (`ML(Crop_Prediction).ipynb`) for training and evaluating the crop prediction model
* Supporting files such as dataset and model artifacts

---

## Table of Contents

1. **Project Summary**
2. **Features**
3. **Dataset**
4. **Technologies Used**
5. **Installation**
6. **Usage**
7. **Model Training (Notebook)**
8. **How It Works**
9. **Customization**
10. **Troubleshooting**
11. **License**

---

## Project Summary

This project builds a predictive model that takes agricultural parameters — such as soil nutrient levels and weather conditions — and recommends the most suitable crop to grow. Using historical data and machine learning techniques, the system assists farmers and agricultural stakeholders in making informed decisions to maximize yield and resource efficiency ([GitHub][1]).

Typical input features include:

* Soil nutrients: Nitrogen (N), Phosphorus (P), Potassium (K)
* Climate parameters: Temperature, Humidity
* Soil pH
* Rainfall

The target output is a recommended crop label (e.g., rice, maize, etc.) based on the input conditions ([GitHub][2]).

---

## Features

* Predicts optimal crop based on soil and environmental parameters
* Trained using machine learning models such as Decision Trees, Random Forests, SVM, etc. (example workflows from similar repos) ([GitHub][1])
* Interactive Python application (`app.py`) for real-time predictions
* Notebook to explore training, evaluation, and model insights
* Scalable for future datasets and deployment

---

## Dataset

A typical crop recommendation dataset (e.g., `Crop_recommendation.csv`) includes records with the following columns:

| Feature     | Description                 |
| ----------- | --------------------------- |
| N           | Nitrogen content in soil    |
| P           | Phosphorous content in soil |
| K           | Potassium content in soil   |
| temperature | Ambient temperature in °C   |
| humidity    | Relative humidity (%)       |
| pH          | Soil pH value               |
| rainfall    | Rainfall in mm              |
| label       | Recommended crop            |

This dataset is used to train the model to learn correlations between input conditions and suitable crops ([GitHub][2]).

---

## Technologies Used

* **Python 3.x**: Core programming language
* **scikit-learn**: Machine learning modeling library ([Wikipedia][3])
* **Pandas & NumPy**: Data loading and processing
* **Jupyter Notebook**: Model exploration and training
* **(Optional) Flask/Streamlit**: Interactive interface for predictions

---

## Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/Kashika221/Crop-Recommendation.git
   cd Crop-Recommendation
   ```

2. **Create a virtual environment** (recommended)

   ```bash
   python -m venv venv
   source venv/bin/activate     # macOS/Linux
   venv\Scripts\activate        # Windows
   ```

3. **Install dependencies**

   Create a `requirements.txt` including:

   ```
   numpy
   pandas
   scikit-learn
   flask        # if using a Flask app
   streamlit    # if using Streamlit
   ```

   Then install:

   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

### 1. Run the App

If `app.py` is an interactive prediction interface (Flask/Streamlit), start it:

**Flask:**

```bash
python app.py
```

**Streamlit:**

```bash
streamlit run app.py
```

This should open a UI (web or local) where you can input environmental parameters and get crop recommendations.

### 2. Train/Evaluate Model in Notebook

Open the Jupyter notebook:

```bash
jupyter notebook "crop recomendation/ML(Crop_Prediction).ipynb"
```

Follow the cells to load the dataset, preprocess features, train the model, and evaluate performance.

---

## How It Works

1. **Data Loading:** Import dataset with features like soil nutrients, temperature, humidity, pH, and rainfall.
2. **Preprocessing:** Handle missing values, scale or encode features as needed.
3. **Model Training:** Train a classification algorithm to learn relationships between inputs and crop types.
4. **Prediction:** Input user values into the trained model to generate crop recommendations.
5. **Deployment:** Run `app.py` to serve predictions via a simple interface.

This approach uses a typical supervised classification pipeline common in crop recommendation systems ([GitHub][1]).

---

## Customization

You can extend this project by:

* Adding weather API integration to fetch real-time environmental data
* Deploying the model via a web service (Flask/Streamlit)
* Evaluating additional ML models (e.g., Gradient Boosting, Random Forest)
* Saving/loading trained models using `pickle` or `joblib`

---

## Troubleshooting

* **Incorrect predictions:** Ensure the model is trained on clean, balanced data.
* **Missing dependencies:** Reinstall via `pip install -r requirements.txt`.
* **Notebook issues:** Verify correct relative paths to data files in the notebook.

---

## License

Add a license file (e.g., MIT, GPL-3.0) to clarify usage and distribution permissions.

---
