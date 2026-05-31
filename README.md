# 🌸 Iris Flower Classification & ML Sandbox Studio

Welcome to the **Iris Flower Classification & Machine Learning Sandbox Studio**! This project is a comprehensive portfolio piece demonstrating data analysis, python-based training pipelines, and a **highly interactive, real-time Machine Learning dashboard** running directly in the browser. 

Developed by **Alimaaz Akhter** as part of the **CodeAlpha Data Science Internship**.

---

## 📁 Repository Structure

* 📓 **[iris_classification.ipynb](iris_classification.ipynb)**: The research Jupyter notebook containing data exploration (EDA), duplicate cleaning, data visualizations (pairplot, correlation heatmap), target encoding, and model fitting in Python.
* 📊 **[Iris.csv](Iris.csv)**: The classic Iris dataset containing 150 samples with sepal/petal dimensions across three flower species.
* 🌐 **[index.html](index.html)**: The structural markup for the modern glassmorphic web dashboard.
* 🎨 **[styles.css](styles.css)**: Custom styles including design system variables, glassmorphism filters, responsive grids, and clean visual transitions.
* 🧠 **[script.js](script.js)**: The core JavaScript logic housing the embedded dataset, **custom-written ML classifiers (Logistic Regression, KNN, and Decision Trees)** running client-side, the dynamic SVG flower renderer, and the Chart.js visual chart controller.
* ⚡ **[app.py](app.py)**: A Python FastAPI API server file that trains/loads a Scikit-Learn model and serves predictions via JSON endpoints.
* ⚙️ **[train_export.py](train_export.py)**: Python training script to compile and serialize the Scikit-Learn model to `.pkl` binaries.
* 📝 **[Q&A](Q&A)**: Interview preparation document answering common questions about this project's methodology.

---

## 🚀 Key Web Dashboard Features

1. **Interactive Predictor (Real-Time Inference)**.
2. **Dynamic SVG Flower Visualizer**: An organic flower SVG drawing expands, contracts, and changes color in real-time to match your slider dimensions and prediction outputs:
   * **Iris Setosa** $\rightarrow$ Purple Glow 🟣
   * **Iris Versicolor** $\rightarrow$ Cyan/Teal Glow 🩵
   * **Iris Virginica** $\rightarrow$ Magenta/Pink Glow 🩷
3. **In-Browser Model Training (ML Sandbox)**.
4. **Data Cluster Explorer**: An interactive 2D scatter plot (using Chart.js) mapping the entire dataset. A blinking rectangular target moves in real-time to show where your slider coordinates lie relative to the species clusters.
5. **Code Showcase**: Tabbing interface comparing Scikit-Learn Python training blocks with raw client-side JavaScript classifier algorithms and FastAPI routing APIs.

---
