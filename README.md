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

1. **Interactive Predictor (Real-Time Inference)**: Drag the sliders for Sepal/Petal lengths and widths to immediately see classification probabilities update.
2. **Dynamic SVG Flower Visualizer**: An organic flower SVG drawing expands, contracts, and changes color in real-time to match your slider dimensions and prediction outputs:
   * **Iris Setosa** $\rightarrow$ Purple Glow 🟣
   * **Iris Versicolor** $\rightarrow$ Cyan/Teal Glow 🩵
   * **Iris Virginica** $\rightarrow$ Magenta/Pink Glow 🩷
3. **In-Browser Model Training (ML Sandbox)**: Adjust split ratios, learning rates, epochs, or neighbor counts ($K$) and click **Train Model**. The app splits the 150 records, trains the model, and displays updated accuracy scores, precision metrics, and a dynamic **Confusion Matrix**—all in milliseconds!
4. **Data Cluster Explorer**: An interactive 2D scatter plot (using Chart.js) mapping the entire dataset. A blinking rectangular target moves in real-time to show where your slider coordinates lie relative to the species clusters.
5. **Code Showcase**: Tabbing interface comparing Scikit-Learn Python training blocks with raw client-side JavaScript classifier algorithms and FastAPI routing APIs.

---

## 💻 How to Run Locally

### 1. The Interactive Dashboard (No setup required)
Because all logic runs on the client-side:
1. Double-click the **`index.html`** file in your local workspace.
2. It opens instantly in any browser. No web servers or installations are required!

### 2. The Python FastAPI Server (Optional)
To test the backend server capability:
1. Make sure you have python dependencies installed:
   ```bash
   pip install fastapi uvicorn scikit-learn pandas joblib
   ```
2. Start the API server:
   ```bash
   python app.py
   ```
3. Open your browser to `http://127.0.0.1:8000/docs` to interact with the API Swagger documentation, or test predictions using curl:
   ```bash
   curl -X POST "http://127.0.0.1:8000/predict" -H "Content-Type: application/json" -d "{\"sepal_length\": 5.1, \"sepal_width\": 3.5, \"petal_length\": 1.4, \"petal_width\": 0.2}"
   ```

---

## 🌐 How to Deploy to GitHub Pages (100% Free)

Since the core app runs entirely client-side, you can host it instantly on GitHub Pages:
1. Push this directory to a GitHub repository (e.g., `iris-classification-sandbox`).
2. Go to the repository on GitHub and click on **Settings** (gear icon).
3. Scroll down the left sidebar and click on **Pages**.
4. Under **Build and deployment**, select **Deploy from a branch**.
5. Choose your branch (typically `main` or `master`) and folder (typically `/ (root)`), then click **Save**.
6. In 1–2 minutes, your project is live at:
   `https://<your-username>.github.io/<your-repository-name>/`
7. Copy the link and add it to your GitHub repository description and resume to wow recruiters!
