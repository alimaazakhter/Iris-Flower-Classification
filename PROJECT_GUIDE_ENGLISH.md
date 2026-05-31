# 🌸 Iris Species Classification & Interactive Machine Learning Studio - Technical Documentation

This document serves as an exhaustive technical guide for the Iris Species Classification project. It details the operational mechanism, architecture, development methodologies of the interactive front-end features, and the full-stack python serving infrastructure (FastAPI).

---

## 📊 1. Dataset & Feature Engineering

* **Dataset Name**: Fisher's Iris Dataset (1936).
* **Source**: Compiled originally by statistician Ronald Fisher. Available globally via UCI Machine Learning Repository and Kaggle.
* **Dataset Properties**: 150 instances containing physical dimensions of three distinct Iris flower species (balanced distribution: 50 instances per class).
* **Target Classes**:
  * `0`: *Iris Setosa* (Linearly separable cluster)
  * `1`: *Iris Versicolor* (Slight overlap with Virginica)
  * `2`: *Iris Virginica* (Slight overlap with Versicolor)
* **Measurement Features**:
  * **Sepal Length (cm)**: The longitudinal length of the outer supporting leaves.
  * **Sepal Width (cm)**: The latitudinal width of the sepal leaves.
  * **Petal Length (cm)**: The longitudinal length of the inner colored petals.
  * **Petal Width (cm)**: The latitudinal width of the inner petals.

---

## 🌐 2. Front-End Features: Design, Mechanism & Implementation

The interactive client-side web application operates as a standalone serverless sandbox. The operational dynamics of each module are defined below:

### A. Interactive Predictor (Live Inference Interface)
* **Use (Purpose)**: Provides a graphical user interface for real-time dimension inputs, removing the barriers of command-line tools or raw script manipulation for non-technical users.
* **Mechanism**: Range inputs register `input` events in the DOM. A central updater extracts these numeric values, feeds them into the active classifier object, calculates predicted class probabilities, and updates progress bars and text nodes.
* **Development**:
  * **HTML**: Formatted with native `<input type="range">` elements.
  * **CSS**: Range tracks and thumbs are custom-styled utilizing browser pseudo-elements (`::-webkit-slider-thumb` & `::-moz-range-thumb`) to provide an elegant look with transition scaling and green shadows.
  * **JavaScript**: Bound to the DOM via listeners (`addEventListener("input", performLivePrediction)`).

### B. Real-Time SVG Flower Visualizer
* **Use (Purpose)**: Translates static numerical metrics (e.g. 4.3 cm) into a living visual representation of the flower's physical boundaries.
* **Mechanism**: The visualizer scales input values from cm units to pixel dimensions in reference to a central coordinate center `(100, 90)`. Trigonometric functions (`Math.cos` and `Math.sin`) calculate radial vectors at designated angles. Outer sepals are drawn at $30^\circ$, $150^\circ$, and $270^\circ$. Inner petals are drawn at $0^\circ$, $72^\circ$, $144^\circ$, $216^\circ$, and $288^\circ$ degree intervals.
* **Development**:
  * **SVG Architecture**: Designed with `<svg>` elements container, exposing dedicated groups (`<g id="sepals-group">` and `<g id="petals-group">`) for dynamic path injection.
  * **Bezier Paths**: Created dynamically using `document.createElementNS` for namespace compliance. Leaf structures are rendered using quadratic curves: `M cx cy Q c1x c1y tipX tipY Q c2x c2y cx cy Z` (defining start point, control point 1, tip coordinate, control point 2, and return).
  * **CSS Variables**: Seamless class switches dynamically inject colors and matching glow variables (`--species-glow`, `--species-color`) into the SVG stylesheet.

### C. Live Model Training Sandbox
* **Use (Purpose)**: Acts as an educational environment showing how hyperparameter tuning (learning rates, split ratios, neighbors counts, tree depths) directly changes model accuracy and classification boundaries.
* **Mechanism**: Clicking "Train Model" runs a Z-score standard scaling module, splits the embedded 150-record array based on the slider ratio, instantiates the chosen JavaScript model class, runs the training fitment logic, evaluates predictions on the test split, and updates the DOM metrics.
* **Development**:
  * **Logistic Regression in JS**: Features Z-score standard scaling. Implements a multi-class softmax regression training loop that updates weights ($W$) and biases ($b$) using cross-entropy gradients: $W_{new} = W_{old} - \alpha \cdot dW$.
  * **KNN in JS**: Implements Euclidean and Manhattan vector calculations. Distance matrices are sorted in ascending order to extract top $K$ votes and return probabilities.
  * **Decision Tree in JS**: Implements a CART algorithm recursively splitting nodes by evaluating Gini impurities to maximize information gain.
  * **Confusion Matrix**: Rendered using a CSS grid containing conditional colors, scaling opacity values dynamically to reflect count densities.

### D. Visual Cluster Plot (Chart.js)
* **Use (Purpose)**: Displays the dataset cluster distribution and visually positions the user's input parameter coordinate relative to known class clusters.
* **Mechanism**: Chart.js draws dataset distributions. A 4th dataset series ("Your Custom Input") updates a single coordinate value whenever the range sliders are manipulated, enabling live target-seeking indicators.
* **Development**:
  * Loaded securely using Chart.js CDN. Event handlers (`change` listeners on axes) adjust scale titles and re-render datasets by triggering a silent refresh (`scatterChart.update('none')`).

---

## 🐍 3. Python Serving Infrastructure (FastAPI)

To build a professional client-server architecture, the directory is equipped with a Python API stack:

### A. FastAPI Framework
**FastAPI** is a high-performance Python web framework for building APIs.
* **Purpose**: Serves as the middle-tier bridge. It receives HTTP requests from clients, executes Python machine learning calculations (Scikit-Learn, NumPy), and returns serialized responses back to the browser.
* **Key Features**: Auto-generates self-documenting Swagger UI environments at `/docs` using OpenAPI standards.

### B. API Server Module (`app.py`)
This file orchestrates the backend server API:
1. ** Lifespan Handlers & Model Training**: On server startup, the application verifies the existence of `iris_logistic_model.pkl` and `label_encoder.pkl`. If missing, the lifespan thread reads `Iris.csv`, trains a fresh Logistic Regression model, and serializes the assets.
2. **CORS Middlewares**: Enables Cross-Origin Resource Sharing (`CORSMiddleware`) to allow local static files to communicate with API endpoints.
3. **Serving Route `/predict`**:
   * **Input**: Takes JSON body mapping to a Pydantic schema containing sepal/petal dimensions.
   * **Computation**: Passes values into the loaded classifier model, computes probability matrices using `predict_proba`, and maps outputs back to string labels.
   * **Output**: JSON payload outlining species name, ID, and confidence probabilities.
4. **Unified Static Serving**: Mounts the static directory:
   ```python
   app.mount("/", StaticFiles(directory=".", html=True), name="static")
   ```
   This catch-all endpoint serves `index.html`, `styles.css`, and `script.js` directly on `http://127.0.0.1:8000/`, allowing the Python server to act as a unified, dual-purpose host.

### C. Compilation Module (`train_export.py`)
This script acts as the offline machine learning compilation utility:
* **Workflow**: Automates dataset loading, removes duplicate instances, encodes string labels, fits the model, prints validation reports (Accuracy, Precision, Recall, F1), and saves binary artifacts using `joblib`.
* **JS Export Reference**: Extracts and displays raw NumPy arrays of model weights and intercept values in the terminal to assist with hardcoded frontend configurations.
