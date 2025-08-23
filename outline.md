## **Session 1: Introduction to Machine Learning** ⚡

### **Lecture Contents:**

* **What is Machine Learning?**

  * Formal definition: *“ML is a field of study that gives computers the ability to learn from data without being explicitly programmed.”* (Arthur Samuel, 1959)
  * Modern definition: *“Algorithms that improve their performance automatically through experience.”*

* **Types of ML**

  * **Supervised Learning** – trained with labeled data (input-output pairs)
  * **Unsupervised Learning** – trained with unlabeled data (hidden patterns)
  * **Reinforcement Learning** – trial-and-error learning using rewards/penalties

* **Applications in Electrical & Computer Science**

  * **Fault detection** in power systems (classification)
  * **Load forecasting** (regression, neural networks)
  * **Renewable integration** (time series, predictive models)
  * **Image recognition** (computer vision)

* **Traditional Programming vs ML**

  * Traditional: *Rules + Data → Output*
  * ML: *Data + Output → Rules (model)*

* **Environment Setup**

  * Install Python (Anaconda distribution recommended)
  * Jupyter Notebook basics (cells, markdown, running code)
  * Intro to Google Colab (alternative cloud-based option)

### **Hands-on / Activity:**

* Install Anaconda & open Jupyter Notebook
* Run first code cell:

  ```python
  print("Hello Machine Learning")
  ```
* Explore markdown cell for note-taking

### **Takeaway:**

* ML is data-driven learning that can automate predictions & classifications.
* Jupyter Notebook is a key environment for ML experiments.

### **Quiz Questions ❓**

1. What is the key difference between traditional programming and ML?
2. Name one real-life EEE application of supervised learning.
3. Which ML type works without labeled data?

### **Solutions ✅**

1. Traditional uses explicit rules, ML learns rules from data.
2. Example: Fault classification in power system.
3. Unsupervised Learning.

---

## **Session 2: Python Basics** 💻

### **Lecture Contents:**

* **Data Types**: int, float, string, bool

* **Variables & Operators**

  * Arithmetic, logical, relational

* **Control Flow**

  * if-else, nested conditions
  * loops (for, while)

* **Functions**

  * Defining functions (`def`)
  * Return values, parameters

* **Data Structures**

  * Lists (indexing, slicing, iteration)
  * Tuples (immutable collections)
  * Dictionaries (key-value storage, useful for storing parameters)

* **Practical Electrical Examples**

  * Ohm’s law calculation: `V = I * R`
  * Store student results in dictionary
  * Calculate total energy cost using loop over days

### **Hands-on / Activity:**

* Write function `power(P, t)` that computes `Energy = P * t`.
* Dictionary: store values of a circuit `{“Voltage”: 220, “Current”: 5}`.
* Write loop to print square of numbers from 1–10.

### **Takeaway:**

* Python basics are foundation for ML coding.
* Lists and dictionaries will be heavily used for data handling.

### **Quiz Questions ❓**

1. What is the difference between list and tuple?
2. Write code to calculate resistance when `V=10, I=2`.
3. How do you declare a dictionary in Python?

### **Solutions ✅**

1. List is mutable, tuple is immutable.
2. `R = 10/2` → 5 Ω.
3. Example: `d = {"Voltage": 220, "Current": 5}`.

---

## **Session 3: Python Libraries for ML** 📊

### **Lecture Contents:**

* **NumPy**

  * Arrays, indexing, slicing
  * Matrix operations (dot product, transpose)
  * Use case: representing electrical signals as arrays

* **Pandas**

  * DataFrames: rows & columns
  * Reading CSV/Excel files
  * Descriptive stats: mean, median, std
  * Filtering data (e.g., select rows where voltage > 220)

* **Matplotlib**

  * Line plots (voltage vs time)
  * Bar charts (energy consumption per day)
  * Histogram (distribution of load demand)

* **Seaborn**

  * Correlation heatmaps (feature relation)
  * Boxplots (outlier detection in current data)

* **Use Case Example:**

  * Load electricity consumption dataset
  * Use Pandas to clean
  * Plot daily demand curve

### **Hands-on / Activity:**

* Import CSV dataset into Pandas
* Compute `df.describe()` for summary stats
* Plot load curve with Matplotlib

### **Takeaway:**

* These four libraries form the foundation of Python ML.

### **Quiz Questions ❓**

1. Which library is best for handling tabular data?
2. How do you plot a line graph in Matplotlib?
3. What does `df.describe()` give?

### **Solutions ✅**

1. Pandas.
2. `plt.plot(x, y)`
3. Statistical summary (count, mean, std, min, max, quartiles).

---

## **Session 4: Data Handling & Preprocessing** 🛠

### **Lecture Contents:**

* **Why Preprocessing is Important**

  * “Garbage in → Garbage out” in ML
  * Data quality directly impacts model accuracy

* **Common Issues in Data**

  * Missing values (NaN)
  * Duplicates
  * Inconsistent units (kWh vs Wh)

* **Handling Missing Data**

  * Drop missing rows
  * Fill with mean/median/mode
  * Forward fill / backward fill

* **Normalization & Standardization**

  * Normalization: scale between \[0,1]
  * Standardization: mean = 0, std = 1
  * Example: normalize load data for NN training

* **Encoding Categorical Data**

  * Label encoding (0/1 for Yes/No)
  * One-hot encoding for categories (e.g., device type: fan, light, AC)

* **Splitting Data**

  * Train-test split (70-30 or 80-20 rule)

### **Hands-on / Activity:**

* Use Pandas to fill missing values in dataset
* Normalize voltage values to \[0,1]
* One-hot encode device type column

### **Takeaway:**

* Preprocessing ensures datasets are clean and ML-ready.

### **Quiz Questions ❓**

1. What is the difference between normalization and standardization?
2. Why is handling missing data important?
3. What is one-hot encoding?

### **Solutions ✅**

1. Normalization scales to \[0,1], standardization makes mean=0, std=1.
2. Missing data causes bias and errors in models.
3. Representing categorical data as binary columns.

---

## **Session 5: Supervised Learning – Regression** 📈

### **Lecture Contents:**

* **Regression Concept**

  * Predict continuous numeric variables
  * Example: predict voltage, temperature, demand

* **Linear Regression**

  * Equation: `y = mx + c`
  * Parameters: slope (m), intercept (c)
  * Loss function: Mean Squared Error (MSE)

* **Applications in EEE**

  * Forecast load demand from historical data
  * Model sensor calibration (voltage vs current)

* **Training Process**

  * Feed data (X = input, Y = output)
  * Fit model (estimate m, c)
  * Predict new values

* **Error Metrics**

  * MSE (Mean Squared Error)
  * RMSE (Root Mean Squared Error)

### **Hands-on / Activity:**

* Use Scikit-learn to implement Linear Regression
* Fit model on dataset: time vs load demand
* Visualize regression line and prediction

### **Takeaway:**

* Regression predicts numeric outputs and is useful for forecasting.

### **Quiz Questions ❓**

1. What does MSE measure?
2. Write regression equation for predicting load (Y) from time (X).
3. Which library is commonly used for ML regression in Python?

### **Solutions ✅**

1. Average squared difference between predicted and actual values.
2. `Y = mX + c`.
3. Scikit-learn.


---

## **Session 6: Supervised Learning – Classification** 🔍

### **Lecture Contents:**

* **What is Classification?**

  * Predicting discrete categories (labels)
  * Examples: fault vs no-fault, spam vs not-spam

* **Logistic Regression**

  * Sigmoid function: outputs probability between 0–1
  * Decision boundary: threshold (e.g., >0.5 = class 1)
  * EEE application: classifying circuit states (healthy/faulty)

* **Decision Trees**

  * Splitting data based on conditions (Yes/No branches)
  * Metrics: Gini Index, Entropy, Information Gain
  * Pros: interpretable, easy to visualize

* **Model Workflow**

  * Load dataset (e.g., circuit fault dataset)
  * Train logistic regression & decision tree models
  * Compare accuracy

### **Hands-on / Activity:**

* Train logistic regression model on dataset of **fault detection** (inputs: voltage/current, output: fault status).
* Visualize decision tree structure using Scikit-learn.

### **Takeaway:**

* Classification predicts categories, useful in EEE for **fault diagnosis**.

### **Quiz Questions ❓**

1. What is the main difference between regression and classification?
2. Which function is used in logistic regression for probability mapping?
3. Why are decision trees popular in ML?

### **Solutions ✅**

1. Regression → numeric values, Classification → categories.
2. Sigmoid function.
3. Easy to interpret and visualize.

---

## **Session 7: Unsupervised Learning** 🔗

### **Lecture Contents:**

* **What is Unsupervised Learning?**

  * Works with **unlabeled data**
  * Goal: find hidden patterns/clusters

* **Clustering with K-Means**

  * Steps: choose K → assign data points → update centroids → repeat
  * Example: group households by energy consumption patterns

* **Dimensionality Reduction with PCA**

  * Reduce number of features while keeping maximum variance
  * Useful for visualization & reducing computational cost

* **Applications in EEE**

  * Group customers for demand-side management
  * Detect unusual energy consumption (anomaly detection)

### **Hands-on / Activity:**

* Apply **K-Means** on electricity consumption dataset → cluster into “low”, “medium”, “high” users.
* Apply **PCA** on high-dimensional sensor dataset, plot reduced 2D features.

### **Takeaway:**

* Unsupervised learning reveals **hidden structures** in unlabeled data.

### **Quiz Questions ❓**

1. Does K-Means need labeled data?
2. What is the role of PCA in ML?
3. Give one EEE application of clustering.

### **Solutions ✅**

1. No, it works on unlabeled data.
2. Reduces dimensionality while preserving variance.
3. Grouping households based on consumption pattern.

---

## **Session 8: Advanced Regression** ⚙

### **Lecture Contents:**

* **Polynomial Regression**

  * Extension of linear regression to non-linear data
  * Equation: `y = a0 + a1x + a2x² + … + anx^n`
  * Example: nonlinear relationship between **temperature and resistance**

* **Support Vector Regression (SVR)**

  * Uses support vectors to fit within a margin
  * Handles outliers better than linear regression
  * Application: predicting noisy load demand

* **Comparison with Linear Regression**

  * Linear = straight line fit
  * Polynomial = curved fit
  * SVR = flexible margin-based fit

### **Hands-on / Activity:**

* Use polynomial regression to fit **temperature vs resistance** curve (thermistor data).
* Train SVR on noisy voltage dataset and compare results with linear regression.

### **Takeaway:**

* Advanced regression handles **non-linear relationships** and noisy datasets.

### **Quiz Questions ❓**

1. Which regression method can model curves?
2. What does SVR use to fit data?
3. Why not always use polynomial regression?

### **Solutions ✅**

1. Polynomial regression.
2. Support vectors with margin.
3. Overfitting risk for high-degree polynomials.

---

## **Session 9: Neural Networks** 🧠

### **Lecture Contents:**

* **Biological Inspiration**

  * Human brain → neurons, synapses
  * Artificial neuron as mathematical model

* **Artificial Neuron Model**

  * Inputs (x1, x2, …, xn)
  * Weights (w1, w2, …, wn)
  * Weighted sum + bias
  * Activation function

* **Activation Functions**

  * Sigmoid (0–1 range, good for probabilities)
  * ReLU (f(x) = max(0,x), popular in deep learning)
  * Tanh (outputs between -1 and 1)

* **Feedforward Neural Network**

  * Layers: input, hidden, output
  * Forward propagation & error calculation

* **Application in EEE**

  * Load forecasting (predicting tomorrow’s power demand)

### **Hands-on / Activity:**

* Build a simple NN using **Keras/TensorFlow** with 1 hidden layer.
* Train on **load forecasting dataset** (inputs: past 7 days, output: next day).

### **Takeaway:**

* Neural networks learn complex relationships and are widely applied in energy forecasting.

### **Quiz Questions ❓**

1. What does an activation function do?
2. Which activation is most common in deep networks?
3. Give one application of NN in EEE.

### **Solutions ✅**

1. Introduces non-linearity into the model.
2. ReLU.
3. Load forecasting.

---

## **Session 10: Model Evaluation** ✅

### **Lecture Contents:**

* **Why Evaluate Models?**

  * High accuracy on training ≠ good model
  * Need to check generalization

* **Train-Test Split**

  * Common split: 70–30, 80–20
  * Avoid testing on training data

* **Cross-Validation (k-fold)**

  * Split dataset into k parts
  * Train on (k-1) parts, test on 1 part
  * Rotate until every part is tested

* **Overfitting vs Underfitting**

  * Overfitting: memorizes training data, fails on new data
  * Underfitting: too simple, fails on both train & test

* **Regularization**

  * L1 (Lasso) → feature selection
  * L2 (Ridge) → penalizes large weights

* **Evaluation Metrics**

  * For regression: RMSE, R² score
  * For classification: accuracy, precision, recall, F1-score

### **Hands-on / Activity:**

* Split dataset into train/test sets, evaluate model accuracy.
* Demonstrate overfitting using high-degree polynomial regression.
* Perform 5-fold cross-validation using Scikit-learn.

### **Takeaway:**

* Model evaluation ensures **reliability and generalization** of ML models.

### **Quiz Questions ❓**

1. Why is cross-validation better than a single train-test split?
2. What is overfitting?
3. Name one regularization method.

### **Solutions ✅**

1. It evaluates on multiple splits, reducing bias.
2. Model performs well on training but poorly on new data.
3. L1 (Lasso) or L2 (Ridge).



---

## **Session 11: Time Series Analysis ⏳**

**Topics Covered:**

* Introduction to time series data

  * Definition and examples (stock prices, power demand, solar irradiation)
  * Components: trend, seasonality, noise
* Techniques for analysis

  * Moving averages (simple & weighted)
  * Exponential smoothing
  * Autocorrelation & partial autocorrelation
* Forecasting approaches

  * ARIMA model basics
  * ML models for time series (LSTM mention only as advanced)

**Hands-on / Activity:**

* Load a time series dataset (e.g., household power consumption)
* Plot and analyze trend & seasonality
* Apply moving average for smoothing
* Forecast next few points using simple ARIMA

**Learning Outcome:**
Students will be able to analyze trends and forecast values in time-dependent data (e.g., electricity demand).

---

## **Session 12: Ensemble Methods 🌳**

**Topics Covered:**

* Limitations of single models
* Concept of ensembles (wisdom of crowds)
* Bagging vs. Boosting

  * Bagging → Random Forest
  * Boosting → Gradient Boosting, XGBoost
* Trade-offs: interpretability vs accuracy
* Real-life applications (fault detection, equipment failure prediction)

**Hands-on / Activity:**

* Apply Random Forest on EEE dataset (fault classification)
* Apply Gradient Boosting for power load classification
* Compare accuracy with logistic regression / decision trees

**Learning Outcome:**
Students will understand how combining models improves robustness and accuracy.

---

## **Session 13: Feature Engineering & Dimensionality Reduction 🔍**

**Topics Covered:**

* Why feature engineering matters
* Techniques for feature engineering

  * Encoding categorical variables
  * Creating polynomial features
  * Domain-driven feature creation (EEE examples: power factor, harmonics)
* Feature selection methods

  * Filter (correlation)
  * Wrapper (recursive feature elimination)
  * Embedded (LASSO regression)
* Dimensionality reduction

  * PCA theory & intuition
  * Application to high-dimensional datasets

**Hands-on / Activity:**

* Perform feature extraction on dataset (create new engineered features)
* Apply PCA and visualize variance explained
* Train model before and after dimensionality reduction, compare performance

**Learning Outcome:**
Students will be able to prepare efficient models for high-dimensional data.

---

## **Session 14: Project Implementation 🛠️**

**Topics Covered:**

* Workflow of a full ML project

  * Define problem (EEE dataset selection)
  * Data preparation & cleaning
  * Model selection & training
  * Evaluation & tuning
* Best practices for team collaboration
* Project management (GitHub/Colab for collaboration)

**Hands-on / Activity:**

* Students form groups (2–3 members)
* Select dataset (EEE related: load forecasting, fault detection, energy consumption)
* Start implementing end-to-end ML pipeline

**Learning Outcome:**
Students will experience real-world problem-solving using ML.

---

## **Session 15: Project Presentation & Wrap-up 🎤**

**Topics Covered:**

* Project presentation structure

  * Problem statement
  * Approach & methodology
  * Results & evaluation metrics
  * Challenges faced and improvements
* Discussion of projects
* Feedback and Q/A session
* Future directions in ML for EEE

**Hands-on / Activity:**

* Each group presents their project (10–15 mins each)
* Peer and instructor evaluation
* Open discussion on improvements

**Learning Outcome:**
Students consolidate their learning, gain confidence in presenting ML solutions, and understand future applications in their field.

---




