# 📘 Lecture 2: Probability for Machine Learning (EEE Students)

---

## 🔹 1. Motivation – Why Probability in ML?

* Machine Learning works in **uncertain environments**.
* Predictions are never 100% sure → probability quantifies uncertainty.
* Used in ML for:

  * Classification (spam detection, fault detection)
  * Prediction (forecasting load demand, weather impact on grids)
  * Bayesian models (updating belief with new data)

💡 **EEE Analogy:**

* In power systems, relay operation is not deterministic → probability of trip depends on fault severity & protection settings.

---

## 🔹 2. Random Variables

* **Random Variable (RV):** A quantity whose outcome is uncertain.

Types:

* **Discrete RV** → Finite outcomes (Switch ON/OFF, Fault/No Fault).
* **Continuous RV** → Infinite outcomes (Voltage, Current, Temperature).

📊 Example:

* Discrete: Probability a motor fails today = 0.1
* Continuous: Voltage fluctuation in distribution line follows a probability distribution.

---

## 🔹 3. Probability Distributions

### 3.1 Normal (Gaussian) Distribution

* Most natural/engineering data follows normal curve.
* Defined by **mean (μ)** and **variance (σ²)**.

📊 EEE Example:

* Measurement errors in sensors follow normal distribution.

ML Use:

* Many algorithms assume input data is normally distributed.

---

### 3.2 Bernoulli Distribution

* Only 2 outcomes (success/failure, yes/no).

📊 EEE Example:

* Circuit breaker → {Trip=1, No Trip=0}.

ML Use:

* Logistic Regression for binary classification.

---

### 3.3 Other Useful Distributions

* **Binomial:** Multiple Bernoulli trials (EEE: relay fails after 5 operations).
* **Poisson:** Number of events in time (EEE: transformer faults per year).

---

## 🔹 4. Conditional Probability

* Probability of event A happening given that event B has occurred.

Formula:

$$
P(A|B) = \frac{P(A \cap B)}{P(B)}
$$

📊 EEE Example:

* Probability(motor failure | high temperature observed).

ML Use:

* Classification models rely heavily on conditional probability.

---

## 🔹 5. Bayes’ Theorem

$$
P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)}
$$

* Updates probability of a hypothesis when new evidence is observed.

📊 Example: Spam Email Detection

* $A$: email is spam
* $B$: email contains word “offer”
* Bayes updates belief: P(Spam | Offer).

EEE Example:

* Probability(fault in cable | relay tripped).

ML Use:

* **Naïve Bayes Classifier** – simple but powerful for text classification, fault prediction.

---

## 🔹 6. Expectation & Variance

* **Expectation (mean):** Weighted average of possible outcomes.
* **Variance:** How much values spread out.

📊 Example:

* Average load demand of a feeder = 500 MW (expectation).
* Fluctuation = variance (day-to-day variability).

ML Use:

* Helps in defining loss functions, regularization, and model uncertainty.

---

## 🔹 7. Case Study – Fault Detection with Probability

**Problem:** Detect whether a motor is faulty using temperature sensor data.

Steps:

1. Define hypothesis: Faulty (F), Healthy (H).
2. Collect sensor data (temperature distribution).
3. Use Bayes’ theorem:

   $$
   P(F|Temperature) = \frac{P(Temperature|F)P(F)}{P(Temperature)}
   $$
4. Decision: If probability > threshold → classify as Faulty.

💡 This is the probabilistic logic behind many ML-based fault detection systems.

---

## 🔹 8. Key Takeaways

* **Random variables** represent uncertain data.
* **Distributions** (Normal, Bernoulli, etc.) describe data behavior.
* **Conditional Probability & Bayes’ Theorem** are foundations for ML classifiers.
* Probability connects raw sensor data → predictions with uncertainty.

💡 Without probability, ML cannot **reason about uncertainty** in predictions.

---
