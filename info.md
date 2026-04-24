# Next-Location Prediction Notebook Overview

This document explains the purpose, role, and algorithmic choices behind every major section and cell within the `Next_Location_Prediction.ipynb` notebook. The goal of the notebook is to build an intelligent system that predicts where a student will go next (Stage, Canteen, or Lab) during a college event day based on context.

---

## 1. Import Libraries
**What is it doing?** We import standard data science packages: `numpy` (math and arrays), `pandas` (tabular data manipulation), `matplotlib` & `seaborn` (visualization), and `sklearn` & `xgboost` (Machine Learning models and tools).
**Why use it?** These libraries provide the foundational tools necessary to handle data, extract patterns, train models, and validate results.

## 2. Synthetic Data Generation
**What is it doing?** Since real-world location tracking data isn't provided, this code simulates realistic student movements. We define "transition probabilities" which encode assumptions (e.g., after an event at the Stage, a student has a 60% chance of going to the Canteen). The script scales this to 10,000 distinct interactions throughout an event day, saving the output to `student_location_data.csv`.
**Why use it?** Machine learning needs data. The transition probabilities let us simulate distinct movement behaviors which models will ultimately try to learn.

## 3. Exploratory Data Analysis (EDA)
This section visually investigates the generated data to confirm the data is logical and identifies patterns before modeling.
*   **3.1 Location Distribution:** Plots bar charts checking how many students visit the Stage, Canteen, and Lab. Ensures there isn't extreme class imbalance.
*   **3.2 Transition Matrix (Heatmap):** Maps the probability of moving from one location to another. 
    *   **Why use it?** It gives an explicit visual of conditional probabilities — verifying our synthetic rules worked.
*   **3.3 Time-Based Movement Trends:** Visualizes how students move around depending on whether it's Morning, Afternoon, or Evening.
*   **3.4 Crowd Density vs Next Location:** Shows how crowd levels impact where a student goes (e.g. maybe students avoid the Canteen if it's "High" density).

## 4. Preprocessing & Feature Engineering
**What is it doing?** Machine learning algorithms (like Random Forest or Logistic Regression) usually require mathematical inputs.
*   **Label Encoding:** We convert text classes (like "morning", "afternoon") into numbers (0, 1, 2) using `LabelEncoder`. This is applied to locations, event types, crowd density, and time of day.
*   **Feature Engineering (Cyclical Encoding):** We extract the specific "hour" of the event from the timestamp (e.g., 8:00 AM, 14:00 PM). Hours are cyclical (23:00 is close to 01:00). We encode them using math (`sin` and `cos` waves).
*   **Train-Test Split:** We divide the data into an 80% training chunk (for learning) and a 20% testing chunk (for evaluation).

## 5. Model Building
**What is it doing?** Here we train typical contextual Machine Learning Classification algorithms. We feed them `(X_train, y_train)`:
*   **Logistic Regression:** Evaluates linear relationships. It's a quick, interpretable baseline model.
*   **Decision Tree:** Creates a flowchart of if-else rules (e.g., "if crowd is high AND location is lab, go to stage"). Easily interprets non-linear logic.
*   **Random Forest:** An *ensemble* model. It trains 200 different Decision Trees on random subsets of data and averages their predictions to prevent overfitting and improve accuracy.
*   **XGBoost:** Another powerful *ensemble* model that continuously trains trees to fix the errors made by previous trees (Gradient Boosting).

## 6. Evaluation
**What is it doing?** We pass the unseen test data (`X_test`) into our models to see how well they learned.
*   **Classification Reports & Confusion Matrix:** These break down the exact performance: When the student actually went to the Canteen, how many times did the model *predict* they would go to the Canteen? 
*   **Feature Importance:** Using the Random Forest model, we plot a bar chart showing which features heavily influenced the algorithm. (E.g., `current_location` will be highly influential). 

## 7. Sequential Model – Markov Chain
**What is it doing?** A first-order Markov Chain is built specifically for dealing with sequential "states" (locations). It creates a mathematical matrix indicating the exact frequencies of one location transitioning to another directly from the training data.
**Why use it?** ML Models look at *global context* (crowd, event, time). A Markov Chain looks *strictly at the sequence* ("If you are here, what's mathematically the most likely next step"). It is important to implement both and compare.

## 8. Transition Probability Visualization
**What is it doing?** Creates an interactive-style network diagram using nodes (circles representing locations) and directed arrows (representing the probability mathematically derived by the Markov Chain). Visualizes the "flow" of the college event.

## 9. ML vs Sequential Model Comparison
**What is it doing?** Graphs the overarching Test-Accuracy of all 5 algorithms side-by-side. 
**Why use it?** It proves that while Markov Chains are great for sequences, adding context (Random Forests reading the crowd and time) creates a smarter prediction tool in real-world scenarios.

## 10. Final Prediction Function & Model Export
*   **Final Prediction Function (`predict_next_location`):** Wraps all the preprocessing (label encoding text into numbers, doing the sin/cos math on time) and runs a single user input query against the trained model to spit back out a human-readable string (e.g., "canteen").
*   **Export (`joblib`):** Dumps the best numerical model, alongside its necessary text-decoders, to disk. This is the bridge allowing a separate backend server (like FastAPI) to use the Jupyter Notebook's brain. 

## 11. Conclusion
Summarizes the steps taken, validates algorithmic choices, highlights the importance of the contextual features the Random Forest uses over the pure sequence mapping of the Markov Chain, and concludes the experiment.
