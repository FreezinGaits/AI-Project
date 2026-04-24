"""Quick model training and export - mirrors the notebook logic."""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import joblib, os

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

np.random.seed(42)
N = 10_000
locations = ['stage', 'canteen', 'lab']
transition_probs = {
    'stage':   [0.15, 0.60, 0.25],
    'canteen': [0.25, 0.30, 0.45],
    'lab':     [0.20, 0.35, 0.45],
}

student_ids = np.random.randint(1, 201, size=N)
current_locs = np.random.choice(locations, size=N)
next_locs = [np.random.choice(locations, p=transition_probs[l]) for l in current_locs]
base = pd.Timestamp('2026-03-15 08:00:00')
timestamps = [base + pd.Timedelta(minutes=int(m)) for m in np.sort(np.random.randint(0, 720, size=N))]

df = pd.DataFrame({
    'student_id': student_ids,
    'timestamp': timestamps,
    'current_location': current_locs,
    'next_location': next_locs,
    'event_type': np.random.choice(['technical', 'cultural', 'sports'], size=N),
    'crowd_density': np.random.choice(['low', 'medium', 'high'], size=N),
    'time_of_day': np.random.choice(['morning', 'afternoon', 'evening'], size=N),
})
df.to_csv('student_location_data.csv', index=False)

# Encode
le_dict = {}
df_enc = df.copy()
for col in ['current_location', 'event_type', 'crowd_density', 'time_of_day']:
    le = LabelEncoder()
    df_enc[col] = le.fit_transform(df_enc[col])
    le_dict[col] = le

le_target = LabelEncoder()
df_enc['next_location'] = le_target.fit_transform(df_enc['next_location'])
df_enc['hour'] = pd.to_datetime(df['timestamp']).dt.hour
df_enc['hour_sin'] = np.sin(2 * np.pi * df_enc['hour'] / 24)
df_enc['hour_cos'] = np.cos(2 * np.pi * df_enc['hour'] / 24)

feature_cols = ['current_location', 'event_type', 'crowd_density', 'time_of_day', 'hour_sin', 'hour_cos']
X = df_enc[feature_cols]
y = df_enc['next_location']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Decision Tree': DecisionTreeClassifier(max_depth=10, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=12, random_state=42),
}
if HAS_XGB:
    models['XGBoost'] = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                                       use_label_encoder=False, eval_metric='mlogloss',
                                       random_state=42, verbosity=0)

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    results[name] = {'model': model, 'accuracy': acc}
    print(f"{name:25s}  Accuracy: {acc:.4f}")

# Markov chain
trans_counts = pd.crosstab(df.loc[X_train.index, 'current_location'], df.loc[X_train.index, 'next_location'])
markov_matrix = trans_counts.div(trans_counts.sum(axis=1), axis=0)
markov_preds_labels = df.loc[X_test.index, 'current_location'].map(lambda l: markov_matrix.loc[l].idxmax())
markov_acc = accuracy_score(y_test, le_target.transform(markov_preds_labels))
results['Markov Chain'] = {'accuracy': markov_acc}
print(f"{'Markov Chain':25s}  Accuracy: {markov_acc:.4f}")

# Export
os.makedirs('model_artifacts', exist_ok=True)
best_name = max({k: v for k, v in results.items() if k != 'Markov Chain'}, key=lambda k: results[k]['accuracy'])
best_model = results[best_name]['model']
joblib.dump(best_model, 'model_artifacts/best_model.pkl')
joblib.dump(le_dict, 'model_artifacts/label_encoders.pkl')
joblib.dump(le_target, 'model_artifacts/target_encoder.pkl')
joblib.dump(feature_cols, 'model_artifacts/feature_cols.pkl')
joblib.dump(markov_matrix, 'model_artifacts/markov_matrix.pkl')
joblib.dump({k: v['accuracy'] for k, v in results.items()}, 'model_artifacts/model_accuracies.pkl')
print(f"\nBest model: {best_name}")
print("Artifacts saved to model_artifacts/")
