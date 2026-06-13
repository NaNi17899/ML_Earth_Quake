# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier, GradientBoostingClassifier, StackingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib



import pandas as pd
from google.colab import drive
# Mount Google Drive
drive.mount('/content/drive')
# Ensure the file path is correct or upload the file to Colab environment.
# Assuming the file is in the root of 'MyDrive' on Google Drive
df = pd.read_csv('/content/drive/MyDrive/earthquake_alert_balanced_dataset.csv')
# Display top rows
print("✅ Dataset Loaded Successfully!")
print(df.head())



from google.colab import drive
drive.mount('/content/drive')




# Dataset summary
print(df.info())
print("\nMissing values:\n", df.isnull().sum())
print("\nClass distribution:\n", df['alert'].value_counts())





# Encode 'alert' labels into numeric values
le = LabelEncoder()
df['alert_encoded'] = le.fit_transform(df['alert'])

# Map encoded labels for reference
label_map = dict(zip(le.classes_, le.transform(le.classes_)))
print("Encoded Labels:", label_map)






from sklearn.feature_selection import SelectKBest, chi2

# Define features (X) and target (y)
X = df.drop(columns=['alert', 'alert_encoded'])
y = df['alert_encoded']

X_chi = X.copy()
X_chi = (X_chi - X_chi.min()) / (X_chi.max() - X_chi.min())  # Normalize (chi2 requires non-negative)

selector = SelectKBest(score_func=chi2, k='all')
chi_scores = selector.fit(X_chi, y)
chi2_results = pd.DataFrame({
    'Feature': X.columns,
    'Chi2_Score': chi_scores.scores_
}).sort_values(by='Chi2_Score', ascending=False)

print("\n📊 Chi-Square Feature Importance:")
print(chi2_results)

# Visualize Chi-Square results
plt.figure(figsize=(8, 4))
sns.barplot(x='Chi2_Score', y='Feature', data=chi2_results, palette='Blues_r', hue='Feature', legend=False)
plt.title("Chi-Square Feature Importance")
plt.show()



# Execute the data splitting cell
# Define features (X) and target (y)
X = df.drop(columns=['alert', 'alert_encoded'])
y = df['alert_encoded']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Training set size:", X_train.shape)
print("Testing set size:", X_test.shape)





# Execute the feature scaling cell
# Standardize numerical features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)





models = {
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "SVM": SVC(kernel='rbf', C=1, gamma='scale', probability=True),
    "Bagging": BaggingClassifier(n_estimators=100, random_state=42),
    "Random Forest (Bagging)": RandomForestClassifier(n_estimators=300, max_depth=10, random_state=42),
    "AdaBoost": AdaBoostClassifier(n_estimators=200, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, random_state=42),
    "XGBoost": XGBClassifier(n_estimators=200, learning_rate=0.1, max_depth=6, eval_metric='mlogloss', random_state=42)
}






estimators = [
    ('knn', KNeighborsClassifier(n_neighbors=5)),
    ('svm', SVC(kernel='rbf', probability=True)),
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42))
]
stacking_model = StackingClassifier(estimators=estimators, final_estimator=GradientBoostingClassifier(random_state=42))
models["Stacking Ensemble"] = stacking_model

# --- Train and evaluate each model ---
results = []

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    results.append((name, acc))

    print(f"=============================")
    print(f"🔹 Model: {name}")
    print(f"✅ Accuracy: {acc*100:.2f}%")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # Confusion Matrix
    plt.figure(figsize=(5, 4))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Greens',
                xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title(f"Confusion Matrix - {name}")
    plt.show()

# --- Accuracy Comparison ---
result_df = pd.DataFrame(results, columns=['Model', 'Accuracy']).sort_values(by='Accuracy', ascending=False)
print("\n🏆 MODEL PERFORMANCE COMPARISON:")
print(result_df)

# Visualize accuracies
plt.figure(figsize=(8, 4))
sns.barplot(x='Accuracy', y='Model', data=result_df, palette='viridis')
plt.title("Model Accuracy Comparison")
plt.xlabel("Accuracy")
plt.ylabel("Model")
plt.show()








# Define features (X) and target (y)
X = df.drop(columns=['alert', 'alert_encoded'])
y = df['alert_encoded']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Training set size:", X_train.shape)
print("Testing set size:", X_test.shape)




# Standardize numerical features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)






from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as snsc
import joblib

# Prepare the data
X = df.drop(columns=['alert'])
y = df['alert']

# Encode the labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train RandomForestClassifier without GridSearchCV (faster tuning)
best_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42
)

best_model.fit(X_train_scaled, y_train)

# Evaluate
y_pred = best_model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

# Display results
print(f"🎯 Model Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=le.classes_))

# Confusion matrix visualization
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Greens',
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.title("Confusion Matrix")
plt.show()

# Save model and scaler to a writable location
joblib.dump(best_model, '/content/earthquake_alert_model.pkl')
joblib.dump(scaler, '/content/earthquake_alert_scaler.pkl')

accuracy






import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Feature importance
importances = pd.Series(best_model.feature_importances_, index=X.columns).sort_values(ascending=False)

plt.figure(figsize=(8,4))
sns.barplot(x=importances, y=importances.index)
plt.title("Feature Importance in Earthquake Alert Prediction")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.show()






import joblib

joblib.dump(best_model, '/content/drive/MyDrive/earthquake_alert_model.pkl')
joblib.dump(scaler, '/content/drive/MyDrive/earthquake_alert_scaler.pkl')

print("✅ Model and Scaler saved successfully to Google Drive!")







import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
df = pd.read_csv('/content/drive/MyDrive/earthquake_alert_balanced_dataset.csv')

# Encode alert column
le = LabelEncoder()
df['alert_encoded'] = le.fit_transform(df['alert'])

# Define X and y (exclude both alert and alert_encoded from features!)
X = df.drop(columns=['alert', 'alert_encoded'])
y = df['alert_encoded']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train RandomForest
model = RandomForestClassifier(n_estimators=300, max_depth=10, random_state=42)
model.fit(X_train_scaled, y_train)

# Evaluate
y_pred = model.predict(X_test_scaled)
print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=le.classes_))

# Save new model and scaler
joblib.dump(model, '/content/drive/MyDrive/earthquake_alert_model_fixed.pkl')
joblib.dump(scaler, '/content/drive/MyDrive/earthquake_alert_scaler_fixed.pkl')
joblib.dump(le, '/content/drive/MyDrive/earthquake_alert_encoder.pkl')

print("🎉 Model retrained and saved successfully!")






!pip install gradio
import gradio as gr
import joblib
import pandas as pd
import numpy as np

# --- Load trained model, scaler, and label encoder ---
# Loading the _fixed versions which were saved after the last successful training
model = joblib.load('/content/drive/MyDrive/earthquake_alert_model_fixed.pkl')
scaler = joblib.load('/content/drive/MyDrive/earthquake_alert_scaler_fixed.pkl')
le = joblib.load('/content/drive/MyDrive/earthquake_alert_encoder.pkl') # Load the saved LabelEncoder

alert_labels = list(le.classes_)

# Debug print
print("✅ Loaded labels:", alert_labels)

# --- Prediction function ---
def predict_alert(magnitude, depth, cdi, mmi, sig):
    # The DataFrame should contain only the features that the model was trained on
    # and in the correct order. 'alert_encoded' was dropped from features X during training.
    data = pd.DataFrame([[magnitude, depth, cdi, mmi, sig]],
                        columns=['magnitude', 'depth', 'cdi', 'mmi', 'sig'])

    scaled_data = scaler.transform(data)
    pred = model.predict(scaled_data)[0]
    label = le.inverse_transform([pred])[0]

    emoji_map = {
        'green': "🟢 Green (Minor)",
        'yellow': "🟡 Yellow (Moderate)",
        'orange': "🟠 Orange (Strong)",
        'red': "🔴 Red (Severe)"
    }
    return f"⚠️ Predicted Alert Level: {emoji_map.get(label, label.capitalize())}"

# --- Gradio interface ---
iface = gr.Interface(
    fn=predict_alert,
    inputs=[
        gr.Number(label="Magnitude"),
        gr.Number(label="Depth (km)"),
        gr.Number(label="CDI"),
        gr.Number(label="MMI"),
        gr.Number(label="Significance (sig)")
    ],
    outputs=gr.Textbox(label="Prediction Result"),
    title="🌍 Earthquake Alert Level Predictor",
    description="Predict the alert level (Green, Yellow, Orange, Red) of an earthquake using a trained ML model."
)

iface.launch(share=True)




