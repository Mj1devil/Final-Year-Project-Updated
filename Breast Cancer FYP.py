import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load the dataset
data = pd.read_csv(r"""E:\FYP's\Breast CAncer\breast-cancer.csv""")



# Drop unnecessary columns and define features (X) and target (y)
data_cleaned = data.drop(columns=['Unnamed: 0'])
X = data_cleaned.drop(columns=['Class'])
y = data_cleaned['Class']

# Encode categorical features and the target variable
le = LabelEncoder()
for col in X.columns:
    X[col] = le.fit_transform(X[col])
y = le.fit_transform(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize the models
log_reg = LogisticRegression(max_iter=1000)
decision_tree = DecisionTreeClassifier(random_state=42)
random_forest = RandomForestClassifier(random_state=42)

# Train each model
log_reg.fit(X_train, y_train)
decision_tree.fit(X_train, y_train)
random_forest.fit(X_train, y_train)

# Make predictions
y_pred_log_reg = log_reg.predict(X_test)
y_pred_decision_tree = decision_tree.predict(X_test)
y_pred_random_forest = random_forest.predict(X_test)

# Evaluate the models
def evaluate_model(y_true, y_pred):
    return {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred),
        'Recall': recall_score(y_true, y_pred),
        'F1 Score': f1_score(y_true, y_pred)
    }

# Collect the evaluations for each model
eval_log_reg = evaluate_model(y_test, y_pred_log_reg)
eval_decision_tree = evaluate_model(y_test, y_pred_decision_tree)
eval_random_forest = evaluate_model(y_test, y_pred_random_forest)

# Display the evaluation results
results_df = pd.DataFrame({
    'Logistic Regression': eval_log_reg,
    'Decision Tree': eval_decision_tree,
    'Random Forest': eval_random_forest
})

print("Hybrid Model Evaluation Results:")
print(results_df)
