
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import graphviz

def train_and_evaluate_decision_tree(data_path="/home/ubuntu/synthetic_root_cause_data.csv"):
    """Trains and evaluates a Decision Tree model for root cause prediction."""
    df = pd.read_csv(data_path)

    # Drop rows where failure_type or root_cause is None (normal operations)
    df_failures = df[df["failure_indicator"] == 1].copy()

    if df_failures.empty:
        print("No failure events found in the data to train root cause prediction.")
        return

    # Features for root cause prediction
    # Exclude 'timestamp', 'failure_indicator', 'failure_type', 'root_cause'
    features = [
        "cpu_load",
        "memory_usage",
        "disk_io",
        "network_latency",
        "service_response_time"
    ]
    X = df_failures[features]
    y = df_failures["root_cause"]

    # Encode target variable (root_cause) to numerical labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    root_cause_classes = le.classes_

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded)

    # Initialize and train the Decision Tree Classifier
    dt_classifier = DecisionTreeClassifier(random_state=42)
    dt_classifier.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = dt_classifier.predict(X_test)

    # Evaluate the model
    print("\n--- Decision Tree Model Evaluation ---")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=root_cause_classes))

    # Visualize the Decision Tree (optional, for smaller trees)
    dot_data = export_graphviz(dt_classifier, out_file=None,
                                feature_names=features,
                                class_names=root_cause_classes,
                                filled=True, rounded=True,
                                special_characters=True)
    graph = graphviz.Source(dot_data)
    graph.render("/home/ubuntu/root_cause_decision_tree", format="png", view=False)
    print("\nDecision Tree visualization saved to /home/ubuntu/root_cause_decision_tree.png")

    return dt_classifier, le

if __name__ == "__main__":
    model, label_encoder = train_and_evaluate_decision_tree()
    if model:
        print("\nRoot Cause Prediction Model training and evaluation complete.")
        # Example prediction for a new data point
        # new_data = pd.DataFrame([[0.9, 0.8, 0.7, 0.1, 0.2]], columns=["cpu_load", "memory_usage", "disk_io", "network_latency", "service_response_time"])
        # predicted_root_cause_encoded = model.predict(new_data)
        # predicted_root_cause = label_encoder.inverse_transform(predicted_root_cause_encoded)
        # print(f"Predicted root cause for new data: {predicted_root_cause[0]}")



