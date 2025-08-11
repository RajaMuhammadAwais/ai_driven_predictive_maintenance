# AI-Driven Root Cause Prediction Model Documentation

## Introduction

This document provides a comprehensive overview of the AI-Driven Root Cause Prediction module, a key component of the broader AI-Driven Predictive Maintenance and Dynamic Resource Optimization system. The primary goal of this module is to automatically identify the underlying causes of system failures, enabling proactive healing and preventing future occurrences. This documentation covers the research, design, implementation, and evaluation of the root cause prediction model.

## Research and Model Selection

To develop an effective root cause prediction model, we conducted research into several relevant machine learning techniques. The following models were considered:

### 1. Causal Inference Models

Causal inference models are designed to distinguish true root causes from mere symptoms by modeling cause-and-effect relationships. They combine domain knowledge with observational data to provide actionable insights. While powerful, these models can be complex to implement and often require significant domain expertise to construct accurate causal graphs.

### 2. Decision Trees

Decision trees are a simple yet powerful supervised learning algorithm that can be used for both classification and regression tasks. For root cause analysis, they are particularly useful because they create a hierarchical structure of rules that can be easily interpreted. They are well-suited for problems with ample historical data and can be visualized to understand the decision-making process.

### 3. Bayesian Networks

Bayesian Networks are probabilistic graphical models that represent the conditional dependencies between a set of variables. They are excellent for handling uncertainty and can integrate both expert knowledge and data. Bayesian Networks can perform diagnostic inference, which is ideal for root cause analysis, as they can calculate the probability of a root cause given observed symptoms.

**Model Selection**: For this initial implementation, we have chosen to use a **Decision Tree Classifier**. This choice is motivated by the following factors:
-   **Interpretability**: Decision trees are highly interpretable, which is crucial for understanding why a particular root cause is predicted.
-   **Ease of Implementation**: They are relatively easy to implement and train.
-   **Visualization**: The ability to visualize the decision tree provides a clear understanding of the learned rules.

While Causal Inference Models and Bayesian Networks offer more advanced capabilities, they also introduce greater complexity. The Decision Tree model serves as a strong baseline and can be extended or replaced with more sophisticated models in future iterations.

## Synthetic Data Generation

To train and evaluate the root cause prediction model, we generated a synthetic dataset that simulates real-world system telemetry and failure events. The `generate_root_cause_data.py` script creates a CSV file (`synthetic_root_cause_data.csv`) with the following key features:
-   **System Metrics**: `cpu_load`, `memory_usage`, `disk_io`, `network_latency`, `service_response_time`.
-   **Failure Information**: `failure_indicator`, `failure_type`.
-   **Root Cause**: The ground truth `root_cause` for each failure event.

The dataset includes various failure types (e.g., server crash, disk failure) and their corresponding root causes (e.g., hardware failure, software bug, resource exhaustion). This allows us to train a supervised learning model to predict the root cause based on the system metrics at the time of failure.

## Model Implementation

The root cause prediction model is implemented in the `root_cause_prediction.py` script. The key steps in the implementation are as follows:

1.  **Data Loading and Preprocessing**:
    -   The script loads the synthetic root cause data from the CSV file.
    -   It filters the data to include only failure events (`failure_indicator == 1`).
    -   The features (system metrics) and the target variable (`root_cause`) are separated.
    -   The categorical `root_cause` labels are encoded into numerical values using `LabelEncoder`.

2.  **Model Training**:
    -   The data is split into training and testing sets.
    -   A `DecisionTreeClassifier` is initialized and trained on the training data.

3.  **Model Evaluation**:
    -   The trained model is used to make predictions on the test set.
    -   The performance of the model is evaluated using an accuracy score and a classification report, which includes precision, recall, and F1-score for each root cause class.

4.  **Model Visualization**:
    -   The trained decision tree is visualized using `graphviz` and saved as a PNG image (`root_cause_decision_tree.png`). This visualization helps in understanding the rules that the model has learned to predict root causes.

## Evaluation Results

The Decision Tree model was evaluated on the test set, and the results are as follows:

**Accuracy**: 0.60

**Classification Report**:
```
                     precision    recall  f1-score   support

configuration_error       1.00      1.00      1.00         7
   hardware_failure       0.45      0.69      0.55        13
      network_issue       0.62      0.62      0.62        13
resource_exhaustion       0.60      0.48      0.53        25
       software_bug       0.60      0.53      0.56        17

           accuracy                           0.60        75
          macro avg       0.65      0.66      0.65        75
       weighted avg       0.61      0.60      0.60        75
```

**Analysis of Results**:
-   **Overall Accuracy**: The model achieves an overall accuracy of 60%, which indicates that it can correctly predict the root cause for 60% of the failure events in the test set.
-   **Class-wise Performance**:
    -   The model performs perfectly for `configuration_error`, with a precision, recall, and F1-score of 1.00. This is likely because the symptoms associated with this root cause are very distinct in the synthetic data.
    -   The performance for other root causes is mixed. For example, `hardware_failure` has a relatively high recall (0.69) but lower precision (0.45), suggesting that the model is good at identifying hardware failures but also misclassifies other failures as hardware failures.
    -   `resource_exhaustion` has a lower recall (0.48), indicating that the model struggles to identify this root cause correctly.
-   **Macro and Weighted Averages**: The macro and weighted averages for precision, recall, and F1-score are around 0.60-0.65, which suggests that the model has a reasonable, but not exceptional, performance across all classes.

**Conclusion on Evaluation**: The Decision Tree model provides a decent baseline for root cause prediction. The interpretability of the model, as seen in the visualized tree, is a significant advantage. However, there is room for improvement, especially in distinguishing between root causes with similar symptoms.

## Future Improvements

To enhance the root cause prediction module, the following steps can be taken:

-   **Advanced Models**: Implement and evaluate more sophisticated models like Bayesian Networks or Causal Inference Models, which can capture more complex relationships and handle uncertainty more effectively.
-   **Feature Engineering**: Create additional features that might provide more information to the model, such as interaction terms between metrics or time-based features.
-   **Hyperparameter Tuning**: Optimize the hyperparameters of the Decision Tree model (e.g., `max_depth`, `min_samples_leaf`) to improve its performance.
-   **Ensemble Methods**: Use ensemble methods like Random Forest or Gradient Boosting, which combine multiple decision trees to create a more robust and accurate model.
-   **Integration with Predictive Failure Detection**: Integrate the root cause prediction module with the predictive failure detection model. When a failure is predicted, the root cause prediction model can be used to identify the likely cause, enabling proactive intervention.
-   **Real-world Data**: Train and evaluate the model on real-world data to ensure its effectiveness in a production environment.

This documentation provides a solid foundation for the AI-Driven Root Cause Prediction module. The initial implementation demonstrates the feasibility of using a Decision Tree model for this task, and the outlined future improvements provide a clear path for further development.


