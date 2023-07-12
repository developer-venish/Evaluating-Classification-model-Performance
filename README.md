# Evaluating-Classification-model-Performance
ML Python Project
---------------------------------------------------------------------------------------
![](https://github.com/developer-venish/Evaluating-Classification-model-Performance/blob/main/graph.png) 

![](https://github.com/developer-venish/Evaluating-Classification-model-Performance/blob/main/graph1.png)

---------------------------------------------------------------------------------------

Note :- All the code in this project has been tested and run successfully in Google Colab. I encourage you to try running it in Colab for the best experience and to ensure smooth execution. Happy coding!

---------------------------------------------------------------------------------------

The given code performs various tasks related to logistic regression and evaluation of a classification model. Let's break down the code step by step:

1. **Data Loading**: The code imports necessary libraries, including pandas and numpy, and uses the `files` module from Google Colab to upload a CSV file named 'DigitalAd_dataset.csv'. The dataset is then loaded using `pd.read_csv()` into a DataFrame called `dataset`.

2. **Data Exploration**: The code displays the shape of the dataset using `dataset.shape` and the first 5 rows of the dataset using `dataset.head()`.

3. **Data Preprocessing**: The code separates the dataset into input features (`X`) and the target variable (`Y`). The features are extracted using `.iloc` to select all rows and all columns except the last one, and the target variable is extracted using `.iloc` to select all rows and only the last column.

4. **Train-Test Split**: The code splits the data into training and testing sets using `train_test_split` from `sklearn.model_selection`. It assigns the input features and target variables for both the training and testing sets to `X_train`, `X_test`, `y_train`, and `y_test`, respectively.

5. **Data Scaling**: The code uses `StandardScaler` from `sklearn.preprocessing` to standardize the features in the training and testing sets. It transforms the features in `X_train` using `.fit_transform()` and transforms the features in `X_test` using `.transform()`.

6. **Model Training**: The code creates an instance of `LogisticRegression` from `sklearn.linear_model` and fits the model to the training data using `.fit()`.

7. **Model Evaluation**: The code predicts the target variable for the testing set using `.predict()` and calculates the accuracy of the model using `accuracy_score` from `sklearn.metrics`. It prints the predicted and actual values of the target variable for comparison.

8. **Confusion Matrix**: The code computes the confusion matrix using `confusion_matrix` from `sklearn.metrics` and prints it.

9. **ROC Curve and AUC**: The code calculates the probabilities of the positive outcome using `.predict_proba()` and calculates the ROC AUC scores using `roc_auc_score` from `sklearn.metrics`. It also plots the ROC curve using `roc_curve` and `matplotlib.pyplot`.

10. **Cross-Validation**: The code performs cross-validation using `cross_val_score` from `sklearn.model_selection`. It performs 10-fold cross-validation using `KFold` and prints the mean cross-validation score.

11. **Stratified K-Fold Cross-Validation**: The code performs stratified k-fold cross-validation using `StratifiedKFold` and prints the mean cross-validation score.

12. **Cumulative Accuracy Profile (CAP) Curve**: The code plots the CAP curve to evaluate the model's performance. It plots a random model, a perfect model, and the logistic regression classifier's performance on the test set.

Overall, the code loads the dataset, preprocesses the data, trains a logistic regression model, evaluates the model using various metrics and visualizations, and performs cross-validation to assess the model's generalization performance.

---------------------------------------------------------------------------------------
Evaluating the performance of a classification model involves assessing how well the model is able to correctly classify instances into their respective classes. It helps determine the effectiveness and reliability of the model in making accurate predictions.

There are several commonly used metrics for evaluating classification model performance:

1. **Accuracy**: It measures the overall correctness of the model by calculating the ratio of correctly predicted instances to the total number of instances. Accuracy = (True Positives + True Negatives) / (True Positives + True Negatives + False Positives + False Negatives).

2. **Confusion Matrix**: It provides a summary of the predicted and actual class labels. It includes four metrics: true positives (TP), true negatives (TN), false positives (FP), and false negatives (FN). From the confusion matrix, other metrics like precision, recall, and F1-score can be derived.

3. **Precision**: Also known as the positive predictive value, it measures the accuracy of positive predictions. Precision = TP / (TP + FP). It focuses on minimizing false positives.

4. **Recall**: Also known as sensitivity or true positive rate, it measures the proportion of actual positives that are correctly identified. Recall = TP / (TP + FN). It focuses on minimizing false negatives.

5. **F1-Score**: It is the harmonic mean of precision and recall, providing a balance between the two metrics. F1-Score = 2 * (Precision * Recall) / (Precision + Recall).

6. **ROC Curve (Receiver Operating Characteristic Curve)**: It plots the true positive rate (TPR) against the false positive rate (FPR) at various classification thresholds. It helps assess the trade-off between sensitivity and specificity.

7. **AUC (Area Under the Curve)**: It represents the area under the ROC curve. A higher AUC indicates better classification performance.

8. **Cross-Validation**: It helps assess the model's generalization performance by dividing the dataset into multiple subsets and evaluating the model on each subset. It provides an estimate of how well the model will perform on unseen data.

These metrics and techniques play a crucial role in evaluating the performance of classification models and guiding the selection and fine-tuning of models for different tasks and datasets.

---------------------------------------------------------------------------------------
