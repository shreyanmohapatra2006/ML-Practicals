MACHINE LEARNING PRACTICAL FILE – 18 PROGRAMS
Codes are selectable and copy-paste friendly.
Note: These programs use built-in sklearn datasets; no keyboard input is needed.

============================================================================
1. K-Fold Cross Validation
============================================================================

CODE:
from sklearn.datasets import load_iris
from sklearn.model_selection import KFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier

iris = load_iris()
X = iris.data
y = iris.target

kfold = KFold(n_splits=5, shuffle=True, random_state=42)
model = DecisionTreeClassifier(random_state=42)

scores = cross_val_score(model, X, y, cv=kfold, scoring="accuracy")

print("Cross Validation Scores:", scores)
print("Mean CV Accuracy:", scores.mean())

SAMPLE INPUT:
Manual input: Not required. The program loads the Iris dataset automatically.

SAMPLE OUTPUT:
Cross Validation Scores: [1.         0.96666667 0.93333333 0.93333333 0.93333333]
Mean CV Accuracy: 0.9533333333333334


============================================================================
2. SLP – Single Layer Perceptron (Iris Dataset)
============================================================================

CODE:
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score, confusion_matrix

iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=21, stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = Perceptron(max_iter=1500, eta0=0.1, random_state=21)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Model Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

new_flower = [[5.4, 3.7, 1.5, 0.2]]
new_flower_scaled = scaler.transform(new_flower)
prediction = model.predict(new_flower_scaled)

print("Predicted Flower:", iris.target_names[prediction[0]])

SAMPLE INPUT:
Manual input: Not required. The Iris dataset and new flower values are given in the code.

SAMPLE OUTPUT:
Model Accuracy: 0.9736842105263158
Confusion Matrix:
[[11  1  0]
 [ 0 13  0]
 [ 0  0 13]]
Predicted Flower: setosa


============================================================================
3. MLP – Multi-Layer Perceptron (Iris Dataset)
============================================================================

CODE:
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=21, stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = MLPClassifier(
    hidden_layer_sizes=(64, 32),
    max_iter=2000,
    random_state=21
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("MLP Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

new_flower = [[5.4, 3.7, 1.5, 0.2]]
new_flower_scaled = scaler.transform(new_flower)
prediction = model.predict(new_flower_scaled)

print("Predicted Flower:", iris.target_names[prediction[0]])

SAMPLE INPUT:
Manual input: Not required. The Iris dataset and new flower values are given in the code.

SAMPLE OUTPUT:
MLP Accuracy: 0.9736842105263158
Confusion Matrix:
[[12  0  0]
 [ 0 12  1]
 [ 0  0 13]]
Predicted Flower: setosa


============================================================================
4. KNN – K-Nearest Neighbours (Iris Dataset)
============================================================================

CODE:
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=21, stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = KNeighborsClassifier(n_neighbors=7)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("KNN Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

new_flower = [[5.4, 3.7, 1.5, 0.2]]
new_flower_scaled = scaler.transform(new_flower)
prediction = model.predict(new_flower_scaled)

print("Predicted Flower:", iris.target_names[prediction[0]])

SAMPLE INPUT:
Manual input: Not required. The Iris dataset and new flower values are given in the code.

SAMPLE OUTPUT:
KNN Accuracy: 0.9210526315789473
Confusion Matrix:
[[12  0  0]
 [ 0 12  1]
 [ 0  2 11]]
Predicted Flower: setosa


============================================================================
5. PCA – Principal Component Analysis
============================================================================

CODE:
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

iris = load_iris()
X = iris.data
y = iris.target

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print("Original Shape:", X.shape)
print("Reduced Shape:", X_pca.shape)
print("Explained Variance Ratio:", pca.explained_variance_ratio_)
print("Total Variance Retained:", pca.explained_variance_ratio_.sum())

for i, name in enumerate(iris.target_names):
    plt.scatter(X_pca[y == i, 0], X_pca[y == i, 1], label=name)

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA on Iris Dataset")
plt.legend()
plt.show()

SAMPLE INPUT:
Manual input: Not required. The program loads the Iris dataset automatically.

SAMPLE OUTPUT:
Original Shape: (150, 4)
Reduced Shape: (150, 2)
Explained Variance Ratio: [0.72962445 0.22850762]
Total Variance Retained: 0.9581320720000166
A PCA scatter graph is also displayed.


============================================================================
6. Regularization – Lasso and Ridge Regression
============================================================================

CODE:
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import mean_squared_error, r2_score

data = load_diabetes()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=21
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

lasso = Lasso(alpha=0.5, max_iter=5000)
lasso.fit(X_train, y_train)
lasso_pred = lasso.predict(X_test)

print("Lasso Regression")
print("MSE:", mean_squared_error(y_test, lasso_pred))
print("R2 Score:", r2_score(y_test, lasso_pred))
print("Coefficients:", lasso.coef_)

ridge = Ridge(alpha=0.5)
ridge.fit(X_train, y_train)
ridge_pred = ridge.predict(X_test)

print("\nRidge Regression")
print("MSE:", mean_squared_error(y_test, ridge_pred))
print("R2 Score:", r2_score(y_test, ridge_pred))
print("Coefficients:", ridge.coef_)

SAMPLE INPUT:
Manual input: Not required. The program loads the Diabetes dataset automatically.

SAMPLE OUTPUT:
Lasso Regression
MSE: 3088.2505269318017
R2 Score: 0.4480709263332643

Ridge Regression
MSE: 3053.1971780465806
R2 Score: 0.45433562610759204
(Coefficient arrays are also printed.)


============================================================================
7. Feature Selection – Best Subset Selection
============================================================================

CODE:
from itertools import combinations
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=21, stratify=y
)

best_score = 0
best_features = None

for size in range(1, X.shape[1] + 1):
    for subset in combinations(range(X.shape[1]), size):
        scaler = StandardScaler()
        X_train_subset = scaler.fit_transform(X_train[:, subset])
        X_test_subset = scaler.transform(X_test[:, subset])

        model = LogisticRegression(max_iter=2000)
        model.fit(X_train_subset, y_train)

        predictions = model.predict(X_test_subset)
        score = accuracy_score(y_test, predictions)

        if score > best_score:
            best_score = score
            best_features = subset

print("Best Features:")
for index in best_features:
    print(feature_names[index])

print("Best Test Accuracy:", best_score)

SAMPLE INPUT:
Manual input: Not required. The program loads the Iris dataset automatically.

SAMPLE OUTPUT:
Best Features:
petal width (cm)
Best Test Accuracy: 1.0


============================================================================
8. Feature Selection – Backward Elimination
============================================================================

CODE:
from sklearn.datasets import load_iris
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

iris = load_iris()
X = iris.data
y = iris.target

model = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=2000)
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=21)

selector = SequentialFeatureSelector(
    model,
    n_features_to_select=2,
    direction="backward",
    scoring="accuracy",
    cv=cv
)

selector.fit(X, y)

print("Selected Features:")
for name, selected in zip(iris.feature_names, selector.get_support()):
    if selected:
        print(name)

print("Feature Mask:", selector.get_support())

SAMPLE INPUT:
Manual input: Not required. The program loads the Iris dataset automatically.

SAMPLE OUTPUT:
Selected Features:
petal length (cm)
petal width (cm)
Feature Mask: [False False  True  True]


============================================================================
9. Feature Selection – Forward Selection
============================================================================

CODE:
from sklearn.datasets import load_iris
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

iris = load_iris()
X = iris.data
y = iris.target

model = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=2000)
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=21)

selector = SequentialFeatureSelector(
    model,
    n_features_to_select=2,
    direction="forward",
    scoring="accuracy",
    cv=cv
)

selector.fit(X, y)

print("Selected Features:")
for name, selected in zip(iris.feature_names, selector.get_support()):
    if selected:
        print(name)

print("Feature Mask:", selector.get_support())

SAMPLE INPUT:
Manual input: Not required. The program loads the Iris dataset automatically.

SAMPLE OUTPUT:
Selected Features:
sepal length (cm)
petal width (cm)
Feature Mask: [ True False False  True]


============================================================================
10. Holdout Method
============================================================================

CODE:
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=21, stratify=y
)

model = DecisionTreeClassifier(max_depth=4, random_state=21)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Holdout Accuracy:", accuracy_score(y_test, y_pred))

SAMPLE INPUT:
Manual input: Not required. The program loads the Iris dataset automatically.

SAMPLE OUTPUT:
Holdout Accuracy: 0.9736842105263158


============================================================================
11. Support Vector Machine (SVM)
============================================================================

CODE:
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=21, stratify=y
)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create SVM model
model = SVC(kernel='linear')
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Evaluation
print("SVM Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Predict new flower
new_flower = [[5.4, 3.7, 1.5, 0.2]]
new_flower_scaled = scaler.transform(new_flower)
prediction = model.predict(new_flower_scaled)

print("Predicted Flower:", iris.target_names[prediction[0]])

SAMPLE INPUT:
Manual input: Not required. The program loads the Iris dataset and new flower values automatically.

SAMPLE OUTPUT:
SVM Accuracy: 0.9736842105263158
Confusion Matrix:
[[12  0  0]
 [ 0 12  1]
 [ 0  0 13]]
Predicted Flower: setosa


============================================================================
12. Hierarchical Clustering
============================================================================

CODE:
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt

# Load dataset
iris = load_iris()
X = iris.data

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Create linkage matrix
Z = linkage(X_scaled, method='ward')

# Plot dendrogram
plt.figure(figsize=(10, 5))
dendrogram(Z)
plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("Data Points")
plt.ylabel("Distance")
plt.show()

# Apply Agglomerative Clustering
model = AgglomerativeClustering(
    n_clusters=3,
    linkage='ward'
)

y_pred = model.fit_predict(X_scaled)

print("Cluster Labels:")
print(y_pred)

print("Number of Clusters:", len(set(y_pred)))

SAMPLE INPUT:
Manual input: Not required. The program loads the Iris dataset automatically.

SAMPLE OUTPUT:
Cluster Labels: (array of cluster labels for the 150 Iris records)
Number of Clusters: 3
A hierarchical clustering dendrogram is also displayed.


============================================================================
13. Naive Bayes Classifier
============================================================================

CODE:
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=21, stratify=y
)

# Create Naive Bayes model
model = GaussianNB()

# Train model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model
print("Naive Bayes Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("Classification Report:")
print(classification_report(y_test, y_pred))

# Predict a new flower
new_flower = [[5.4, 3.7, 1.5, 0.2]]
prediction = model.predict(new_flower)

print("Predicted Flower:", iris.target_names[prediction[0]])

SAMPLE INPUT:
Manual input: Not required. The program loads the Iris dataset and new flower values automatically.

SAMPLE OUTPUT:
Naive Bayes Accuracy: 0.9736842105263158
Confusion Matrix:
[[12  0  0]
 [ 0 12  1]
 [ 0  0 13]]
Predicted Flower: setosa
(A classification report is printed between the confusion matrix and prediction.)


============================================================================
14. Simple Linear Regression
============================================================================

CODE:
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
data = load_diabetes()
X = data.data
y = data.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=21
)

# Create Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Evaluation
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

print("Actual Value:", y_test[0])
print("Predicted Value:", y_pred[0])

SAMPLE INPUT:
Manual input: Not required. The program loads the Diabetes dataset automatically.

SAMPLE OUTPUT:
Mean Squared Error: 3049.985676559466
R2 Score: 0.4549095824707189
Actual Value: 178.0
Predicted Value: 166.54289144555278


============================================================================
15. Multiple Linear Regression
============================================================================

CODE:
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
data = load_diabetes()
X = data.data
y = data.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=21
)

# Create model
model = LinearRegression()

# Train model
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Evaluation
print("Multiple Linear Regression")
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

print("Actual Value:", y_test[0])
print("Predicted Value:", y_pred[0])
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)

SAMPLE INPUT:
Manual input: Not required. The program loads the Diabetes dataset automatically.

SAMPLE OUTPUT:
Multiple Linear Regression
MSE: 3049.985676559466
R2 Score: 0.4549095824707189
Actual Value: 178.0
Predicted Value: 166.54289144555278
(Coefficients and intercept are also printed.)


============================================================================
16. Polynomial Regression
============================================================================

CODE:
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics
