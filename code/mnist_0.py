from sklearn.datasets import load_digits
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split


digits = load_digits()                      # type(digits) = bunch; similar to dictionary

# separating predictors(independent variables) from targets(or labels or dependent variables)
features = digits['data']
labels = digits['target']

# Splitting data into training and testing set
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, stratify=labels)

# Classifier
knn = KNeighborsClassifier(n_neighbors=3)

# Fit classifier to training set
knn.fit(X_train, y_train)

# Check accuracy on the training set
training_accuracy = knn.score(X_train, y_train)

# Check accuracy on the test set
testing_accuracy = knn.score(X_test, y_test)

print(f"Training Accuracy: {training_accuracy}")
print(f"Testing Accuracy: {testing_accuracy}")