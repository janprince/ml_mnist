import matplotlib
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
import numpy as np

# load mnist dataset
digits = load_digits()              # its a dictionary

keys = print(digits.keys())

# Data
X = digits['data']
y = digits['target']


# Plotting a sample image
some_digit = X[1119]
some_image_digit = some_digit.reshape(8, 8)
plt.imshow(some_image_digit, cmap= matplotlib.cm.binary, interpolation="nearest")
plt.axis("off")         # removing both the x and y axis
plt.show()

print(f"Actual Value: {y[1119]}")      # actual value of the X

# splitting the data
X_train, X_test, y_train, y_test = X[:1100], X[1100:], y[:1100], y[1100:]

shuffle_index = np.random.permutation(1100)
X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]


"""
Let’s simplify the problem for now and only try to identify one digit—for example,
the number 5. This “5-detector” will be an example of a binary classifier, capable of
distinguishing between just two classes, 5 and not-5. Let’s create the target vectors for
this classification task:
"""

# NB: Binary Classification; classifying only two classes; 5 or not 5

y_train_5 = (y_train == 5)  # True for all 5s; False for all other digits
y_test_5 = (y_test == 5)

# Now our y arrays have only two classes; True and False (5 or not 5)

from sklearn.linear_model import SGDClassifier

sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train_5)

# testing
some_digit_pred = sgd_clf.predict([some_digit])
print(f"predicted value: {some_digit_pred}")
