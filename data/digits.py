
from sklearn import datasets
from sklearn.model_selection import train_test_split

def load_digits_dataset():
    # Load dataset
    digits = datasets.load_digits()

    X = digits.data
    y = digits.target

    # Split training and testing data
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # Normalize pixel values
    X_train_scaled = X_train / 16.0
    X_test_scaled = X_test / 16.0

    return X_train_scaled, X_test_scaled, y_train, y_test
