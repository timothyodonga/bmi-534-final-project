from random import shuffle
import pandas as pd


def generate_folds(subjects, n_folds=5):
    """
    This function generates n_folds (default 5) of train and test splits from a list of subjects.

    Args:
        subjects: A list of subjects.
        n_folds: The number of folds for cross-validation (default 5).

    Returns:
        A list of dictionaries. Each dictionary contains two keys:
            'train': A list of subjects for training in the current fold.
            'test': A list of subjects for testing in the current fold.
    """

    # Shuffle the list of subjects for randomization
    shuffle(subjects)

    # Calculate the size of each fold
    fold_size = len(subjects) // n_folds

    # Initialize an empty list to store folds
    folds = {}

    # Iterate through n_folds
    for i in range(n_folds):
        # Starting and ending indices for the current fold's test set
        test_start = i * fold_size
        test_end = (i + 1) * fold_size

        # Extract test subjects for the current fold
        test_set = subjects[test_start:test_end]

        # Create a copy of the subject list to avoid modifying the original
        train_set = subjects.copy()

        # Remove test subjects from the training set
        del train_set[test_start:test_end]

        # Create a dictionary for the current fold
        fold = {"train": train_set, "test": test_set}

        # Append the fold dictionary to the folds list
        folds[f"{i}"] = fold

    return folds


# Do this for the HARTH dataset
df = pd.read_csv(
    r"C:\Users\timot\OneDrive\Desktop\EMORY\Spring 2024\BMI-534\project-code\code\bmi-534-final-project\harth_preprocessed_data_150_window.csv"
)

folds = generate_folds(subjects=list(df["subject"].unique()))
print(folds)
