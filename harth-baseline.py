import numpy as np
import pandas as pd
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
import datetime
import math

pd.set_option('display.max_columns', None)


def load_harth():
    # Get files
    files = os.listdir('harth/')

    # List of subjects datasets
    subjects = []

    for file in files:
        df = pd.read_csv('harth/' + file)

        # Some have an extra column called 'index'
        if 'index' in df.columns:
            df = df.drop(columns=['index'])

        # Some have an unnamed index column
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

        # We will use a sliding window, so don't need timestamp
        df = df.drop(columns=['timestamp'])

        df = df.to_numpy()

        # Get chunks of length 150
        chunks = []

        # Aggregate chunks to one line
        while len(df) > 0:
            chunk = df[:150, :]
            df = df[150:, :]

            # We will put all of our desired features into this list
            all_features = []

            # We take the means as our first five features
            means = chunk.mean(axis=0)
            all_features.extend(means[:-1])

            # Then, we take the standard deviations as our next 5
            stds = chunk.std(axis=0)
            all_features.extend(stds[:-1])

            # Take final label as label for the chunk
            all_features.append(chunk[len(chunk) - 1, 6])

            chunks.append(np.array(all_features))

            '''
            # Take final label as label for the chunk
            means[6] = chunk[len(chunk) - 1, 6]

            chunks.append(means)
            '''

        # Put chunked aggregations back into a single dataset and append to subject list
        subjects.append(np.vstack(chunks))

    return subjects


# Run KNN
def run_knn(x_tr, x_te, y_tr, y_te):
    clf = KNeighborsClassifier()
    clf.fit(x_tr, y_tr)

    acc = clf.score(x_te, y_te)

    y_pred = clf.predict_proba(x_te)
    try:
        auc = roc_auc_score(y_te, y_pred, multi_class='ovr')
    except:
        auc = math.nan

    y_pred = clf.predict(test_X)
    try:
        f1 = f1_score(y_te, y_pred, average='macro')
    except:
        f1 = math.nan

    return np.array([acc, auc, f1])


def run_nb(x_tr, x_te, y_tr, y_te):
    clf = GaussianNB()
    clf.fit(x_tr, y_tr)

    acc = clf.score(x_te, y_te)

    y_pred = clf.predict_proba(x_te)
    try:
        auc = roc_auc_score(y_te, y_pred, multi_class='ovr')
    except:
        auc = math.nan

    y_pred = clf.predict(test_X)
    try:
        f1 = f1_score(y_te, y_pred, average='macro')
    except:
        f1 = math.nan

    return np.array([acc, auc, f1])


def run_rf(x_tr, x_te, y_tr, y_te):
    clf = RandomForestClassifier()
    clf.fit(x_tr, y_tr)

    acc = clf.score(x_te, y_te)

    y_pred = clf.predict_proba(x_te)
    try:
        auc = roc_auc_score(y_te, y_pred, multi_class='ovr')
    except:
        auc = math.nan

    y_pred = clf.predict(test_X)
    try:
        f1 = f1_score(y_te, y_pred, average='macro')
    except:
        f1 = math.nan

    return np.array([acc, auc, f1])


def run_log_reg(x_tr, x_te, y_tr, y_te):
    clf = LogisticRegression(max_iter=500)
    clf.fit(x_tr, y_tr)

    acc = clf.score(x_te, y_te)

    y_pred = clf.predict_proba(x_te)
    try:
        auc = roc_auc_score(y_te, y_pred, multi_class='ovr')
    except:
        auc = math.nan

    y_pred = clf.predict(test_X)
    try:
        f1 = f1_score(y_te, y_pred, average='macro')
    except:
        f1 = math.nan

    return np.array([acc, auc, f1])


if __name__ == '__main__':

    print(datetime.datetime.now())
    print('Gathering data...')
    print()
    subs = load_harth()

    # 10 trials, 5 folds, 3 metrics: Accuracy, AUC, F1-Score
    knn_stats = np.zeros((50, 3))
    nb_stats = np.zeros((50, 3))
    rf_stats = np.zeros((50, 3))
    log_reg_stats = np.zeros((50, 3))

    for trial in range(10):
        print('TRIAL %d' % trial)

        kf = KFold(shuffle=True)

        for k, (train_index, test_index) in enumerate(kf.split(subs)):
            print(datetime.datetime.now())
            print('Fold number %d' % k)
            print()

            # Divide data
            train = np.vstack([subs[i] for i in train_index])
            train_X = train[:, :-1]
            train_Y = train[:, -1]
            test = np.vstack([subs[i] for i in test_index])
            test_X = test[:, :-1]
            test_Y = test[:, -1]

            # Run KNN
            knn_stats[(trial * 5) + k] = run_knn(train_X, test_X, train_Y, test_Y)

            # Run Naive Bayes
            nb_stats[(trial * 5) + k] = run_nb(train_X, test_X, train_Y, test_Y)

            # Run Random Forest
            rf_stats[(trial * 5) + k] = run_rf(train_X, test_X, train_Y, test_Y)

            # Run Logistic Regression
            log_reg_stats[(trial * 5) + k] = run_log_reg(train_X, test_X, train_Y, test_Y)


    # Get summary statistics
    print(datetime.datetime.now())
    print('Finished running tests. Results:')
    print('KNN Average Accuracy, AUC, and F1:\t', np.nanmean(knn_stats, axis=0))
    print('KNN Std Dev Accuracy, AUC, and F1:\t', np.nanstd(knn_stats, axis=0))
    print('NB Average Accuracy, AUC, and F1:\t', np.nanmean(nb_stats, axis=0))
    print('NB Std Dev Accuracy, AUC, and F1:\t', np.nanstd(nb_stats, axis=0))
    print('RF Average Accuracy, AUC, and F1:\t', np.nanmean(rf_stats, axis=0))
    print('RF Std Dev Accuracy, AUC, and F1:\t', np.nanstd(rf_stats, axis=0))
    print('Log Reg Average Accuracy, AUC, and F1:\t', np.nanmean(log_reg_stats, axis=0))
    print('Log Reg Std Dev Accuracy, AUC, and F1:\t', np.nanstd(log_reg_stats, axis=0))
