from math import log2
from statistics import mean, stdev

import matplotlib.pyplot as plt
import seaborn as sns

from pandas import read_csv
from scipy.stats import norm

from model.linear_regression import LinearRegression
from optimizer.gradient_descent import GD


def main():
    # Read dataset and drop useless columns.
    housing = read_csv("housing.csv")
    housing = housing.drop(columns="ocean_proximity")
    housing = housing.dropna()

    # Calculate the split point between training dataset and test dataset.
    split_point = int(len(housing) * 0.8)

    # Select features and split them into training features and test features with standardizing.
    x = housing.drop(columns="median_house_value")
    x["total_bedrooms_over_total_rooms"] = x["total_bedrooms"] / x["total_rooms"]
    x["population_over_households"] = x["households"] / x["population"]
    x = x.drop(columns=["total_rooms", "population", "total_bedrooms", "households"])
    x_train = x.iloc[:split_point]
    x_test = x.iloc[split_point:]
    x_train_log = x_train.applymap(lambda value: -log2(abs(value)) if value < 0 else log2(value))
    x_test_log = x_test.applymap(lambda value: -log2(abs(value)) if value < 0 else log2(value))
    x_train_mean = x_train.mean()
    x_train_log_mean = x_train_log.mean()
    x_train_std = x_train.std()
    x_train_log_std = x_train_log.std()
    x_train_raw = (x_train - x_train_mean) / x_train_std
    x_test_raw = (x_test - x_train_mean) / x_train_std
    x_train_log = (x_train_log - x_train_log_mean) / x_train_log_std
    x_test_log = (x_test_log - x_train_log_mean) / x_train_log_std
    x_train_raw.insert(0, "leading_one", 1)
    x_test_raw.insert(0, "leading_one", 1)
    x_train_log.insert(0, "leading_one", 1)
    x_test_log.insert(0, "leading_one", 1)
    x_train_raw = x_train_raw.values
    x_test_raw = x_test_raw.values
    x_train_log = x_train_log.values
    x_test_log = x_test_log.values

    # Select labels and split them into training labels and test labels.
    y = housing["median_house_value"]
    y_train = y.iloc[:split_point]
    y_test = y.iloc[split_point:]
    y_train_raw = y_train.values
    y_test_raw = y_test.values
    y_train_log = y_train.map(log2).values
    y_test_log = y_test.map(log2).values

    # Initialize LinearRegression class.
    para_num = len(x.columns)
    linear_regression_raw = LinearRegression(inputs=para_num + 1)
    linear_regression_raw.compile(optimizer=GD(lr=0.3), loss=mean_squared_error)
    linear_regression_log = LinearRegression(inputs=para_num + 1)
    linear_regression_log.compile(optimizer=GD(lr=0.3), loss=mean_squared_error)

    # Train model and test model.
    train_loss_raw = linear_regression_raw.fit(x_train_raw, y_train_raw)
    test_loss_raw = linear_regression_raw.evaluate(x_test_raw, y_test_raw)
    train_loss_log = linear_regression_log.fit(x_train_log, y_train_log)
    test_loss_log = linear_regression_log.evaluate(x_test_log, y_test_log)

    # Get the prediction of training dataset and test dataset.
    y_train_pred_raw = linear_regression_raw.predict(x_train_raw)
    y_test_pred_raw = linear_regression_raw.predict(x_test_raw)
    y_train_pred_log = linear_regression_log.predict(x_train_log)
    y_test_pred_log = linear_regression_log.predict(x_test_log)
    y_train_pred_log = [2 ** value for value in y_train_pred_log]
    y_test_pred_log = [2 ** value for value in y_test_pred_log]

    # Print the loss of test dataset.
    print("Test dataset loss: {}".format(test_loss_raw))
    print("Test dataset loss (dataset scale: log): {}".format(test_loss_log))

    # Print the R2 score
    print("Adjusted R2 score and R2 score for training dataset: {}".format(
        r2_scores(y_train_pred_raw, y_train_raw, para_num)))
    print(
        "Adjusted R2 score and R2 score for test dataset: {}".format(r2_scores(y_test_pred_raw, y_test_raw, para_num)))
    print("Adjusted R2 score and R2 score for training dataset (dataset scale: log): {}".format(
        r2_scores(y_train_pred_log, y_train_raw, para_num)))
    print("Adjusted R2 score and R2 score for test dataset (dataset scale: log): {}".format(
        r2_scores(y_test_pred_log, y_test_raw, para_num)))

    # Plot loss changes line.
    plt.rc("font", size=20)
    plt.figure(figsize=(32, 24))

    line_plot(211, train_loss_raw)
    line_plot(212, train_loss_log, dataset_scale="log")

    plt.tight_layout()
    plt.show()

    # Plot prediction discrepancies scatter.
    plt.figure(figsize=(32, 24))

    scatter_plot(221, y_train_pred_raw, y_train_raw, dataset="Training")
    scatter_plot(222, y_test_pred_raw, y_test_raw, dataset="Test")
    scatter_plot(223, y_train_pred_log, y_train_raw, dataset="Training", dataset_scale="log")
    scatter_plot(224, y_test_pred_log, y_test_raw, dataset="Test", dataset_scale="log")

    plt.tight_layout()
    plt.show()

    # Plot Residual.
    plt.figure(figsize=(32, 24))

    resid_plot(221, y_train_pred_raw, y_train_raw, dataset="Training")
    resid_plot(222, y_test_pred_raw, y_test_raw, dataset="Test")
    resid_plot(223, y_train_pred_log, y_train_raw, dataset="Training", dataset_scale="log")
    resid_plot(224, y_test_pred_log, y_test_raw, dataset="Test", dataset_scale="log")

    plt.tight_layout()
    plt.show()

    # Plot residual hist.
    plt.figure(figsize=(32, 24))

    hist_plot(221, y_train_pred_raw, y_train_raw, dataset="Training")
    hist_plot(222, y_test_pred_raw, y_test_raw, dataset="Test")
    hist_plot(223, y_train_pred_log, y_train_raw, dataset="Training", dataset_scale="log")
    hist_plot(224, y_test_pred_log, y_test_raw, dataset="Test", dataset_scale="log")

    plt.tight_layout()
    plt.show()


def mean_squared_error(features, targets, weights):
    """
    Calculate the mean squared error.
    :param features: Input data.
    :param targets: Target data.
    :param weights: The coefficient of features.
    :return: Mean squared error.
    """
    return sum(label - sum(x * weight for x, weight in zip(feature, weights)) for feature, label in
               zip(features, targets)) ** 2 / len(targets)


def r2_scores(y_pred, y_true, p):
    """
    Calculate adjusted R2 score and R2 score.
    :param y_pred: Prediction data.
    :param y_true: Target data.
    :param p: Independent variable numbers.
    :return: (Adjusted R2 score, R2 score) pair.
    """
    y_true_mean = mean(y_true)
    sse = sum((pred - true) ** 2 for pred, true in zip(y_pred, y_true))
    sst = sum((true - y_true_mean) ** 2 for true in y_true)
    r2 = 1 - sse / sst
    n = len(y_true)

    return 1 - (1 - r2 ** 2) * (n - 1) / (n - p), r2


def line_plot(position, loss, dataset_scale=None):
    """
    Plot loss changes line.
    :param position: Subplot position.
    :param loss: Loss.
    :param dataset_scale: log or None.
    """
    plt.subplot(position)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.yscale("log")
    plt.title("Changes in Loss Function" + (" (dataset scale: log)" if dataset_scale == "log" else ""))

    sns.lineplot(loss)


def scatter_plot(position, y_pred, y_true, dataset=None, dataset_scale=None):
    """
    Plot prediction discrepancies scatter.
    :param position: Subplot position.
    :param y_pred: Prediction data.
    :param y_true: Target data.
    :param dataset: Training or test.
    :param dataset_scale: log or None.
    """
    plt.subplot(position)
    plt.xlabel("Target")
    plt.ylabel("Prediction")
    plt.title("Analyzing {} Dataset Prediction Discrepancies".format(dataset) + (
        " (dataset scale: log)" if dataset_scale == "log" else ""))

    sns.scatterplot(x=y_true, y=y_pred)


def resid_plot(position, y_pred, y_true, dataset=None, dataset_scale=None):
    """
    Plot residual.
    :param position: Subplot position.
    :param y_pred: Prediction data.
    :param y_true: Target data.
    :param dataset: Training or test.
    :param dataset_scale: log or None.
    """
    plt.subplot(position)
    plt.xlabel("Prediction")
    plt.ylabel("Residual")
    plt.title(
        "Residual Plot for {} Dataset".format(dataset) + (" (dataset scale: log)" if dataset_scale == "log" else ""))

    sns.residplot(x=y_true, y=y_pred)


def hist_plot(position, y_pred, y_true, dataset=None, dataset_scale=None):
    """
    Plot residual hist.
    :param position: Subplot position.
    :param y_pred: Prediction data.
    :param y_true: Target data.
    :param dataset: Training or test.
    :param dataset_scale: log or None.
    """
    residual = y_true - y_pred
    residual_mean = mean(residual)
    residual_std = stdev(residual)

    plt.subplot(position)
    plt.xlabel("Residual")
    plt.ylabel("Density")
    plt.title("Residual Histogram for {} Dataset".format(dataset) + (
        " (dataset scale: log)" if dataset_scale == "log" else ""))

    x = range(int(min(residual)), int(max(residual)))
    y = norm.pdf(x, residual_mean, residual_std)

    sns.histplot(residual, stat="density")
    sns.kdeplot(residual, color='r', label="Kernel distribution")
    sns.lineplot(x=x, y=y, color='g', label="Normal distribution")


if __name__ == "__main__":
    main()
