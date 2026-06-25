from __future__ import annotations

import numpy as np


def bland_altman_plot(actuals, predictions, *, ax=None):
    import matplotlib.pyplot as plt

    actuals = np.asarray(actuals)
    predictions = np.asarray(predictions)
    differences = predictions - actuals
    means = (predictions + actuals) / 2
    mean_difference = np.mean(differences)
    sd_difference = np.std(differences)
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(means, differences, alpha=0.5)
    ax.axhline(mean_difference, color="red", linestyle="--")
    ax.axhline(mean_difference + 1.96 * sd_difference, color="gray", linestyle="--")
    ax.axhline(mean_difference - 1.96 * sd_difference, color="gray", linestyle="--")
    ax.set_title("Bland-Altman Plot")
    ax.set_xlabel("Mean of Predictions and Actuals")
    ax.set_ylabel("Prediction - Actual")
    ax.grid(True)
    return ax


def visualize_linear_layer_weights(linear_layer, *, ax=None):
    import matplotlib.pyplot as plt

    weights = linear_layer.weight.detach().cpu().numpy().squeeze()
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6))
    ax.bar(np.arange(len(weights)), weights)
    ax.set_xlabel("Feature Index")
    ax.set_ylabel("Weight")
    ax.set_title("Linear Layer Weights")
    return ax
