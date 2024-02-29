import argparse
import numpy as np
from sklearn.metrics import adjusted_rand_score as arc


def evaluate(y_pred, y_val):
    """
    Evaluates model predictions with the Adjusted rand Score metric.

    :param y_pred: ndarray, model predictions, shape (n_images, 512x512)
    :param y_val: ndarray, ground truth, shape (n_images, 512x512)

    :return: float, evaluation score.
    """
    rands = [
        arc(pred, truth) for pred, truth in zip(y_pred, y_val)
    ]

    return np.mean(rands)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-preds", required=True)
    parser.add_argument("-truths", default="./preds/y_val.npy")
    args = parser.parse_args()

    y_pred = np.load(args.preds)
    y_val = np.load(args.truths)

    print("Evaluating...")
    score = evaluate(y_pred, y_val)
    print(f"Score: {score}")







