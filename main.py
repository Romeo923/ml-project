import numpy as np
from HopfieldNetwork import HopfieldNetwork, sign, sigmoid


def main():
    rows = 5
    cols = 5
    num_patterns = 5
    prob = 0.5
    num_cells = rows * cols

    training_patterns = [
        np.array([[1, -1], [-1, -1], [1, -1]]),
        np.array([[1, 1], [-1, 1], [1, 1]]),
        np.array([[-1, -1], [1, -1], [1, -1]]),
    ]

    (
        HopfieldNetwork(num_cells)
        # .train(training_patterns)
        .train_random_patterns(
            rows=rows, cols=cols, num_patterns=num_patterns, prob=prob
        )
        .run_simulation(
            instances=15,
            steps=10,
            rows=rows,
            cols=cols,
            activ_func=sigmoid(beta=1.5),
        )
        .plot_simulation("./plot/simulation.png")
    )


if __name__ == "__main__":
    main()
