import random
from typing import Callable, TypedDict
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")

Result = TypedDict(
    "Result",
    {"output": np.ndarray | None, "ordered_params": list[np.ndarray]},
)
SimResult = TypedDict(
    "SimResult",
    {
        "ordered_params": list[list[np.ndarray]],
        "averages": list[np.ndarray],
        "pattern": int,
        "seeds": list[int],
    },
)


class HopfieldNetwork:
    def __init__(self, num_cells: int):
        self.N = num_cells
        self.W = np.zeros((num_cells, num_cells))
        self.patterns: list[np.ndarray] = []
        self._seed: int = 0
        self._states: list[np.ndarray] = []
        self._ordered_params: list[list[int]] = []
        self._result: Result = {"output": None, "ordered_params": []}
        self._sim_result: SimResult = {
            "ordered_params": [],
            "averages": [],
            "pattern": 0,
            "seeds": [],
        }

    def setSeed(self, seed: int):
        self._seed = seed
        random.seed(seed)
        return self

    def train(self, patterns: list[np.ndarray]) -> "HopfieldNetwork":
        self.patterns = patterns.copy()
        for pattern in patterns:
            pattern = pattern.flatten()
            self.W += np.outer(pattern, pattern) / self.N
            np.fill_diagonal(self.W, 0)
        return self

    def train_random_patterns(
        self, rows: int, cols: int, num_patterns: int, prob: float
    ) -> "HopfieldNetwork":
        patterns = generateRandomPatterns(
            rows=rows, cols=cols, num_patterns=num_patterns, prob=prob
        )
        self.train(patterns=patterns)

        return self

    def calc_ordered_params(self) -> "HopfieldNetwork":
        ordered_params = []
        for pattern in self.patterns:
            pat = pattern.flatten()
            ops = []
            for state in self._states:
                st = state.flatten()
                op = np.dot(st, pat)
                op /= self.N
                ops.append(op)
            ordered_params.append(np.array(ops))
        self._ordered_params = ordered_params
        self._result["ordered_params"] = ordered_params
        return self

    def _update_cell(self, input: np.ndarray, i: int, activ_func) -> np.ndarray:
        shape = input.shape
        s = input.flatten()

        b = np.dot(self.W[i], s)
        s[i] = activ_func(b)

        return s.reshape(shape)

    def async_update(
        self,
        input: np.ndarray,
        steps: int = 10,
        rand_select: bool = True,
        activ_func: Callable[[float], int] = np.sign,
    ) -> "HopfieldNetwork":
        s = input
        self._states = [s]
        cells = (
            random.sample(list(range(self.N)), self.N) if rand_select else range(self.N)
        )
        for _ in range(steps):
            for cell in cells:
                s = self._update_cell(s, cell, activ_func=activ_func)
            self._states.append(s)
        self._result["output"] = s
        return self

    def run_simulation(
        self,
        instances: int,
        steps: int,
        rows: int,
        cols: int,
        activ_func: Callable[[float], int],
    ):
        # selects a random pattern
        idx = random.randint(0, len(self.patterns) - 1)
        # different seed for each instance
        seeds = [i + 1 for i in range(instances)]
        self._sim_result["seeds"] = seeds
        # ordered params for each simulation
        sim_ops = [
            (
                HopfieldNetwork(rows * cols)
                .train(self.patterns)
                .setSeed(seeds[i])
                .async_update(
                    input=self.patterns[idx],
                    steps=steps,
                    activ_func=activ_func,
                )
                .calc_ordered_params()
                .result()["ordered_params"]
            )
            for i in range(instances)
        ]
        # calculate ordered param averages for each simulation
        sim_avg: list[np.ndarray] = []
        for pattern in range(len(self.patterns)):
            pattern_avg = np.zeros((1 + steps,))
            for sim in sim_ops:
                pattern_ops = sim[pattern]
                pattern_avg += pattern_ops
            pattern_avg /= instances
            sim_avg.append(pattern_avg)

        self._sim_result["ordered_params"] = sim_ops
        self._sim_result["averages"] = sim_avg
        self._sim_result["pattern"] = idx

        return self

    def result(self):
        return self._result

    def sim_result(self):
        return self._sim_result

    def plot_state_evolution(
        self, img_path: str = "./plot/hopfield.png"
    ) -> "HopfieldNetwork":
        rows = 1 + len(self.patterns)
        cols = len(self._states)
        fig, axs = plt.subplots(rows, cols)

        # Plots evolution of states
        for t, s in enumerate(self._states):
            state_plot = (s - 1) // -2
            axs[0, t].axis("off")
            color_map = plt.cm.colors.ListedColormap(["grey", "black"])
            axs[0, t].imshow(state_plot, cmap=color_map)
            axs[0, t].set_title(f"S({t=})")

        # Plots Trained Patterns
        for i, pattern in enumerate(self.patterns):
            state_plot = (pattern - 1) // -2
            color_map = plt.cm.colors.ListedColormap(["grey", "black"])
            axs[i + 1, 0].imshow(state_plot, cmap=color_map)
            axs[i + 1, 0].set_title(f"Pattern {i+1}")
            [axs[i + 1, j].axis("off") for j in range(len(self._states))]

        # Ordered Params
        for i, op in enumerate(self._ordered_params):
            axs[i + 1, 1].axis("on")
            axs[i + 1, 1].plot(op)
            axs[i + 1, 1].set_ylim(-1.5, 1.5)

        # Saves to file
        fig.suptitle(img_path)
        plt.tight_layout(h_pad=1.5)
        plt.savefig(img_path)
        return self

    def plot_simulation(
        self, img_path: str = "./plot/simulation.png"
    ) -> "HopfieldNetwork":
        cols = 5
        num_rows = int(np.ceil(len(self.patterns) / cols))
        total_rows = 2 * num_rows
        fig, axs = plt.subplots(total_rows, cols, figsize=(cols * 2, total_rows * 2))
        [axs[i, j].axis("off") for i in range(total_rows) for j in range(cols)]

        # Plots Trained Patterns
        for i, pattern in enumerate(self.patterns):
            state_plot = (pattern - 1) // -2
            color_map = plt.cm.colors.ListedColormap(["grey", "black"])
            row = 0 + 2 * (i // cols)
            axs[row, i % cols].imshow(state_plot, cmap=color_map)
            axs[row, i % cols].set_title(f"Pattern {i+1}")

        sim_ops = self._sim_result["ordered_params"]
        # Ordered Params
        for i, avg_op in enumerate(self._sim_result["averages"]):
            row = 1 + 2 * (i // cols)
            [
                axs[row, i % cols].plot(sim_ops[sim][i], color="grey")
                for sim in range(len(sim_ops))
            ]
            axs[row, i % cols].plot(avg_op, color="r")
            axs[row, i % cols].set_ylim(-1.5, 1.5)
            axs[row, i % cols].set_xlabel("T")
            axs[row, i % cols].set_ylabel(f"m{i+1}(T)")
            axs[row, i % cols].axis("on")

        # Saves to file
        pattern_idx = self._sim_result["pattern"]
        itterations = len(self._sim_result["ordered_params"])
        fig.suptitle(f"Pattern: {pattern_idx + 1}, Itterations: {itterations}")
        plt.tight_layout(h_pad=1.5)
        plt.savefig(img_path)
        return self

# activation function
def sign(x: float) -> int:
    return 1 if x >= 0 else -1

# returns activation function
def sigmoid(beta: float = 1):

    def p(x: float) -> int:
        power = -2 * beta * x
        result = 1 + np.exp(power)
        prob = 1 / result
        return random.choices((1, -1), weights=(prob, 1 - prob), k=1)[0]

    return p


def generateRandomState(rows, cols, prob):
    value = randomValue(prob)
    state = [[next(value) for _ in range(cols)] for _ in range(rows)]
    return np.array(state)


def generateRandomPatterns(rows, cols, num_patterns, prob):
    pattern_list = []
    while len(pattern_list) != num_patterns:
        pattern = generateRandomState(rows, cols, prob)
        is_in_list = any(np.array_equal(pattern, p) for p in pattern_list)
        if not is_in_list or len(pattern_list) == 0:
            pattern_list.append(pattern)
    return pattern_list


def randomValue(prob):
    while True:
        x = random.random()
        yield 1 if x < prob else -1
