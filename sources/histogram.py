import sys
from matplotlib import pyplot as plt
import pandas as pd
from describe import describe


def histogram(path: str):
    try:
        f_dataset = pd.read_csv(path)
    except FileNotFoundError:
        print("Error: dataset not found")
        exit()
    f_dataset = f_dataset.dropna()
    f_dataset = f_dataset.reset_index(drop=True)
    f_dataset = f_dataset.set_index("Hogwarts House")
    f_keys = list(f_dataset.keys())
    f_index = f_keys.index("Arithmancy")
    f_keys = f_keys[f_index:]
    f_houses = ["Ravenclaw", "Hufflepuff", "Slytherin", "Gryffindor"]

    f_nrow, f_ncol = 5, 3
    f_fig, f_axes = plt.subplots(f_nrow, f_ncol, figsize=(15, 15))
    i, j = 0, 0
    for course in f_keys:
        for house in f_houses:
            f_axes[i][j].hist(f_dataset[course].loc[house],
                              bins=50,
                              alpha=0.4,
                              cumulative=False)
            f_axes[i][j].set_xlabel("Notes")
            f_axes[i][j].set_ylabel("Amount")
            f_axes[i][j].set_title(course)
        j += 1
        if j >= f_ncol:
            i += 1
            j = 0

    f_fig.legend(f_houses)
    f_fig.tight_layout()
    plt.show()


def histogram2(path: str):
    try:
        f_dataset = pd.read_csv(path)
    except FileNotFoundError:
        print("Error: dataset not found")
        exit()
    f_dataset = f_dataset.dropna()
    f_dataset = f_dataset.reset_index(drop=True)
    f_dataset = f_dataset.set_index("Hogwarts House")
    f_houses = ["Ravenclaw", "Hufflepuff", "Slytherin", "Gryffindor"]

    f_n = 3
    f_fig, f_axes = plt.subplots(1, f_n, figsize=(15, 7))
    i = 0
    for course in ["Arithmancy", "Care of Magical Creatures", "Potions"]:
        for house in f_houses:
            f_axes[i].hist(f_dataset[course].loc[house],
                           bins=50,
                           alpha=0.4,
                           cumulative=False)
            f_axes[i].set_xlabel("Notes")
            f_axes[i].set_ylabel("Amount")
            f_axes[i].set_title(course)
        i += 1

    f_fig.legend(f_houses)
    f_fig.tight_layout()
    plt.show()


def histogram3(path: str):
    try:
        f_dataset = pd.read_csv(path)
    except FileNotFoundError:
        print("Error: dataset not found")
        exit()
    f_dataset = f_dataset.dropna()
    f_dataset = f_dataset.reset_index(drop=True)
    f_dataset = f_dataset.set_index("Hogwarts House")
    f_houses = ["Ravenclaw", "Hufflepuff", "Slytherin", "Gryffindor"]
    f_features = describe(path)

    f_n = 3
    f_fig, f_axes = plt.subplots(1, f_n, figsize=(15, 7))
    i = 0
    for course in ["Arithmancy", "Care of Magical Creatures", "Potions"]:
        for house in f_houses:
            f_axes[i].hist(f_dataset[course].loc[house],
                           bins=50,
                           alpha=0.4,
                           cumulative=False)
            if abs(f_features[course]["min"]) > abs(f_features[course]["max"]):
                f_max = abs(f_features[course]["min"])
            else:
                f_max = abs(f_features[course]["max"])
            f_axes[i].set_xlim((-f_max, f_max))
            f_axes[i].set_xlabel("Notes")
            f_axes[i].set_ylabel("Amount")
            f_axes[i].set_title(course)
        i += 1

    f_fig.legend(f_houses)
    f_fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    g_args = sys.argv
    if len(g_args) != 2:
        print("Error: program takes 1 argument, the path of the dataset")
        exit()
    g_path = g_args[1]
    histogram(g_path)
    histogram2(g_path)
    histogram3(g_path)
