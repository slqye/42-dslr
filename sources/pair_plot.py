import sys
from matplotlib import pyplot as plt
import pandas as pd
from describe import describe


def plair_plot(path: str):
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
    f_features = describe(path)

    f_n = len(f_features)
    f_fig, f_axes = plt.subplots(f_n, f_n, figsize=(30, 30))
    for i, feature1 in enumerate(f_features):
        for j, feature2 in enumerate(f_features):
            if feature1 == feature2:
                for house in f_houses:
                    f_axes[i][j].hist(f_dataset[feature1].loc[house],
                                      bins=25,
                                      alpha=0.5)
            else:
                for house in f_houses:
                    f_axes[i][j].scatter(f_dataset[feature1].loc[house],
                                         f_dataset[feature2].loc[house],
                                         alpha=0.5,
                                         s=1)
            if i != f_n - 1:
                f_axes[i][j].set_xticks([])
            else:
                f_axes[i][j].set_xlabel(feature2)
            if j != 0:
                f_axes[i][j].set_yticks([])
            else:
                f_axes[i][j].set_ylabel(feature1)

    f_fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    g_args = sys.argv
    if len(g_args) != 2:
        print("Error: program takes 1 argument, the path of the dataset")
        exit()
    g_path = g_args[1]
    plair_plot(g_path)
