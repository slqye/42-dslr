import sys
from matplotlib import pyplot as plt
import pandas as pd
from describe import describe


def scatter_plot(path: str):
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
    f_features = describe(path)

    for i in range(len(f_keys)):
        for j in range(i, len(f_keys)):
            course1 = f_keys[i]
            course2 = f_keys[j]
            if course1 != course2:

                data1_normalized = ((f_dataset[course1] -
                                    f_features[course1]["mean"]) /
                                    f_features[course1]["std"])
                data2_normalized = ((f_dataset[course2] -
                                     f_features[course2]["mean"]) /
                                    f_features[course2]["std"])

                plt.scatter(data1_normalized,
                            data2_normalized,
                            alpha=0.5,
                            s=1)

    plt.gcf().set_dpi(500)
    plt.show()


def scatter_plot2(path: str):
    try:
        f_dataset = pd.read_csv(path)
    except FileNotFoundError:
        print("Error: dataset not found")
        exit()
    f_dataset = f_dataset.dropna()
    f_dataset = f_dataset.reset_index(drop=True)
    f_dataset = f_dataset.set_index("Hogwarts House")
    f_keys = ["Astronomy", "Defense Against the Dark Arts"]
    f_features = describe(path)

    course1 = f_keys[0]
    course2 = f_keys[1]
    data1_normalized = ((f_dataset[course1] -
                        f_features[course1]["mean"]) /
                        f_features[course1]["std"])
    data2_normalized = ((f_dataset[course2] -
                         f_features[course2]["mean"]) /
                        f_features[course2]["std"])

    plt.scatter(data1_normalized,
                data2_normalized,
                alpha=0.5,
                s=1)

    plt.xlabel(course1)
    plt.ylabel(course2)

    plt.gcf().set_dpi(500)
    plt.show()


def scatter_plot3(path: str, select: list):
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
    f_features = describe(path)

    f_n = len(select)
    f_m = len(f_keys)
    f_fig, f_axes = plt.subplots(f_n, f_m, figsize=(30, 4))
    for i, feature1 in enumerate(select):
        for j, feature2 in enumerate(f_features):
            data1_normalized = ((f_dataset[feature1] -
                                f_features[feature1]["mean"]) /
                                f_features[feature1]["std"])
            data2_normalized = ((f_dataset[feature2] -
                                 f_features[feature2]["mean"]) /
                                f_features[feature2]["std"])

            f_axes[i][j].scatter(data1_normalized,
                                 data2_normalized,
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

    scatter_plot(g_path)
    scatter_plot2(g_path)
