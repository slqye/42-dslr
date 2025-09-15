import pandas as pd
import sys


def ft_count(data: list):
    """function to count the number of values in a dataset"""
    f_n: int = 0
    for _ in data:
        f_n += 1
    return f_n


def ft_sum(data: list):
    """function to calculate the sum of the values in a dataset"""
    f_res: int = 0
    for value in data:
        f_res += value
    return f_res


def ft_mean(data: list):
    """function to calculate the mean of the values in a dataset"""
    f_n = ft_count(data)
    f_sum = ft_sum(data)
    return f_sum / f_n


def ft_mode(data: list):
    """
    function to calculate the mode of the values in a dataset, 
    the mode is the value with most occurence in the dataset"""

    if len(data) == 0:
        return None

    sorted_list = sorted(data)

    count = 0
    last_value = sorted_list[0]

    max_count = 0
    mode = None

    for value in sorted_list:
        if last_value == value:
            count += 1
        else:
            if count > max_count:
                max_count = count
                mode = last_value
            elif count == max_count:
                mode = None
            count = 1
        last_value = value
    
    if count > max_count:
        mode = last_value
    elif count == max_count:
        mode = None

    return mode


def ft_min(data: list):
    """function to find the minimum value of a dataset"""
    f_min = data[0]
    for value in data:
        if value < f_min:
            f_min = value
    return f_min


def ft_max(data: list):
    """function to find the maximum value of a dataset"""
    f_max = data[0]
    for value in data:
        if value > f_max:
            f_max = value
    return f_max


def ft_var(data: list):
    """function to compute the variance of a dataset"""
    f_mean = ft_mean(data)
    f_n = ft_count(data)
    f_sum = 0
    for value in data:
        f_sum += (value - f_mean) ** 2
    f_var = f_sum / f_n
    return f_var


def ft_std(data: list):
    """function to calculate the standard deviation
of the values in a dataset"""
    f_var = ft_var(data)
    return f_var ** 0.5


def ft_percentile(data: list, perc: int):
    """function to calculate the percentile value in a dataset
data: dataset
perc: the percentile wanted (0-100)"""
    f_n = ft_count(data)
    f_r = (perc / 100) * (f_n - 1) + 1
    f_ri = int(f_r)
    f_rf = f_r - int(f_r)
    f_sorted = sorted(data)
    f_p = f_sorted[f_ri] + f_rf * (f_sorted[f_ri + 1] - f_sorted[f_ri])
    return f_p


def ft_skewness(data: list):
    """function to calculate the skewness of a dataset"""
    f_mean = ft_mean(data)
    f_median = ft_percentile(data, 50)
    f_std = ft_std(data)
    f_skew = (3 * (f_mean - f_median)) / f_std
    return f_skew


def describe(path: str, verbose: bool = False):
    """
    describe function
    displays informations about a dataset
    """
    f_dataset = pd.read_csv(path)
    f_dataset = f_dataset.dropna()
    f_dataset = f_dataset.reset_index(drop=True)
    f_keys = list(f_dataset.keys())
    f_index = f_keys.index("Arithmancy")
    f_keys = f_keys[f_index:]
    f_features = {}
    for key in f_keys:
        f_datalist = list(f_dataset[key])
        f_features[key] = {}
        f_features[key]["count"] = ft_count(f_datalist)  # number of values
        f_features[key]["mean"] = ft_mean(f_datalist)   # mean
        f_features[key]["mode"] = ft_mode(f_datalist)   # mode
        f_features[key]["var"] = ft_var(f_datalist)     # variance
        f_features[key]["std"] = ft_std(f_datalist)     # standard deviation
        f_features[key]["min"] = ft_min(f_datalist)     # minimum value
        f_features[key]["first"] = ft_percentile(f_datalist, 25)    # first quartile
        f_features[key]["second"] = ft_percentile(f_datalist, 50)   # seconde quartile
        f_features[key]["third"] = ft_percentile(f_datalist, 75)    # third quartile
        f_features[key]["max"] = ft_max(f_datalist)     # maximum value
        f_features[key]["skew"] = ft_skewness(f_datalist)   # skewness

    if verbose:
        print("{:16}".format(""), end="")
        for feature in f_features:
            f_txt = feature if len(feature) <= 16 else feature[:13] + "..."
            print(f"{f_txt:>16}  ", end="")
        print()
        f_vals = ["count",
                  "mean",
                  "mode",
                  "var",
                  "std",
                  "min",
                  "first",
                  "second",
                  "third",
                  "max",
                  "skew"
                  ]
        for key in f_vals:
            print(f"{key: <16}", end="")
            for feature in f_features.values():
                if feature[key] is not None:
                    print(f"{feature[key]: >16.4f}  ", end="")
                else:
                    print(f"            None  ", end="")
            print()

    return f_features


if __name__ == '__main__':
    print(sys.executable)
    g_args = sys.argv
    if len(g_args) != 2:
        print("Error: program takes 1 argument, the path of the dataset")
        exit()
    g_path = g_args[1]
    describe(g_path, verbose=True)
