# DSLR

Desc of the project...

## Usage

> [!NOTE]
> To run the program from source, you need to have `uv` installed on your system.
> https://docs.astral.sh/uv/getting-started/installation/

To see the dataset analysis
```bash
uv run sources/describe.py datasets/[.csv]
uv run sources/histogram.py datasets/[.csv]
uv run sources/pair_plot.py datasets/[.csv]
```

To train the model
```bash
uv run sources/logreg_train.py
```

To make predictions
```bash
uv run sources/logreg_predict.py
```

## Explanation

## Result

![Visualization](assets/visualization.gif)
