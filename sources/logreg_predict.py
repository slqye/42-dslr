import sys
import math
import pandas as pd

HOUSES = ["Gryffindor", "Slytherin", "Ravenclaw", "Hufflepuff"]
FEATURES = ["Herbology", "Astronomy", "Ancient Runes", "Charms"]
LABEL = "Hogwarts House"

def retrieve_models(training_file: str) -> list[list[float]]:
	"""
	Load logistic regression model weights from file.
	"""
	try:
		with open(training_file, "r", encoding="utf-8") as file:
			lines: list[str] = file.readlines()
			wn: int = len(FEATURES) + 1
			models: list[list[float]] = []
			for i in range(len(FEATURES)):
				models.append(lines[i * wn:(i + 1) * wn])
			for i in range(len(models)):
				models[i] = [float(x) for x in models[i]]
			return models
	except Exception as error:
		print("Error:", error, file=sys.stderr)
		sys.exit(1)

def format_df(dataset: pd.DataFrame, features: list[str], label: str) -> pd.DataFrame:
	"""
	Formats the dataset by selecting relevant columns and filling missing values.
	"""
	columns: list[str] = features + [label]
	dataset = dataset[columns]
	dataset = dataset.fillna(0)
	return dataset

def predict(models: list[list[float]], dataset: pd.DataFrame) -> list[tuple[int, str]]:
	"""
	Predict Hogwarts houses for all students.
	"""
	dataset = dataset.copy()
	result: list[tuple[int, str]] = []
	rows = list(dataset.itertuples(index=False))
	models_predictions: list[float] = []

	for index, student in enumerate(rows):
		models_predictions = []
		for model in models:
			z: float = sum(model[i] * student[i] for i in range(len(FEATURES))) + model[-1]
			models_predictions.append(1 / (1 + math.exp(-z)))
		house_index = models_predictions.index(max(models_predictions))
		result.append((index, HOUSES[house_index]))
	return result

def save_predictions(predictions: list[tuple[int, str]]) -> None:
	"""
	Save predictions to houses.csv.
	"""
	with open("houses.csv", "w", encoding="utf-8") as file:
		file.write("Index,Hogwarts House\n")
		for i in predictions:
			file.write(f"{str(i[0])},{i[1]}\n")

def main(argv: list[str]) -> None:
	"""
	Run the prediction pipeline.
	"""
	try:
		if len(argv) != 3:
			raise ValueError("Usage: logreg_predict.py [training_file] [dataset]")
		models: list[list[float]] = retrieve_models(argv[1])
		dataset: pd.DataFrame = pd.read_csv(argv[2])
		dataset = format_df(dataset, FEATURES, LABEL)
		predictions: list[tuple[int, str]] = predict(models, dataset)
		save_predictions(predictions)
	except Exception as error:
		print("Error:", error, file=sys.stderr)
		sys.exit(1)


if __name__ == "__main__":
	main(sys.argv)
