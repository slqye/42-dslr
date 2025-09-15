import sys
import math
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn as sns
from tqdm import tqdm

HOUSES = ["Gryffindor", "Slytherin", "Ravenclaw", "Hufflepuff"]
FEATURES = ["Herbology", "Astronomy", "Ancient Runes", "Charms"]
LABEL = "Hogwarts House"

class Model:
	"""
	Logistic regression model for binary classification using gradient descent variants.
	"""

	def __init__(
		self,
		dataset: pd.DataFrame,
		features: list[str],
		label: str,
		target: str,
		to_loss: bool = False
	) -> None:
		"""
		Initializes the model with dataset, features, label, target, and loss tracking option.
		"""
		self._dataset: pd.DataFrame = dataset
		self._features: list[str] = features
		self._label: str = label
		self._target: str = target
		self._weights: list[float] = [0.0] * (len(self._features) + 1)
		self._to_loss = to_loss
		self._losses: list[float] = []

	def compute_loss(self) -> float:
		"""
		Computes the logistic loss (cross-entropy) for the current model weights.
		"""
		rows: list[tuple] = list(self._dataset.itertuples(index=False))
		sigma: float = 0
		for i in rows:
			yi = i[-1]
			z = sum(self._weights[j] * i[j] for j in range(len(self._features))) + self._weights[-1]
			sigma += yi * math.log(1 / (1 + math.exp(-z))) + (1 - yi) * math.log(1 - (1 / (1 + math.exp(-z))))
		return -(sigma / self._dataset.shape[0])

	def train_gd(
		self,
		epochs: int = 1000,
		learning_rate: float = 1,
		minimum_step: float = 0.1
	) -> None:
		"""
		Trains the model using gradient descent.
		"""
		self._weights = [0.0] * (len(self._features) + 1)
		self._losses = []
		self._dataset[self._label] = self._dataset[self._label].apply(lambda x: int(x == self._target))
		features_length: int = len(self._features)
		students: list[tuple] = list(self._dataset.itertuples(index=False))

		for epoch in tqdm(range(epochs)):
			gradient: list[float] = [0.0] * (features_length + 1)
			for student in students:
				z: float = np.dot(self._weights[:-1], [student[i] for i in range(features_length)]) + self._weights[-1]
				predicted_value: float = 1 / (1 + math.exp(-z))
				actual_value: int = student[-1]
				for j in range(features_length):
					gradient[j] += (predicted_value - actual_value) * student[j]
				gradient[-1] += (predicted_value - actual_value) * 1
			for j in range(features_length + 1):
				self._weights[j] -= learning_rate / self._dataset.shape[0] * gradient[j]
			if np.linalg.norm(gradient) < minimum_step:
				break
			if (self._to_loss):
				self._losses.append(self.compute_loss())

	def train_sgd(
		self,
		epochs: int = 1000,
		learning_rate: float = 1
	) -> None:
		"""
		Trains the model using stochastic gradient descent.
		"""
		self._weights = [0.0] * (len(self._features) + 1)
		self._losses = []
		self._dataset[self._label] = self._dataset[self._label].apply(lambda x: int(x == self._target))
		features_length: int = len(self._features)
		students: list[tuple] = list(self._dataset.itertuples(index=False))

		for epoch in tqdm(range(epochs)):
			gradient: list[float] = [0.0] * (features_length + 1)
			student: tuple = random.choice(students)
			z: float = np.dot(self._weights[:-1], [student[i] for i in range(features_length)]) + self._weights[-1]
			predicted_value: float = 1 / (1 + math.exp(-z))
			actual_value: int = student[-1]
			for j in range(features_length):
				gradient[j] += (predicted_value - actual_value) * student[j]
			gradient[-1] += (predicted_value - actual_value) * 1
			for j in range(features_length + 1):
				self._weights[j] -= learning_rate * gradient[j]
			if (self._to_loss):
				self._losses.append(self.compute_loss())

	def train_mbgd(
		self,
		epochs: int = 1000,
		learning_rate: float = 1,
		batch: int = 10
	) -> None:
		"""
		Trains the model using mini-batch gradient descent.
		"""
		self._weights = [0.0] * (len(self._features) + 1)
		self._losses = []
		self._dataset[self._label] = self._dataset[self._label].apply(lambda x: int(x == self._target))
		features_length: int = len(self._features)
		students: list[tuple] = list(self._dataset.itertuples(index=False))

		for epoch in tqdm(range(epochs)):
			gradient: list[float] = [0.0] * (features_length + 1)
			student_batch: tuple = random.sample(students, k=batch)
			for student in student_batch:
				z: float = np.dot(self._weights[:-1], [student[i] for i in range(features_length)]) + self._weights[-1]
				predicted_value: float = 1 / (1 + math.exp(-z))
				actual_value: int = student[-1]
				for j in range(features_length):
					gradient[j] += (predicted_value - actual_value) * student[j]
				gradient[-1] += (predicted_value - actual_value) * 1
			for j in range(features_length + 1):
				self._weights[j] -= learning_rate / batch * gradient[j]
			if (self._to_loss):
				self._losses.append(self.compute_loss())

	def unormalize(self, dataset_max: pd.DataFrame, dataset_min: pd.DataFrame) -> None:
		"""
		Reverse the effect of the normalization on the weight of the models.
		"""
		alpha: list[float] = list(1 / (dataset_max - dataset_min))
		beta: list[float] = list(-(dataset_min / (dataset_max - dataset_min)))

		for index, weight in enumerate(self._weights[:-1]):
			self._weights[-1] += weight * beta[index]
		for index, weight in enumerate(self._weights[:-1]):
			self._weights[index] = weight * alpha[index]

def format_df(dataset: pd.DataFrame, features: list[str], label: str) -> pd.DataFrame:
	"""
	Formats the dataset by selecting relevant columns and filling missing values.
	"""
	columns: list[str] = features + [label]
	dataset = dataset[columns]
	dataset = dataset.fillna(0)
	return dataset

def normalize_df(dataset: pd.DataFrame, features: list[str]) -> pd.DataFrame:
	"""
	Normalizes feature columns to the range [0, 1].
	"""
	h_features: pd.DataFrame = dataset[features]
	dataset[features] = (h_features - h_features.min()) / (h_features.max() - h_features.min())
	return dataset

def save_models(models: list[Model]) -> None:
	"""
	Saves the trained model weights to 'training.out'.
	"""
	with open("training.out", "w", encoding="utf-8") as file:
		for model in models:
			for weight in model._weights:
				file.write(str(weight) + "\n")

def loss_visualization(dataset: pd.DataFrame, house: str) -> None:
	"""
	Animates the loss curves for GD, SGD, and BGD during training.
	"""
	epochs: int = 1000
	losses: list[list[float]] = []
	gd_names: list[str] = ["GD", "SGD", "MBGD"]
	m1: Model = Model(dataset.copy(), FEATURES, LABEL, house, to_loss=True)
	m2: Model = Model(dataset.copy(), FEATURES, LABEL, house, to_loss=True)
	m3: Model = Model(dataset.copy(), FEATURES, LABEL, house, to_loss=True)
	m1.train_gd(epochs=epochs, minimum_step=0)
	m2.train_sgd(epochs=epochs)
	m3.train_mbgd(epochs=epochs)
	losses.append(m1._losses)
	losses.append(m2._losses)
	losses.append(m3._losses)
	fig, ax = plt.subplots()
	ax.set_ylim(0, max(max(loss) for loss in losses) * 1.01)
	ax.set_xlabel("Epoch")
	ax.set_ylabel("Loss")
	ax.set_title(f"Models Loss Over Epochs ({house})")
	lines = [ax.plot([], [], label=gd_names[i])[0] for i in range(len(losses))]
	ax.legend()

	def update(frame):
		for i, line in enumerate(lines):
			line.set_data(range(frame + 1), losses[i][:frame + 1])
		ax.set_xlim(0, (frame + 1) * 1.01)
		ax.set_xticks(range(0, frame + 1, int(epochs / 5)))
		return lines
	ani = animation.FuncAnimation(fig, update, frames=range(0, epochs, 2), interval=10, blit=False, repeat=False)
	plt.show()
	plt.close()

def main(argv: list[str]) -> None:
	"""
	Run the training pipeline.
	"""
	try:
		if len(argv) != 2:
			raise ValueError("wrong number of arguments.")
		dataset: pd.DataFrame = pd.read_csv(argv[1])
		dataset = format_df(dataset, FEATURES, LABEL)
		dataset_max: pd.DataFrame = dataset[FEATURES].max()
		dataset_min: pd.DataFrame = dataset[FEATURES].min()
		dataset = normalize_df(dataset, FEATURES)
		models: list[Model] = []
		for house in HOUSES:
			print("training:", house, file=sys.stdout)
			model: Model = Model(dataset.copy(), FEATURES, LABEL, house)
			model.train_gd(epochs=5000)
			model.unormalize(dataset_max, dataset_min)
			models.append(model)
		print("training: complete", file=sys.stdout)
		save_models(models)
		print("visualization: computing", file=sys.stdout)
		loss_visualization(dataset, HOUSES[0])
		print("visualization: complete", file=sys.stdout)
	except Exception as error:
		print("error:", error, file=sys.stderr)

if __name__ == '__main__':
	main(sys.argv)
