from load_data import load_data_normalized


def main():
	# Étape 2 — Préparer les données: normalisation + vérification min/max
	(x_train, y_train), (x_test, y_test) = load_data_normalized()

	# Vérifications additionnelles (types et plages)
	print("Types:", x_train.dtype, x_test.dtype)
	print("Train min/max:", x_train.min(), x_train.max())
	print("Test  min/max:", x_test.min(), x_test.max())
	print("Tailles:", x_train.shape, x_test.shape)


if __name__ == "__main__":
	main()
