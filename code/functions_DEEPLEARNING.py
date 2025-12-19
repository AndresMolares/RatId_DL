import math
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import numpy as np
import os
from collections import Counter

def get_dataset_size(dataset):
    count = dataset.reduce(0, lambda x, _: x + 1)
    return count.numpy()

def _train_model(type_model, imageShape, classes, train_dataset, val_dataset, n_imagenes_x_class_tr_val, params, dataAugmentation=None):

	# Definir data augmentation
	data_augmentation = None
	if dataAugmentation == 'FLIP':
		data_augmentation = tf.keras.Sequential([
			tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal_and_vertical'),
		])
	if dataAugmentation == 'ROTA':
		data_augmentation = tf.keras.Sequential([
			tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
		])
	if dataAugmentation == 'ZOOM':
		data_augmentation = tf.keras.Sequential([
			tf.keras.layers.experimental.preprocessing.RandomZoom(0.1),
		])
	if dataAugmentation == 'MIX':
		data_augmentation = tf.keras.Sequential([
			tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal_and_vertical'),
			tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
			tf.keras.layers.experimental.preprocessing.RandomZoom(0.1),
		])

	# Diccionario para seleccionar el modelo base y su preprocesamiento
	model_dict = {
		'Xception': (tf.keras.applications.Xception, tf.keras.applications.xception.preprocess_input),
		'ResNet50': (tf.keras.applications.ResNet50, tf.keras.applications.resnet.preprocess_input),
		'InceptionV3': (tf.keras.applications.InceptionV3, tf.keras.applications.inception_v3.preprocess_input),
		'MobileNetV2': (tf.keras.applications.MobileNetV2, tf.keras.applications.mobilenet_v2.preprocess_input)
	}

	if type_model not in model_dict:
		raise ValueError(f"Unsupported model type: {type_model}")

	base_model_class, preprocess_input = model_dict[type_model]
	base_model = base_model_class(weights='imagenet', input_shape=(imageShape[0], imageShape[1], 3), include_top=False)
	base_model.trainable = False

	def create_model(num_classes, params):
		def preprocess_and_extract_features(inputs, base_model):
			# Consistently resize images to expected shape
			x = tf.keras.layers.Lambda(lambda img: tf.image.resize(img, (imageShape[0], imageShape[1])))(inputs)
			x = preprocess_input(x)
			features = base_model(x, training=False)
			return features

		inputs = tf.keras.Input(shape=(128, 128, 3))

		if dataAugmentation:
			x = data_augmentation(inputs)
		else:
			x = inputs

		# Apply data augmentation on-the-fly within the training loop
		processed_features = preprocess_and_extract_features(x, base_model)

		x = tf.keras.layers.GlobalAveragePooling2D()(processed_features)
		x = tf.keras.layers.Dropout(0.2)(x)
		x = tf.keras.layers.Dense(params[1][0])(x)
		if len(params[1]) == 2:
			x = tf.keras.layers.Dense(params[1][1])(x)
		outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
		model = tf.keras.Model(inputs, outputs)
		return model

	model = create_model(len(classes), params)
	model.compile(
		optimizer=tf.keras.optimizers.Adam(params[0]),
		loss=tf.keras.losses.SparseCategoricalCrossentropy(),
		metrics=['accuracy']
	)

	def potencia_de_dos(a, b, c):
		suma = a + b + c
		exponente = math.ceil(math.log2(suma))
		resultado = 2 ** exponente
		return  resultado

	batch_size = 2048

	# Preparar datasets
	train_dataset = train_dataset.shuffle((len(classes) * n_imagenes_x_class_tr_val * 7), seed=42, reshuffle_each_iteration=True)
	train_dataset = train_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
	val_dataset = val_dataset.batch(batch_size).cache().prefetch(tf.data.AUTOTUNE)

	# Definir callback de early stopping
	callback = tf.keras.callbacks.EarlyStopping(min_delta=0.001, monitor='val_loss', patience=5, restore_best_weights=True)

	# Entrenar el modelo
	history = model.fit(
		train_dataset,
		epochs=1000,
		verbose=1,
		callbacks=[callback],
		validation_data=val_dataset,
		shuffle=True,
		validation_freq=1
	)

	base_model.trainable = True

	fine_tune_at = round(len(base_model.layers) * 0.1)
	for layer in base_model.layers[:(len(base_model.layers) - fine_tune_at)]:
		layer.trainable = False

	model.compile(
		optimizer=tf.keras.optimizers.Adam(params[0]*0.1),
		loss=tf.keras.losses.SparseCategoricalCrossentropy(),
		metrics=['accuracy']
	)

	callback2 = tf.keras.callbacks.EarlyStopping(min_delta=0.001, monitor='val_loss', patience=3, restore_best_weights=True)

	model.fit(
		train_dataset,
		verbose=1,
		callbacks=[callback2],
		validation_data=val_dataset,
		shuffle=True,
		validation_freq=1,
		epochs=1100,  # number of training epochs
		initial_epoch=history.epoch[-1]
	)

	return model
def train_model(ratas_selected, dict_videos, directory, video_test, video_validacion, n_imagenes_x_class_tr_val, n_imagenes_x_class_te, imageShape, tipo_modelo, params, dataAugmentation=None, color=True, frames=None):
	def parse_record(record, label):
		feature_description = {
			'image_raw': tf.io.FixedLenFeature([], tf.string),
		}
		example = tf.io.parse_single_example(record, feature_description)
		image = tf.image.decode_image(example['image_raw'], channels=3 if color else 1)
		if not color:
			image = tf.image.grayscale_to_rgb(image)
		image = tf.cast(image, tf.uint8)
		label = tf.constant(label, tf.uint8)
		return image, label

	def filter_elements(idx, positions_to_select):
		return tf.reduce_any(tf.equal(idx, positions_to_select))

	def create_datasets(filter_elements):

		idx_to_select_tr_val = [i for i in range(0, 10000, int(10000 / n_imagenes_x_class_tr_val))]
		if frames:
			idx_to_select_te = [i for i in range(0, 10000)]
		else:
			idx_to_select_te = [i for i in range(0, 10000, int(10000 / n_imagenes_x_class_te))]

		training_dataset, test_dataset, val_dataset = None, None, None

		for idx_rata in ratas_selected:
			for video in dict_videos:
				tfrecord_file = os.path.join(directory, video, os.listdir(os.path.join(directory, video))[idx_rata])

				dataset_aux = tf.data.TFRecordDataset(tfrecord_file)
				dataset_aux = dataset_aux.enumerate().filter(lambda x, _: filter_elements(x,
																						  idx_to_select_tr_val if video != video_test else
																						  idx_to_select_te))
				dataset_aux = dataset_aux.map(lambda _, x: parse_record(x, ratas_selected.index(idx_rata)))
				if video != video_test and video != video_validacion:
					training_dataset = dataset_aux if training_dataset is None else training_dataset.concatenate(
						dataset_aux)
				elif video == video_test:
					test_dataset = dataset_aux if test_dataset is None else test_dataset.concatenate(dataset_aux)
				else:
					val_dataset = dataset_aux if val_dataset is None else val_dataset.concatenate(dataset_aux)

		return training_dataset, test_dataset, val_dataset

	training_dataset, test_dataset, val_dataset = create_datasets(filter_elements)

	model = _train_model(tipo_modelo, imageShape, ratas_selected, training_dataset, val_dataset, n_imagenes_x_class_tr_val, params, dataAugmentation)

	return model, test_dataset

def test_model(model, test_dataset, frames=None):
	# Convertir el dataset de prueba en matrices NumPy
	x_test, y_test = zip(*[(x.numpy(), y.numpy()) for x, y in test_dataset])
	x_test = np.array(x_test)
	y_test = np.array(y_test)

	# Obtener las predicciones del modelo
	if frames:
		predictions = []
		reals = []
		for i in range(0, len(x_test), 10000):
			block = x_test[i:i + 10000]
			for j in range(0, 10000 - frames + 1, frames):
				sequence = block[j:j + frames]
				# Hacer predicciones para la secuencia
				sequence_preds = model.predict(sequence)
				# Convertir las predicciones en clases (hard voting)
				sequence_classes = np.argmax(sequence_preds, axis=1)
				# Contar las ocurrencias de cada clase y elegir la clase más común
				most_common_class = Counter(sequence_classes).most_common(1)[0][0]
				predictions.append(most_common_class)
				reals.append(y_test[i + j])

		y_pred = np.array(predictions)
		y_test2 = np.array(reals)
		results_score = accuracy_score(y_test2, y_pred)
		cm = confusion_matrix(y_test2, y_pred)
	else:
		y_pred = np.argmax(model.predict(x_test), axis=1)

		# Calcular la matriz de confusión y la precisión
		cm = confusion_matrix(y_test, y_pred)
		results_score = accuracy_score(y_test, y_pred)

	return results_score, cm