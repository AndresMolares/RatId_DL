import tensorflow as tf
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import numpy as np
import os
from collections import Counter
import csv


def create_datasets(ratas_selected, directory, n_imagenes_x_class_tr_val, n_imagenes_x_class_te, video_length: int = 10000, frames=None):
	def _parse_record(record, label):
		feature_description = {
			'image_raw': tf.io.FixedLenFeature([], tf.string),
		}
		example = tf.io.parse_single_example(record, feature_description)
		image = tf.image.decode_image(example['image_raw'], channels=3)
		image = tf.cast(image, tf.uint8)
		label = tf.constant(label, tf.uint8)
		return image, label
	def _filter_elements(idx, positions_to_select):
		return tf.reduce_any(tf.equal(idx, positions_to_select))

	# Índices de imágenes a usar en train/val y test
	idx_to_select_tr_val = [i for i in range(0, video_length, int(video_length / n_imagenes_x_class_tr_val))]
	if frames:
		idx_to_select_te = [i for i in range(0, video_length)]
	else:
		idx_to_select_te = [i for i in range(0, video_length, int(video_length / n_imagenes_x_class_te))]

	training_dataset, test_dataset, val_dataset = None, None, None

	# Rutas base
	directory = os.path.join(directory, "/tfRecords/")
	train_dir = os.path.join(directory, "train")
	test_dir = os.path.join(directory, "test")

	# Listado de vídeos en train y test
	train_videos = sorted(os.listdir(train_dir))
	test_videos = sorted(os.listdir(test_dir))

	# Elegimos UN vídeo de train como validación (por ejemplo, el último)
	video_validacion = train_videos[-1]
	train_videos_wo_val = [v for v in train_videos if v != video_validacion]

	# ------------------------
	#   TRAINING + VALIDACIÓN
	# ------------------------
	for idx_rata in ratas_selected:
		# Vídeos de entrenamiento (train sin el de validación)
		for video in train_videos_wo_val:
			# tfrecord correspondiente a esa rata en ese vídeo
			tfrecord_file = os.path.join(
				train_dir,
				video,
				os.listdir(os.path.join(train_dir, video))[idx_rata]
			)

			dataset_aux = tf.data.TFRecordDataset(tfrecord_file)
			# Para train/val usamos los índices de tr_val
			dataset_aux = dataset_aux.enumerate().filter(
				lambda x, _: _filter_elements(x, idx_to_select_tr_val)
			)
			dataset_aux = dataset_aux.map(
				lambda _, x: _parse_record(x, ratas_selected.index(idx_rata))
			)

			training_dataset = dataset_aux if training_dataset is None \
				else training_dataset.concatenate(dataset_aux)

		# Vídeo de validación (un vídeo de train)
		video = video_validacion
		tfrecord_file = os.path.join(
			train_dir,
			video,
			os.listdir(os.path.join(train_dir, video))[idx_rata]
		)

		dataset_aux = tf.data.TFRecordDataset(tfrecord_file)
		dataset_aux = dataset_aux.enumerate().filter(
			lambda x, _: _filter_elements(x, idx_to_select_tr_val)
		)
		dataset_aux = dataset_aux.map(
			lambda _, x: _parse_record(x, ratas_selected.index(idx_rata))
		)

		val_dataset = dataset_aux if val_dataset is None \
			else val_dataset.concatenate(dataset_aux)

	# -------------
	#      TEST
	# -------------
	for idx_rata in ratas_selected:
		for video in test_videos:
			tfrecord_file = os.path.join(
				test_dir,
				video,
				os.listdir(os.path.join(test_dir, video))[idx_rata]
			)

			dataset_aux = tf.data.TFRecordDataset(tfrecord_file)
			# Para test usamos los índices de test
			dataset_aux = dataset_aux.enumerate().filter(
				lambda x, _: _filter_elements(x, idx_to_select_te)
			)
			dataset_aux = dataset_aux.map(
				lambda _, x: _parse_record(x, ratas_selected.index(idx_rata))
			)

			test_dataset = dataset_aux if test_dataset is None \
				else test_dataset.concatenate(dataset_aux)

	return training_dataset, test_dataset, val_dataset

def train_model(type_model, imageShape, ratas_selected, training_dataset, val_dataset, n_imagenes_x_class_tr_val, params, dataAugmentation=None):
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

	model = create_model(len(ratas_selected), params)
	model.compile(
		optimizer=tf.keras.optimizers.Adam(params[0]),
		loss=tf.keras.losses.SparseCategoricalCrossentropy(),
		metrics=['accuracy']
	)

	batch_size = 2048

	# Preparar datasets
	train_dataset = training_dataset.shuffle((len(ratas_selected) * n_imagenes_x_class_tr_val * 7), seed=42,
										  reshuffle_each_iteration=True)
	train_dataset = train_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
	val_dataset = val_dataset.batch(batch_size).cache().prefetch(tf.data.AUTOTUNE)

	# Definir callback de early stopping
	callback = tf.keras.callbacks.EarlyStopping(min_delta=0.001, monitor='val_loss', patience=25,
												restore_best_weights=True)
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
		optimizer=tf.keras.optimizers.Adam(params[0] * 0.1),
		loss=tf.keras.losses.SparseCategoricalCrossentropy(),
		metrics=['accuracy']
	)

	callback2 = tf.keras.callbacks.EarlyStopping(min_delta=0.001, monitor='val_loss', patience=3,
												 restore_best_weights=True)

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

def test_model(model,
               test_dataset,
               frames: int | None = None,
               video_length: int = 10000,
               batch_size: int = 256,
               save_mode: str = "metrics",           # <-- nuevo
               output_file: str | None = None):      # <-- nuevo
    """
    Evalúa un modelo sobre un dataset de test (no batcheado).

    save_mode:
        - "metrics": guarda accuracy y matriz de confusión.
        - "predictions": guarda por línea (sample_id, predicted_class).
    """

    # 1) Convertir dataset a NumPy
    x_list, y_list = [], []
    for x, y in test_dataset:
        x_list.append(x.numpy())
        y_list.append(y.numpy())

    x_test = np.stack(x_list, axis=0)
    y_test = np.array(y_list)

    # ============================================================
    # MODO SIMPLE (frames=None) → predicción imagen a imagen
    # ============================================================
    if not frames or frames <= 1:

        preds = model.predict(x_test, batch_size=batch_size, verbose=0)
        y_pred = np.argmax(preds, axis=1)

        # Guardar predicciones si pide modo "predictions"
        if save_mode == "predictions" and output_file is not None:
            with open(output_file, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["sample_id", "predicted_class"])
                for idx, cls in enumerate(y_pred):
                    writer.writerow([idx, int(cls)])
            print(f"Predicciones guardadas en: {output_file}")

        # Calcular métricas
        cm = confusion_matrix(y_test, y_pred)
        acc = accuracy_score(y_test, y_pred)

        # Guardar métricas si save_mode=metrics
        if save_mode == "metrics" and output_file is not None:
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(f"Accuracy: {acc}\n")
                f.write(f"Confusion matrix:\n{cm}\n")
            print(f"Métricas guardadas en: {output_file}")

        return acc, cm

    # ============================================================
    # MODO SECUENCIAL (frames>1)
    # ============================================================
    num_samples = len(x_test)

    sequences = []
    start_indices = []
    for start_block in range(0, num_samples, video_length):
        end_block = min(start_block + video_length, num_samples)
        block_x = x_test[start_block:end_block]
        block_len = len(block_x)

        if block_len < frames:
            continue

        for j in range(0, block_len - frames + 1, frames):
            seq_start = start_block + j
            seq_end = seq_start + frames
            sequences.append(x_test[seq_start:seq_end])
            start_indices.append(seq_start)

    sequences = np.stack(sequences, axis=0)
    num_seqs, f, H, W, C = sequences.shape
    flat_sequences = sequences.reshape(num_seqs * f, H, W, C)

    # Predicción en batch
    flat_preds = model.predict(flat_sequences, batch_size=batch_size, verbose=0)
    flat_classes = np.argmax(flat_preds, axis=1).reshape(num_seqs, frames)

    # Voto mayoritario secuencia a secuencia
    predictions = []
    reals = []
    for seq_idx in range(num_seqs):
        seq_classes = flat_classes[seq_idx]
        pred = Counter(seq_classes).most_common(1)[0][0]
        predictions.append(pred)

        real_idx = start_indices[seq_idx]
        reals.append(y_test[real_idx])

    y_pred = np.array(predictions)
    y_real = np.array(reals)

    acc = accuracy_score(y_real, y_pred)
    cm = confusion_matrix(y_real, y_pred)

    # Guardar predicciones si así se pide
    if save_mode == "predictions" and output_file is not None:
        with open(output_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["sequence_id", "predicted_class"])
            for idx, cls in enumerate(y_pred):
                writer.writerow([idx, int(cls)])
        print(f"Predicciones guardadas en: {output_file}")

    # Guardar métricas si así se pide
    if save_mode == "metrics" and output_file is not None:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(f"Accuracy: {acc}\n")
            f.write(f"Confusion matrix:\n{cm}\n")
        print(f"Métricas guardadas en: {output_file}")

    return acc, cm



def create_tfrecords(base_path):
    """
    Creates TFRecord files from an image directory structured as:

        base_path/
            Images/
                train/
                    Video01/
                        Rat01/
                            img_0001.jpg
                            img_0002.jpg
                            ...
                        Rat02/
                        ...
                test/
                    Video01/
                    Video02/
                    ...

    The function generates:

        base_path/
            tfRecords/
                train/
                    Video01/
                        Rat01.tfrecord
                        Rat02.tfrecord
                test/
                    Video01/
                    ...

    Parameters
    ----------
    base_path : str
        Root directory containing an 'Images' folder.
    """

    def _bytes_feature(value):
        """Returns a bytes_list feature for a TFRecord."""
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def _serialize_image(image_path):
        """Reads a single image file and converts it into a TF Example."""
        with open(image_path, "rb") as f:
            image_raw = f.read()
        feature = {
            "image_raw": _bytes_feature(image_raw),
        }
        return tf.train.Example(features=tf.train.Features(feature=feature))

    # Input and output folders
    images_path = os.path.join(base_path, "Images")
    tfrecords_path = os.path.join(base_path, "tfRecords")

    dataset_types = ["train", "test"]

    for dtype in dataset_types:

        input_dir = os.path.join(images_path, dtype)
        output_dir = os.path.join(tfrecords_path, dtype)

        print(f"\nReading from:      {input_dir}")
        print(f"Writing TFRecords: {output_dir}")

        # Ensure that the output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # List all videos in the dataset type
        videos = sorted(os.listdir(input_dir))
        print(f"Processing {dtype} videos:", videos)

        for video in videos:
            video_input_path = os.path.join(input_dir, video)
            if not os.path.isdir(video_input_path):
                continue

            # Output folder for this video
            video_output_path = os.path.join(output_dir, video)
            os.makedirs(video_output_path, exist_ok=True)

            rats = sorted(os.listdir(video_input_path))

            for rat in rats:
                rat_input_path = os.path.join(video_input_path, rat)
                if not os.path.isdir(rat_input_path):
                    continue

                images = sorted(os.listdir(rat_input_path))

                tfrecord_name = f"{rat[:4]}.tfrecord"
                tfrecord_path = os.path.join(video_output_path, tfrecord_name)

                print(f"  Creating TFRecord: {tfrecord_path}")

                # Write TFRecord file
                with tf.io.TFRecordWriter(tfrecord_path) as writer:
                    for img_file in images:
                        img_path = os.path.join(rat_input_path, img_file)
                        example = _serialize_image(img_path)
                        writer.write(example.SerializeToString())
