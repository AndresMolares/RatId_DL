import tensorflow as tf
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from scipy import ndimage as nd
from skimage.filters import gabor_kernel
from sklearn import svm
import numpy as np
import os
from skimage.feature import graycomatrix, graycoprops
from sklearn.decomposition import PCA

# GLCM properties
def _contrast_feature(matrix_coocurrence):
    contrast = graycoprops(matrix_coocurrence, 'contrast')
    return contrast

def _dissimilarity_feature(matrix_coocurrence):
    dissimilarity = graycoprops(matrix_coocurrence, 'dissimilarity')
    return dissimilarity

def _homogeneity_feature(matrix_coocurrence):
    homogeneity = graycoprops(matrix_coocurrence, 'homogeneity')
    return homogeneity

def _energy_feature(matrix_coocurrence):
    energy = graycoprops(matrix_coocurrence, 'energy')
    return energy

def _correlation_feature(matrix_coocurrence):
    correlation = graycoprops(matrix_coocurrence, 'correlation')
    return correlation

def _compute_feats(image, kernels):

	feats = np.zeros((len(kernels) * 15), dtype=np.double)
	for k, kernel in enumerate(kernels):
		for chanel in range(3):
			filtered = nd.convolve(image[:, :, chanel], kernel, mode='wrap')
			bins32 = np.array(
				[0, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 136, 144, 152, 160,
				 168, 176, 184, 192, 200, 208, 216, 224, 232, 240, 248, 255])  # 32-bit
			np.clip(filtered, 0, 255, out=filtered)
			image_aux = filtered.astype('uint8')
			inds = np.digitize(image_aux, bins32)
			max_value = inds.max() + 1
			matrix_coocurrence = graycomatrix(inds, [1], [0], levels=max_value, normed=False, symmetric=False)
			feats[(k * 15) + (chanel * 5 + 0)] = _contrast_feature(matrix_coocurrence)
			feats[(k * 15) + (chanel * 5 + 1)] = _dissimilarity_feature(matrix_coocurrence)
			feats[(k * 15) + (chanel * 5 + 2)] = _homogeneity_feature(matrix_coocurrence)
			feats[(k * 15) + (chanel * 5 + 3)] = _energy_feature(matrix_coocurrence)
			feats[(k * 15) + (chanel * 5 + 4)] = _correlation_feature(matrix_coocurrence)

	return feats

def _train_model(model, type_model, imageShape, classes, training_dataset, params, dataAugmentation=False):
	if type_model == 'Gabor':

		x_data = []
		y_data = []
		for x, y in training_dataset:
			x_data.append(x.numpy())  # Convertir el tensor a una matriz NumPy y agregarlo a la lista de características
			y_data.append(y.numpy())  # Convertir el tensor a una matriz NumPy y agregarlo a la lista de etiquetas

		# Convertir las listas de características y etiquetas en matrices NumPy
		x_data = np.array(x_data)
		y_data = np.array(y_data)

		kernels = []
		for theta in range(4):
			theta = theta / 4. * np.pi
			for sigma, frequency in [[2.24, 0.25], [4.48, 0.125]]:
				kernel = np.real(gabor_kernel(frequency, theta=theta, sigma_x=sigma, sigma_y=sigma))
				kernels.append(kernel)

		x_train_gabor = []
		for f1 in x_data:
			x_train_gabor.append(_compute_feats(f1, kernels))
		pca = PCA(n_components=0.999, svd_solver='full').fit(x_train_gabor, y_data)

		n_componentes = pca.n_components_
		print("Número de componentes seleccionados:", n_componentes)

		x_train_gabor = pca.transform(x_train_gabor)

		if not model:
			model = svm.SVC(C=params[1], kernel=params[0], decision_function_shape='ovr')
		model.fit(x_train_gabor, y_data)

	return model, pca

def _filas(index, rango):
	if index in rango:
		return False
	return True

def train_model_tfrecords(ratas_selected, dict_videos, directory, video_test, n_imagenes_x_class_tr, n_imagenes_x_class_te, imageShape, tipo_modelo, params):
	def parse_record(record, label):
		feature_description = {
			'image_raw': tf.io.FixedLenFeature([], tf.string),
		}
		example = tf.io.parse_single_example(record, feature_description)
		image = tf.image.decode_image(example['image_raw'], channels=3)  # Decodificar la imagen
		image = tf.cast(image, tf.uint8)
		label = tf.constant(label, tf.uint8)
		return image, label

	def filter_elements(idx, positions_to_select):
		return tf.reduce_any(tf.equal(idx, positions_to_select))

	positions_to_select_tr = [i for i in range(0, 10000, int(10000 / n_imagenes_x_class_tr))]
	positions_to_select_te = [i for i in range(0, 10000, int(10000 / n_imagenes_x_class_te))]
	model = None

	training_dataset = None
	test_dataset = None
	for idx_rata in ratas_selected:
		for video in dict_videos:
			directorio_ratas = os.listdir(directory + video + '/')
			directorio_ratas.sort()
			if video != video_test:

				tfrecord_file = directory + video + '/' + directorio_ratas[idx_rata]
				training_dataset_aux = tf.data.TFRecordDataset(tfrecord_file)
				training_dataset_aux = training_dataset_aux.enumerate().filter(lambda x, _: filter_elements(x, positions_to_select_tr))
				training_dataset_aux = training_dataset_aux.map(lambda _, x: parse_record(x, ratas_selected.index(idx_rata)))

				if training_dataset == None:
					training_dataset = training_dataset_aux
				else:
					training_dataset = training_dataset.concatenate(training_dataset_aux)

			else:
				tfrecord_file = directory + video + '/' + directorio_ratas[idx_rata]
				test_dataset_aux = tf.data.TFRecordDataset(tfrecord_file)
				test_dataset_aux = test_dataset_aux.enumerate().filter(
					lambda x, _: filter_elements(x, positions_to_select_te))
				test_dataset_aux = test_dataset_aux.map(
					lambda _, x: parse_record(x, ratas_selected.index(idx_rata)))

				if test_dataset == None:
					test_dataset = test_dataset_aux
				else:
					test_dataset = test_dataset.concatenate(test_dataset_aux)

	training_dataset = training_dataset.shuffle(buffer_size=n_imagenes_x_class_tr, reshuffle_each_iteration=False)
	model, pca = _train_model(model, tipo_modelo, imageShape, ratas_selected, training_dataset, params)

	return model, pca, test_dataset

def test_model(model, pca, type_model, test_dataset):

	x_test = []
	y_test = []
	for x, y in test_dataset:
		x_test.append(x.numpy())  # Convertir el tensor a una matriz NumPy y agregarlo a la lista de características
		y_test.append(y.numpy())  # Convertir el tensor a una matriz NumPy y agregarlo a la lista de etiquetas

	# Convertir las listas de características y etiquetas en matrices NumPy
	x_test = np.array(x_test)
	y_test = np.array(y_test)

	if type_model == 'Gabor':
		kernels = []
		for theta in range(4):
			theta = theta / 4. * np.pi
			for sigma, frequency in [[2.24, 0.25], [4.48, 0.125]]:
				kernel = np.real(gabor_kernel(frequency, theta=theta, sigma_x=sigma, sigma_y=sigma))
				kernels.append(kernel)

		x_test_gabor = []
		for f1 in x_test:
			x_test_gabor.append(_compute_feats(f1, kernels))

		x_test_gabor = pca.transform(x_test_gabor)

		y_pred = model.predict(x_test_gabor)
		cm = confusion_matrix(y_test, y_pred)
		results_score = accuracy_score(y_test, y_pred)

	return results_score, cm
