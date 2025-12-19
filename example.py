import functions as fn
import os
import csv
import tensorflow as tf

############ CONFIGURATION PARAMETERS ########################################################

directory = './data'
carpeta = './result'

os.makedirs(carpeta, exist_ok=True)

n_classes = 5
n_imagenes_x_class_tr_val = 1000
n_imagenes_x_class_te = 2000
n_frames = 100


############ MAIN CODE ######################################################################

def ejecucion(n_frames):

	modelo = 'ResNet50'
	params = [1e-3, [16], None]
	imageShape = (128, 128, 3)

	ratas_selected = list(range(n_classes))
	print('Ratas selected:', ratas_selected)

	fn.crear_tfrecords(directory)

	training_dataset, val_dataset, test_dataset = fn.create_datasets(ratas_selected, directory, n_imagenes_x_class_tr_val, n_imagenes_x_class_te, frames=n_frames)

	model = fn.train_model(modelo, imageShape, ratas_selected, training_dataset, val_dataset, n_imagenes_x_class_tr_val, params)
	model.save("modelo.h5")
	model = tf.keras.models.load_model("modelo.h5")

	acc, cm = fn.test_model(model, test_dataset, frames=None, save_mode="predictions", output_file="predicciones.csv")
	print(acc)
	print(cm)


	del model
	if n_frames:
		n_frames_aux = n_frames
	else:
		n_frames_aux = 1
	with open(carpeta + '/results.csv', 'a+', newline='') as f:
		write = csv.writer(f)
		write.writerow([n_frames_aux, acc] + ratas_selected)

ejecucion(n_frames)





