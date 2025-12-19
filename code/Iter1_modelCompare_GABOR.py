import functions_SVM_GABOR as fn
import os
import random
import argparse
import csv
import threading

configuraciones_gabor = []

for kernel in ['linear', 'rbf']:
	for c in [1, 0.1, 0.01]:
		for n_prueba in range(5):
			configuraciones_gabor.append({
				'kernel': kernel,
				'c': c,
				'n_prueba': n_prueba,
			})

def filas(index, rango):
	if index in rango:
		return False
	return True

#os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

parser = argparse.ArgumentParser(description='Process some params.')
parser.add_argument('-c', '--count', default=0)
parser.add_argument('-m', '--model', default='Gabor')
parser.add_argument('-t', '--tag', default=0)
args = parser.parse_args()
args_count = int(args.count)
modelo = args.model
tag = args.tag

cesga = True

if cesga:
	directory = '/mnt/lustre/scratch/nlsas/home/ulc/co/amu/Dataset_128_16x9x10K_TFRecord/Dataset_128_16x9x10K_Color/'
	carpeta = './results/DEEPTRACK/' + tag + '/Iter1_modelCompare/'
else:
	directory = 'D:/Dataset_128_16x9x10K_TFRecord/Dataset_128_16x9x10K_Color/'
	carpeta = './result/Iter1_modelCompare/'

os.makedirs(carpeta, exist_ok=True)

imageShape = (128, 128, 3)
n_classes = 5
n_imagenes_x_class_tr = 2000
n_imagenes_x_class_te = 2000
n_videos = 9

semaphore = threading.Semaphore(2)

def ejecucion(params, n_prueba, video_test):
	with semaphore:
		results_score_total = []
		random.seed(n_prueba * 42)
		ratas_selected = random.sample(range(0, 16), n_classes)
		print('Ratas selected:', ratas_selected)

		model, pca, test_dataset = fn.train_model_tfrecords(ratas_selected, list_videos,
																		   directory, video_test,
																		   n_imagenes_x_class_tr,
																		   n_imagenes_x_class_te, params, True)

		results_score, cm = fn.test_model(model, test_dataset, pca)
		results_score_total.append(results_score)
		del model

		with open(carpeta + 'results' + modelo + '.csv', 'a+', newline='') as f:
			write = csv.writer(f)
			write.writerow([modelo, params[0], params[1], params[2], n_prueba, video_test] + ratas_selected + [results_score])


threads = []
list_videos = []
for idx_video in range(n_videos):
	list_videos.append('Video' + str(idx_video + 1))

for video_test in list_videos:
	params = [configuraciones_gabor[args_count]['kernel'], configuraciones_gabor[args_count]['c'], None]
	thread = threading.Thread(target=ejecucion, args=(params, configuraciones_gabor[args_count]['n_prueba'], video_test))
	thread.start()
	threads.append(thread)

for thread in threads:
	thread.join()