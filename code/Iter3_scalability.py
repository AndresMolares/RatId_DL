import functions_DEEPLEARNING as fn
import os
import csv
import threading
import argparse
import random

parser = argparse.ArgumentParser(description='Process some params.')
parser.add_argument('-c', '--count', default=0)
parser.add_argument('-t', '--tag', default='')
args = parser.parse_args()
args_count = int(args.count)
tag = args.tag

cesga = True

if cesga:
	directory = '/mnt/lustre/scratch/nlsas/home/ulc/co/amu/Dataset_128_16x9x10K_TFRecord/Dataset_128_16x9x10K_Color/'
	carpeta = './results/Iter3_scalability/' + tag + '/'
else:
	directory = 'D:/Dataset_128_16x9x10K_TFRecord/Dataset_128_16x9x10K_Color/'
	carpeta = './result/Iter3_scalability/'

os.makedirs(carpeta, exist_ok=True)

n_imagenes_x_class_te = 2000
n_videos = 9

modelo = 'ResNet50'
params = [1e-3, [16], None]
imageShape = (128, 128, 3)
dataAugmentation = None

############ MAIN CODE ########################################################

semaphore = threading.Semaphore(2)
def ejecucion(n_classes, n_imagenes_x_class_tr_val, n_prueba, video_test, video_validacion):
	with semaphore:

		results_score_total = []
		random.seed(n_prueba * 42)
		ratas_selected = random.sample(range(0, 16), n_classes)
		print('Ratas selected:', ratas_selected)

		model, test_dataset = fn.train_model(ratas_selected, list_videos,
											 directory, video_test, video_validacion,
											 n_imagenes_x_class_tr_val,
											 n_imagenes_x_class_te, imageShape,
											 modelo, params)

		results_score, cm = fn.test_model(model, test_dataset)
		#print('Results score:', results_score, 'Video:', video_test)
		results_score_total.append(results_score)
		#cm_total += cm
		del model
		with open(carpeta + 'results.csv', 'a+', newline='') as f:
			write = csv.writer(f)
			write.writerow([n_classes, n_imagenes_x_class_tr_val, n_prueba, video_test, results_score] + ratas_selected)

############ TEST #################################################################


list_pruebas = []
for n_classes in [2, 4, 6, 8, 10, 12, 14, 16]:
	if n_classes in [2]:
		reps = 8
	elif n_classes in [4]:
		reps = 4
	elif n_classes in [6]:
		reps = 3
	elif n_classes in [8, 10]:
		reps = 2
	else:
		reps = 1
	for n_imagenes_x_class_tr in [50, 100, 200, 500, 1000, 1500, 2000]:
		for n_prueba in range(reps):
			list_pruebas.append({
				'n_classes': n_classes,
				'n_imagenes_x_class_tr': n_imagenes_x_class_tr,
				'n_prueba': n_prueba,
			})

threads = []
list_videos = []
for idx_video in range(n_videos):
	list_videos.append('Video' + str(idx_video + 1))

for i in range(len(list_videos)):
	video_test = list_videos[i]
	if i == 0:
		video_validacion = list_videos[len(list_videos) - 1]
	else:
		video_validacion = list_videos[i - 1]

	#print(list_pruebas[args_count]['n_classes'], list_pruebas[args_count]['n_imagenes_x_class_tr'])
	thread = threading.Thread(target=ejecucion, args=(list_pruebas[args_count]['n_classes'], list_pruebas[args_count]['n_imagenes_x_class_tr'], list_pruebas[args_count]['n_prueba'], video_test, video_validacion))
	thread.start()
	threads.append(thread)

for thread in threads:
	thread.join()




