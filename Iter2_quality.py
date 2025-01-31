import functions_DEEPLEARNING as fn
import os
import csv
import threading
import argparse
import random

parser = argparse.ArgumentParser(description='Process some params.')
parser.add_argument('-c', '--count', default=0)
parser.add_argument('-t', '--tag', default=0)
args = parser.parse_args()
args_count = int(args.count)
tag = args.tag

cesga = True

if cesga:
	directory = '/mnt/lustre/scratch/nlsas/home/ulc/co/amu/Dataset_128_16x9x10K_TFRecord/Dataset_128_16x9x10K_Color/'
	carpeta = './results/Iter2_quality/' + tag + '/'
else:
	directory = 'D:/Dataset_128_16x9x10K_TFRecord/Dataset_128_16x9x10K_Color/'
	carpeta = './result/Iter2_quality/'

os.makedirs(carpeta, exist_ok=True)

n_classes = 5
n_imagenes_x_class_tr_val = 500
n_imagenes_x_class_te = 2000
n_videos = 9

modelo = 'ResNet50'
params = [1e-3, [16], None]

############ MAIN CODE ########################################################

semaphore = threading.Semaphore(2)
def ejecucion(quality_test_type, quality_param, n_prueba, video_test, video_validacion):

	directory = '/mnt/lustre/scratch/nlsas/home/ulc/co/amu/Dataset_128_16x9x10K_TFRecord/Dataset_128_16x9x10K_Color/'
	color = True
	dataAugmentation = None
	imageShape = (128, 128, 3)

	with semaphore:
		if quality_param[-2:] == 'BN':
			directory = '/mnt/lustre/scratch/nlsas/home/ulc/co/amu/Dataset_128_16x9x10K_TFRecord/Dataset_128_16x9x10K_BN/'
			color = False

		if quality_param[0:4] == 'FLIP':
			dataAugmentation = 'FLIP'
		if quality_param[0:4] == 'ROTA':
			dataAugmentation = 'ROTA'
		if quality_param[0:4] == 'ZOOM':
			dataAugmentation = 'ZOOM'
		if quality_param[0:4] == 'MIXX':
			dataAugmentation = 'MIX'

		if quality_param[:2] == '32':
			imageShape = (32, 32, 3)
		elif quality_param[:2] == '64':
			imageShape = (64, 64, 3)

		results_score_total = []
		random.seed(n_prueba * 42)
		ratas_selected = random.sample(range(0, 16), n_classes)
		print('Ratas selected:', ratas_selected)

		model, test_dataset = fn.train_model(ratas_selected, list_videos,
											 directory, video_test, video_validacion,
											 n_imagenes_x_class_tr_val,
											 n_imagenes_x_class_te, imageShape,
											 modelo, params, dataAugmentation, color)

		results_score, cm = fn.test_model(model, test_dataset)
		#print('Results score:', results_score, 'Video:', video_test)
		results_score_total.append(results_score)
		#cm_total += cm
		del model

		with open(carpeta + 'results' + quality_test_type + '.csv', 'a+', newline='') as f:
			write = csv.writer(f)
			write.writerow([quality_test_type, quality_param, n_prueba, video_test] + ratas_selected + [results_score])



############ TEST #################################################################

pruebas = [
	{'quality_test_type': 'RESOLUTION', 'quality_param': 32},
	{'quality_test_type': 'RESOLUTION', 'quality_param': 64},
	{'quality_test_type': 'RESOLUTION', 'quality_param': 128},
	{'quality_test_type': 'DATAAUGMENTATION', 'quality_param': 'FLIP'},
	{'quality_test_type': 'DATAAUGMENTATION', 'quality_param': 'ROTA'},
	{'quality_test_type': 'DATAAUGMENTATION', 'quality_param': 'ZOOM'},
	{'quality_test_type': 'DATAAUGMENTATION', 'quality_param': 'MIXX'},
	{'quality_test_type': 'DATAAUGMENTATION', 'quality_param': 'MIXX_BN'},
	{'quality_test_type': 'COLOR', 'quality_param': 'COLOR'},
	{'quality_test_type': 'COLOR', 'quality_param': 'BN'}

]

list_pruebas = []
for prueba in pruebas:
	for n_prueba in range(3):
		prueba['n_prueba'] = n_prueba
		list_pruebas.append(prueba.copy())

threads = []
list_videos = []
for idx_video in range(n_videos):
	list_videos.append('Video' + str(idx_video + 1))

#for video_test in list_videos:
for i in range(len(list_videos)):
	video_test = list_videos[i]
	if i == 0:
		video_validacion = list_videos[len(list_videos) - 1]
	else:
		video_validacion = list_videos[i - 1]
	thread = threading.Thread(target=ejecucion, args=(list_pruebas[args_count]['quality_test_type'], list_pruebas[args_count]['quality_param'], list_pruebas[args_count]['n_prueba'], video_test, video_validacion))
	thread.start()
	threads.append(thread)

for thread in threads:
	thread.join()




