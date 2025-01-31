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
	carpeta = './results/Iter4_frames/' + tag + '/'
else:
	directory = 'D:/Dataset_128_16x9x10K_TFRecord/Dataset_128_16x9x10K_Color/'
	carpeta = './result/Iter4_frames/'

os.makedirs(carpeta, exist_ok=True)

n_classes = 5
n_imagenes_x_class_tr_val = 1000
n_imagenes_x_class_te = 2000
n_videos = 9

modelo = 'ResNet50'
params = [1e-3, [16], None]
imageShape = (128, 128, 3)

############ MAIN CODE ########################################################

semaphore = threading.Semaphore(2)
def ejecucion(n_frames, n_prueba, video_test, video_validacion):
	with semaphore:

		results_score_total = []
		random.seed(n_prueba * 42)
		ratas_selected = random.sample(range(0, 16), n_classes)
		print('Ratas selected:', ratas_selected)

		model, test_dataset = fn.train_model(ratas_selected, list_videos,
											 directory, video_test, video_validacion,
											 n_imagenes_x_class_tr_val,
											 n_imagenes_x_class_te, imageShape,
											 modelo, params, frames=n_frames)

		results_score, cm = fn.test_model(model, test_dataset, frames=n_frames)
		results_score_total.append(results_score)
		del model
		if n_frames:
			n_frames_aux = n_frames
		else:
			n_frames_aux = 1
		with open(carpeta + 'results.csv', 'a+', newline='') as f:
			write = csv.writer(f)
			write.writerow([n_frames_aux, n_prueba, video_test, results_score] + ratas_selected)

############ TEST #################################################################

list_pruebas = []
for n_frames in [None, 25, 50, 75, 100, 125, 150, 175, 200, 225, 250]:
	for n_prueba in range(3):
		list_pruebas.append({
			'n_frames': n_frames,
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

	thread = threading.Thread(target=ejecucion, args=(list_pruebas[args_count]['n_frames'], list_pruebas[args_count]['n_prueba'], video_test, video_validacion))
	thread.start()
	threads.append(thread)

for thread in threads:
	thread.join()




