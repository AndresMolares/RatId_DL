import functions_DEEPLEARNING as fn
import os
import random
import argparse
import csv
import threading

configuraciones_rna = []

for lr_base in [1e-3]:
	for n_neurons in [[2], [8], [16], [64], [8, 2], [16, 8]]:
		for n_prueba in range(3):
			configuraciones_rna.append({
				'lr_base': lr_base,
				'n_neurons': n_neurons,
				'n_prueba': n_prueba,
			})

#os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

parser = argparse.ArgumentParser(description='Process some params.')
parser.add_argument('-c', '--count', default=0)
parser.add_argument('-m', '--model', default=0)
parser.add_argument('-t', '--tag', default=0)
args = parser.parse_args()
args_count = int(args.count)
modelo = args.model
tag = args.tag

cesga = True

if cesga:
	directory = '/mnt/lustre/scratch/nlsas/home/ulc/co/amu/Dataset_128_16x9x10K_TFRecord/Dataset_128_16x9x10K_Color/'
	carpeta = './results/Iter1_modelCompare/' + tag + '/'
else:
	directory = 'D:/Dataset_128_16x9x10K_TFRecord/Dataset_128_16x9x10K_Color/'
	carpeta = './result/Iter1_modelCompare/'

os.makedirs(carpeta, exist_ok=True)

imageShape = (128, 128, 3)
n_classes = 5
n_imagenes_x_class_tr_val = 500
n_imagenes_x_class_te = 2000
n_videos = 9

semaphore = threading.Semaphore(2)

def ejecucion(params, n_prueba, video_test, video_validacion):
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

		with open(carpeta + 'results' + modelo + '.csv', 'a+', newline='') as f:
			write = csv.writer(f)
			write.writerow([modelo, params[0], params[1], params[2], n_prueba, video_test] + ratas_selected + [results_score])


threads = []
list_videos = []
for idx_video in range(n_videos):
	list_videos.append('Video' + str(idx_video + 1))

for j in range(len(list_videos)):
	video_test = list_videos[j]
	if j == 0:
		video_validacion = list_videos[len(list_videos) - 1]
	else:
		video_validacion = list_videos[j - 1]
	params = [configuraciones_rna[args_count]['lr_base'], configuraciones_rna[args_count]['n_neurons'], None]
	thread = threading.Thread(target=ejecucion, args=(params, configuraciones_rna[args_count]['n_prueba'], video_test, video_validacion))
	thread.start()
	threads.append(thread)

for thread in threads:
	thread.join()


