# Notes for extension of script:
# 	- User readline() to interactively search for word groups
# 	- On a word miss, use L2 or cosine distance to select the nearest word vector
# 		- This would require all 6B tokens to loaded in ram (but not clustered)
#		- Or use levenshtein distance assuming the word is spelled the same.
#   - Provide an interface to perform basic arithmetic on words (king - man + woman = queen)
# Look at this result from 2014 English Wikipedia:
# 'islamic', 'militant', 'islam', 'radical', 'extremists', 'islamist', 'extremist', 'outlawed'
# 'war' - 'violence' + 'peace' = 'treaty' | 300d

from sklearn.cluster import KMeans
from numbers import Number
from pandas import DataFrame
import numpy as np
import os, sys, codecs, argparse, pprint, time
from utils import *
from word_arithmetic import *
from scipy.spatial.distance import cosine

def find_word_clusters(labels_array, cluster_labels):
	cluster_to_words = autovivify_list()
	for c, i in enumerate(cluster_labels):
		cluster_to_words[i].append(labels_array[c])
	return cluster_to_words


def find_nearest(words, vec, id_to_word, df, num_results, method='cosine'):
	if method == 'cosine':
		minim = [] # min, index
		for i, v in enumerate(df):
			# skip the base word, its usually the closest
			if id_to_word[i] in words:
				continue
			dist = cosine(vec, v)
			minim.append((dist, i))
		minim = sorted(minim, key=lambda v: v[0])
		# return list of (word, cosine distance) tuples
		return [(id_to_word[minim[i][1]], minim[i][0]) for i in range(num_results)]
	else:
		raise Exception('{} is not an excepted method parameter'.format(method))

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--vector_dim', '-d',
						type=int,
						choices=[50, 100, 200, 300],
						default=100,
						help='What vector GloVe vector dimension to use '
							 '(default: 100).')
	parser.add_argument('--num_words', '-n',
						type=int,
						default=10000,
						help='The number of lines to read from the GloVe '
							 'vector file (default: 10000).')
	parser.add_argument('--num_clusters', '-k',
						default=1000,
						type=int,
						help='Number of resulting word clusters. '
						'The number of K in K-Means (default: 1000).')
	parser.add_argument('--n_jobs', '-j',
						type=int,
						default=-1,
						help='Number of cores to use when fitting K-Means. '
						     '-1 = all cores. '
							 'More cores = less time, more memory (default: -1).')
	parser.add_argument('--glove_path', '-i',
		                default='/home/jupyter-msiper/algorithmic_ml_sandbox/datasets/GloVe',
		                help='GloVe vector file path (default: data/glove)')
	return parser.parse_args()

if __name__ == '__main__':

	args = parse_args()

	filename = path = '/home/jupyter-msiper/algorithmic_ml_sandbox/datasets/GloVe/{}'.format(get_cache_filename_from_args(args))
	cluster_to_words = None
	start_time = time.time()

	vector_file = args.glove_path + '/' + 'glove.6B.' + str(args.vector_dim) + 'd.txt'
	df, labels_array, word_to_vector_dict = build_word_vector_matrix(vector_file, args.num_words)

	# if these are clustering parameters we've never seen before
	if not os.path.isfile(filename):

		print('No cached cluster found. Clustering using K-Means... ')
		kmeans_model = KMeans(init='k-means++', n_clusters=args.num_clusters, n_jobs=args.n_jobs, n_init=10, max_iter=300)
		kmeans_model.fit(df)

		cluster_labels   = kmeans_model.labels_
		cluster_to_words = list(find_word_clusters(labels_array, cluster_labels).values())

		df_dict = {'word':[], 'vector': [], 'cluster_id': [], 'centroid_vector': [], 'nearest_0_id': [], 'nearest_0_vec': [], 'nearest_1_id': [], 'nearest_1_vec': [], 'nearest_2_id': [], 'nearest_2_vec': [], 'nearest_3_id': [], 'nearest_3_vec': [], 'nearest_4_id': [], 'nearest_4_vec': []}
		for cluster_id, words in enumerate(cluster_to_words):
			dict_to_pass_to_find_nearest = {idx: word for idx, word in enumerate(words)}
			words_df = np.array([word_to_vector_dict[w] for w in words])
			word_to_id = {w_: id for id, w_ in enumerate(words)}
			for idx, word in enumerate(words):
				df_dict['word'].append(word)
				df_dict['vector'].append(str(word_to_vector_dict[word]))
				df_dict['cluster_id'].append(cluster_id)
				df_dict['centroid_vector'].append(str(kmeans_model.cluster_centers_[cluster_id]))
				res = find_nearest([word], word_to_vector_dict[word], dict_to_pass_to_find_nearest, words_df, 5, method='cosine')
				for i in range(5):
					df_dict[f'nearest_{i}_id'].append(word_to_id[res[i][0]])
					df_dict[f'nearest_{i}_vec'].append(str(word_to_vector_dict[res[i][0]]))

		df_to_reconstruct = DataFrame(df_dict)
		df_to_reconstruct.to_csv(f'/home/jupyter-msiper/algorithmic_ml_sandbox/datasets/GloVe/df_{args.num_clusters}_clusters_{args.num_words}.csv', index=False)

		# cache these clustering results
		save_json(path, cluster_to_words)
		print('Saved {} clusters to {}. Cached for later use.'.format(len(cluster_to_words), path))

	# if this kmeans fitting has already been cached
	else:
		print('Cached K-Means cluster found, loading from disk.')
		cluster_to_words = load_json(filename)

	for i, words in enumerate(cluster_to_words):
		pass
		# print('CLUSTER {}: {}'.format(i + 1, ', '.join(words)))

	if start_time != None:
			print("--- {:.2f} seconds ---".format((time.time() - start_time)))
           
            
