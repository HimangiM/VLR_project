from scipy.spatial.distance import cosine, cdist
import pickle
import numpy as np
from tqdm import tqdm
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file')
    parser.add_argument('output_file')
    parser.add_argument('mode')
    args = parser.parse_args()
    vector_dict = pickle.load(open(args.input_file, 'rb'))
    vectors = []
    for i in range(len(vector_dict)):
        vectors.append(vector_dict[i])

    vectors = np.array(vectors, 'float32')
    nearest_neighbor = {}

    for i in tqdm(range(len(vectors))):
        if args.mode == 'cosine':
            distances = cdist(np.expand_dims(vectors[i], 0), vectors, metric='cosine')
            nearest_neighbor[i] = np.argsort(distances[0])[1]
        elif args.mode == 'euclidean':
            distances = cdist(np.expand_dims(vectors[i], 0), vectors, metric='euclidean')
            nearest_neighbor[i] = np.argsort(distances[0])[1]

    with open(args.output_file, 'wb') as file_:
        pickle.dump(nearest_neighbor, file_, protocol=pickle.HIGHEST_PROTOCOL)
