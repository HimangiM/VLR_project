from scipy.spatial.distance import cosine, cdist
import pickle
import numpy as np
from tqdm import tqdm

if __name__ == '__main__':
    vector_dict = pickle.load(open('latent_vectors.pickle', 'rb'))
    vectors = []
    for i in range(len(vector_dict)):
        vectors.append(vector_dict[i])

    vectors = np.array(vectors, 'float32')
    nearest_neighbor = {}
    for i in tqdm(range(len(vectors))):
        distances = cdist(np.expand_dims(vectors[i], 0), vectors, metric='cosine')
        nearest_neighbor[i] = np.argsort(distances[0])[1]

    with open('neares_neighbor.pickle', 'wb') as file_:
        pickle.dump(nearest_neighbor, file_, protocol=pickle.HIGHEST_PROTOCOL)