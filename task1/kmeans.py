import random
import time
import heapq
import os

import numpy as np

import distances as d

_EPSILON = 3

def _assign(point, centroids, distance_func):
    distances = np.array([distance_func(point, c) for c in centroids], dtype=float)
    best = np.argmin(distances)
    return (best, distances[best])

def _assign_points(data, centroids, distance_func):
    clusters = [[] for c in centroids]
    dists = [0] * len(data)
    for i, point in enumerate(data):
        cluster, dist = _assign(point, centroids, distance_func)
        clusters[cluster].append(point)
        dists[i] = (-dist, point, cluster)
    # assign most outlying points to all empty clusters
    heapq.heapify(dists)
    for i, c in enumerate(clusters):
        if not c:
            _, point, cluster = heapq.heappop(dists)
            while not clusters[cluster]:
                _, point, cluster = heapq.heappop(dists)
            clusters[cluster].remove(point)
            clusters[i].append(point)
    return clusters

def _mean_point(cluster):
    return [round(np.mean(p), _EPSILON) for p in zip(*cluster)]

def _update_centroids(clusters):
    return np.array([_mean_point(c) for c in clusters])

def _SSE(clusters, centroids):
    return sum([(d.euclidean(point, c))**2 for (cluster, c) in zip(clusters, centroids) for point in cluster])

def kmeans(data, k, distance_func):
    prev_centroids, curr_centroids = np.empty(1), data[np.random.choice(len(data), size=10, replace=False)]
    clusters = [[]]
    prev_SSE, curr_SSE = 1e12, 1e12 - 1
    i = 0
    while i < 100: # prev_SSE > curr_SSE: # (not np.array_equal(curr_centroids, prev_centroids)): # and  and 
        clusters = _assign_points(data, curr_centroids, distance_func)
        prev_centroids, curr_centroids = curr_centroids, _update_centroids(clusters)

        dists = [round(distance_func(x, y), _EPSILON) for x, y in zip(curr_centroids, prev_centroids)]
        sizes = [len(c) for c in clusters]
        prev_SSE, curr_SSE = curr_SSE, _SSE(clusters, curr_centroids)
        print(f'Distances each centroid moved on iteration {i} are: {dists}')
        print(f'Number of elements in each cluster: {sizes}')
        print(f'Total SSE: {curr_SSE}')
        i += 1 #, 
    return (clusters, curr_centroids, curr_SSE, i)

if __name__ == '__main__':
    random.seed(time.time())
    k = 10

    data = np.loadtxt('kmeans_data/data.csv', delimiter=',', dtype=int) / 255
    
    for f in [d.euclidean, d.cosine_similarity, d.generalized_jaccard_similarity]:
        file_name = f'results/{f.__name__}.csv'

        if not os.path.exists(file_name):
            print(f'Performing kmeans using {f.__name__}...')
            clusters, centroids, SSE, iterations = kmeans(data, k, f)
            with open(file_name, 'w') as file:
                w = csv.writer(file)
                w.writerows(centroids)
                w.writerow([SSE, iterations])
                print(f'{f.__name__}.csv created!')
        else:
            print(f'{file_name} already exists, skipping generation of {f.__name__}...')
        
        print(f'Computing prediction accuracy of {f.__name__}...')
        with open('kmeans_data/label.csv') as label_file:
            with open(file_name, 'r') as centroids_file:
                centroids = np.loadtxt(centroids_file, delimiter=',', dtype=float, max_rows=10)

                true_labels = np.loadtxt(label_file, dtype=int) # [int(''.join(line)) for line in csv.reader(label_file)]
                pred_labels = np.fromiter((_assign(point, centroids, f)[0] for point in data), dtype=int)

                cluster_mappings = np.zeros((k, k))
                for t, p in np.stack((pred_labels, true_labels), axis=1):
                    for c in range(k):
                        if p == c:
                            cluster_mappings[c,t] += 1
                cluster_mappings = np.argmax(cluster_mappings, axis=0)
                pred_labels = np.stack(np.vectorize(lambda p: cluster_mappings[p])(pred_labels))

                hits = np.sum(np.equal(pred_labels, true_labels))
                accuracy = hits / len(data)
                print(f'{f.__name__} accuracy: {accuracy}\n')
