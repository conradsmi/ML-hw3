import math

import numpy as np

def euclidean(p1, p2):
    return np.sqrt(np.sum(np.square(p1 - p2)))
    # return math.sqrt(sum([(x1-x2)**2 for x1, x2 in zip(p1, p2)]))

def cosine_similarity(p1, p2):
    return 1 - (p1.dot(p2) / (np.sqrt(p1.dot(p1))*np.sqrt(p2.dot(p2))))

def generalized_jaccard_similarity(p1, p2):
    mins = np.sum(np.minimum(p1, p2))
    maxs = np.sum(np.maximum(p1, p2))
    # mins, maxs = [sum(m) for m in zip(*[[min(x1, x2), max(x1, x2)] for x1, x2 in zip(p1, p2)])]
    return 1 - (mins / maxs)