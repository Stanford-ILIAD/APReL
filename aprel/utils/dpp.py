"""
This module handles greedy estimation of the mode of a determinantal point process (DPP).
The technique is based on `Biyik et al. (2019) <https://arxiv.org/abs/1906.07975>`_.
The code is adopted from https://github.com/Stanford-ILIAD/DPP-Batch-Active-Learning/blob/master/reward_learning/dpp_sampler.py.
"""

import numpy as np
from scipy.spatial.distance import cdist, pdist, squareform


class Kernel(object):
    # Returns the submatrix of the kernel indexed by ps and qs
    def getKernel(self, ps, qs):
        return np.squeeze(ps[:, None] == qs[None, :]).astype(float)
        # Readable version: return np.array([[1. if p==q else 0. for q in qs] for p in ps])

    def __getitem__(self, args):
        ps, qs = args
        return self.getKernel(ps, qs)


class ScoredKernel(Kernel):
    def __init__(self, R, distances, scores):
        self.R = R
        self.distances = distances
        self.scores = scores

    def getKernel(self, p_ids, q_ids):
        p_ids = np.squeeze(np.array([p_ids]))
        if len(p_ids.shape) < 1:
            p_ids = np.array([p_ids])
        q_ids = np.squeeze(np.array([q_ids]))
        if len(q_ids.shape) < 1:
            q_ids = np.array([q_ids])

        D = self.distances[np.ix_(p_ids,q_ids)] ** 2

        # Readable version: D = np.array([[np.dot(p-q, p-q) for q in qs] for p in ps])
        D = np.exp(-D / (2 * self.R ** 2))

        # I added the below line to have different sample scores
        D = ((D * self.scores[p_ids]).T * self.scores[q_ids]).T

        return D


class Sampler(object):
    def __init__(self, kernel, distances, k):
        self.kernel = kernel
        self.distances = distances
        self.k = k
        # norms will hold the diagonals of the kernel
        self.norms = np.array([self.kernel[p_id, p_id][0][0] for p_id in range(len(self.distances))])
        self.clear()

    def clear(self):
        # S will hold chosen set of k points
        self.S = []
        # M will hold the inverse of the kernel on S
        self.M = np.zeros(shape=(0, 0))

    def append(self, ind):
        if len(self.S) == 0:
            self.S = [ind]
            self.M = np.array([[1. / self.norms[ind]]])
        else:
            u = self.kernel[self.S, ind]
            # Compute Schur complement inverse
            v = np.dot(self.M, u)
            scInv = 1. / (self.norms[ind] - np.dot(u.T, v))
            self.M = np.block([[self.M + scInv * np.outer(v, v), -scInv * v], [-scInv * v.T, scInv]])
            self.S.append(ind)

    def ratios(self, item_ids=None):
        if item_ids is None:
            item_ids = np.arange(len(self.distances))
        if len(self.S) == 0:
            return self.norms[item_ids]
        else:
            U = self.kernel[item_ids, self.S]
            return self.norms[item_ids] - np.sum(np.dot(U, self.M) * U, axis=1)

    def addGreedy(self):
        self.append(np.argmax(self.ratios()))

    def warmStart(self):
        for i in range(self.k):
            self.addGreedy()

    def sample(self):
        return self.S


def setup_sampler(distances, scores, k):
    dn = np.array(distances)
    dn_flat = dn[np.tril_indices(len(dn))]
    R = np.mean(np.min(np.random.choice(dn_flat, (1000,k)), axis=1))
    s = Sampler(ScoredKernel(R, distances, scores), distances, k)
    s.warmStart()
    return s


def sample_ids_mc(distances, scores, k):
    distances = np.array(distances)
    scores = np.array(scores).reshape(-1, 1)
    s = setup_sampler(distances, scores, k)
    return s.sample()


def dpp_mode(distances, scores, k):
    return sample_ids_mc(distances, scores, k)