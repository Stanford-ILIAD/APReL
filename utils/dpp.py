"""
This file contains functions useful for the DPP method of generating a diverse batch of optimal queries.
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
    def __init__(self, R, distances, scores, alpha=2, gamma=1):
        self.R = R
        self.distances = distances
        if alpha > 0:
            self.scores = scores ** (float(gamma) / alpha)
        else:
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


class ConditionedScoredKernel(Kernel):
    def __init__(self, R, distances, scores, cond_ids, alpha=2, gamma=1):
        self.R = R
        self.distances = np.array(distances)
        self.cond_ids = np.sort(cond_ids)
        if alpha > 0:
            self.scores = np.array(scores ** (float(gamma) / alpha)).reshape(-1)
        else:
            self.scores = scores
        self.kernel = self.computeFullKernel()

    def computeFullKernel(self):
        # Construct kernel matrix
        D = self.distances ** 2
        D = np.exp(-D / (2 * self.R ** 2))
        D = ((D * self.scores).T * self.scores).T

        # conditioning
        if len(self.cond_ids) > 0:
            eye1 = np.eye(len(self.distances))
            for id in self.cond_ids:
                eye1[id, id] = 0
            D = np.linalg.inv(D + eye1)
            mask = np.full(len(self.distances), True, dtype=bool)
            mask[self.cond_ids] = False
            D = D[mask, :]
            D = D[:, mask]
            D = np.linalg.inv(D) - np.eye(len(self.distances) - len(self.cond_ids))
            for id in self.cond_ids:  # cond_ids is sorted
                D = np.insert(D, id, 0, axis=0)
                D = np.insert(D, id, 0, axis=1)

        return D

    def getKernel(self, p_ids, q_ids):
        p_ids = np.squeeze(np.array([p_ids]))
        if len(p_ids.shape) < 1:
            p_ids = np.array([p_ids])
        q_ids = np.squeeze(np.array([q_ids]))
        if len(q_ids.shape) < 1:
            q_ids = np.array([q_ids])

        D = self.kernel[p_ids, :]
        D = D[:, q_ids]
        return D


class Sampler(object):
    def __init__(self, kernel, distances, k, cond_ids=[]):
        self.kernel = kernel
        self.distances = distances
        self.k = k
        self.cond_ids = cond_ids
        # norms will hold the diagonals of the kernel
        self.norms = np.array([self.kernel[p_id, p_id][0][0] for p_id in range(len(self.distances))])
        self.clear()

    def clear(self):
        # S will hold chosen set of k points
        self.S = list(self.cond_ids)
        # M will hold the inverse of the kernel on S
        if len(self.cond_ids) == 0:
            self.M = np.zeros(shape=(0, 0))
        else:
            self.M = np.linalg.pinv(self.kernel[self.S, self.S])

    def makeSane(self):
        self.M = np.linalg.pinv(self.kernel[self.S, self.S])

    def testSanity(self):
        eps = 1e-4
        assert self.M.shape == (len(self.S), len(self.S))
        if len(self.S) > 0:
            diff = np.abs(np.dot(self.M, self.kernel[self.S, self.S]) - np.eye(len(self.S)))
            assert np.all(np.abs(diff) <= eps)

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

    def remove(self, i):
        if len(self.S) == 1:
            self.S = []
            self.M = np.zeros(shape=(0, 0))
        else:
            mask = [True] * len(self.S)
            mask[i] = False
            # Readable version: mask = [j!=i for j in range(len(self.S))]
            scInv = self.M[i, i]
            v = self.M[mask, i]
            self.M = self.M[mask, :][:, mask] - np.outer(v, v) / scInv
            self.S = self.S[:i] + self.S[i + 1:]

    # Return array containing ratio of increase in kernel determinant after adding each point in space
    def ratios(self, item_ids=None):
        if item_ids is None:
            item_ids = np.arange(len(self.distances))
        if len(self.S) == 0:
            return self.norms[item_ids]
        else:
            U = self.kernel[item_ids, self.S]
            return self.norms[item_ids] - np.sum(np.dot(U, self.M) * U, axis=1)

    # Finds greedily the item to add to maximize the determinant of the kernel
    def addGreedy(self):
        self.append(np.argmax(self.ratios()))

    # Important step, because we need to start from a point whose probability is not too small
    def warmStart(self):
        self.clear()
        for i in range(self.k):
            self.addGreedy()

    def keepCurrentState(self):
        self.backup_S = self.S.copy()
        self.backup_M = self.M.copy()

    def restoreState(self):
        self.S = self.backup_S.copy()
        self.M = self.backup_M.copy()

    # Run one step of Markov chain
    def step(self, alpha=1.):
        temp = np.random.randint(len(self.cond_ids), len(self.S))
        remove_id = self.S[temp]
        self.remove(temp)

        add_id = np.random.randint(len(self.distances))

        new_prob = np.maximum(self.ratios(add_id), 0.) ** alpha
        old_prob = np.maximum(self.ratios(remove_id), 0.) ** alpha

        if np.random.rand() < new_prob / old_prob:
            self.append(add_id)
        else:
            self.append(remove_id)

    def sample(self, alpha=2., steps=1000, makeSaneEvery=10):
        for iter in range(steps):
            self.step(alpha=alpha)
            if iter % makeSaneEvery == 0:
                self.makeSane()
        return self.S


def setup_sampler(distances, scores, k, alpha, gamma, cond_ids):
    dn = np.array(distances)
    dn_flat = dn[np.tril_indices(len(dn))]
    R = np.mean(np.min(np.random.choice(dn_flat, (1000,k)), axis=1))
    s = Sampler(ScoredKernel(R, distances, scores, alpha, gamma), distances, k, cond_ids)
    s.warmStart()
    return s


def sample_ids_mc(distances, scores, k, alpha=2., gamma=1., cond_ids=[], steps=1000):
    distances = np.array(distances)
    scores = np.array(scores).reshape(-1, 1)
    s = setup_sampler(distances, scores, k, alpha, gamma, cond_ids)
    x = s.sample(alpha=alpha, steps=steps)
    return x[len(cond_ids):]


def dpp_mode(distances, scores, k):
    return sample_ids_mc(distances, scores, k, steps=0)