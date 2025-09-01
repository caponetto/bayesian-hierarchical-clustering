# -*- coding: utf-8 -*-

# License: GPL 3.0

import numpy as np
from scipy.special import gammaln

import bhc.api as api


class BayesianHierarchicalClustering(
    api.AbstractBayesianBasedHierarchicalClustering
):
    """
    Reference: HELLER, Katherine A.; GHAHRAMANI, Zoubin.
               Bayesian hierarchical clustering.
               In: Proceedings of the 22nd international conference on
                   Machine learning. 2005. p. 297-304.
               http://mlg.eng.cam.ac.uk/zoubin/papers/icml05heller.pdf
    """

    def __init__(self, data, model, alpha, cut_allowed):
        super().__init__(data, model, alpha, cut_allowed)

    def build(self):
        n_objects = self.data.shape[0]

        weights = []

        # active nodes (all)
        active_nodes = np.arange(n_objects)
        # assignments - starting each point in its own cluster
        assignments = np.arange(n_objects)
        # stores information from temporary merges
        tmp_merge = None
        hierarchy_cut = False

        # for every single data point
        log_p = np.zeros(n_objects)
        log_d = np.zeros(n_objects)
        n = np.ones(n_objects)
        for i in range(n_objects):
            # compute log(d_k)
            log_d[i] = BayesianHierarchicalClustering.__calc_log_d(
                self.alpha, n[i], None
            )
            # compute log(p_i)
            log_p[i] = self.model.calc_log_mlh(self.data[i])

        ij = n_objects - 1

        # for every pair of data points
        pair_count = n_objects * (n_objects - 1) // 2
        tmp_merge = np.empty((pair_count, 5), dtype=float)
        row = 0
        for i in range(n_objects):
            log_p_k_row = self.model.row_of_log_likelihood_for_pairs(
                self.data, i
            )
            for j in range(i + 1, n_objects):
                # compute log(d_k)
                n_ch = n[i] + n[j]
                log_d_ch = log_d[i] + log_d[j]
                log_dk = BayesianHierarchicalClustering.__calc_log_d(
                    self.alpha, n_ch, log_d_ch
                )
                # compute log(pi_k)
                log_pik = np.log(self.alpha) + gammaln(n_ch) - log_dk
                # compute log(p_k)
                log_p_k = log_p_k_row[j - i - 1]  # since j starts at i + 1
                # compute log(r_k)
                log_p_ch = log_p[i] + log_p[j]
                r1 = log_pik + log_p_k
                r2 = log_d_ch - log_dk + log_p_ch
                log_r = r1 - r2
                # store results
                tmp_merge[row] = [i, j, log_r, r1, r2]
                row += 1

        # find clusters to merge
        arc_list = np.empty(0, dtype=api.Arc)
        data_per_cluster = [np.array([self.data[i]]) for i in range(n_objects)]
        while active_nodes.size > 1:
            # find i, j with the highest probability of the merged hypothesis
            position = np.argmax(
                tmp_merge[:, 2]
            )  # returns the first occurrence
            i, j, log_r, r1, r2 = tmp_merge[position]
            i = int(i)
            j = int(j)
            weights.append(log_r)

            # cut if required and stop
            if self.cut_allowed and log_r < 0:
                hierarchy_cut = True
                break

            # new node ij
            ij = n.size
            n_ch = n[i] + n[j]
            n = np.append(n, n_ch)
            # compute log(d_ij)
            log_d_ch = log_d[i] + log_d[j]
            log_d_ij = BayesianHierarchicalClustering.__calc_log_d(
                self.alpha, n[ij], log_d_ch
            )
            log_d = np.append(log_d, log_d_ij)
            # update cluster assignments
            data_per_cluster.append(
                np.vstack((data_per_cluster[i], data_per_cluster[j]))
            )
            data_per_cluster[i] = None
            data_per_cluster[j] = None
            assignments[np.argwhere(assignments == i)] = ij
            assignments[np.argwhere(assignments == j)] = ij

            # create arcs from ij to i,j
            arc_i = api.Arc(ij, i)
            arc_j = api.Arc(ij, j)
            arc_list = np.append(arc_list, [arc_i, arc_j])

            # delete i,j from active list and add ij
            i_idx = np.argwhere(active_nodes == i).flatten()
            j_idx = np.argwhere(active_nodes == j).flatten()
            active_nodes = np.delete(active_nodes, [i_idx, j_idx])
            active_nodes = np.append(active_nodes, ij)

            # clean up tmp_merge
            # keep rows where neither column 0 nor column 1 equals i or j
            mask = ~np.isin(tmp_merge[:, :2], [i, j]).any(axis=1)
            tmp_merge = tmp_merge[mask]

            # compute log(p_ij)
            t1 = np.maximum(r1, r2)
            t2 = np.minimum(r1, r2)
            log_p_ij = t1 + np.log(1 + np.exp(t2 - t1))
            log_p = np.append(log_p, log_p_ij)

            # for every pair ij x active
            collected_merge_info = np.empty(
                (len(active_nodes) - 1, 5), dtype=float
            )
            for k in range(active_nodes.size - 1):
                # compute log(d_k)
                n_ch = n[k] + n[ij]
                log_d_ch = log_d[k] + log_d[ij]
                log_dij = BayesianHierarchicalClustering.__calc_log_d(
                    self.alpha, n_ch, log_d_ch
                )
                # compute log(pi_k)
                log_pik = np.log(self.alpha) + gammaln(n_ch) - log_dij
                # compute log(p_k)
                data_merged = np.vstack(
                    (data_per_cluster[ij], data_per_cluster[active_nodes[k]])
                )
                log_p_ij = self.model.calc_log_mlh(data_merged)
                # compute log(r_k)
                log_p_ch = log_p[ij] + log_p[active_nodes[k]]
                r1 = log_pik + log_p_ij
                r2 = log_d_ch - log_dij + log_p_ch
                log_r = r1 - r2
                # store results
                collected_merge_info[k] = [ij, active_nodes[k], log_r, r1, r2]

            tmp_merge = np.vstack((tmp_merge, collected_merge_info))

        return api.Result(
            arc_list,
            np.arange(0, ij + 1),
            log_p[-1],
            np.array(weights),
            hierarchy_cut,
            len(np.unique(assignments)),
        )

    @staticmethod
    def __calc_log_d(alpha, nk, log_d_ch):
        if nk == 1 and log_d_ch is None:
            return np.log(alpha)
        else:
            dk_t1 = np.log(alpha) + gammaln(nk)
            dk_t2 = log_d_ch
            a = np.maximum(dk_t1, dk_t2)
            b = np.minimum(dk_t1, dk_t2)
            return a + np.log(1 + np.exp(b - a))
