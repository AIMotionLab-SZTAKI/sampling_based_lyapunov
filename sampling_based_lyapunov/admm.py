import matplotlib.pyplot as plt
import cvxpy as cvx
from sampling_based_lyapunov.util import format_matrix
import numpy as np
import casadi as ca
from sampling_based_lyapunov.system_models import System, VanDerPol, Trophino143, Trophino144, Trophino137, Quadrotor,\
    AnghelA, Trophino145, Example4D, Example5D
import os
import pickle
from sampling_based_lyapunov.lyapunov import Lyapunov
from concurrent.futures import ProcessPoolExecutor, as_completed


class ADMM(Lyapunov):
    def __init__(self, der, num_samples, system: System, eps, delta, verbose, slices, threads, max_iter, rho, tol):
        super(ADMM, self).__init__(der, num_samples, system, eps, delta, verbose)
        self.slices = slices
        self.threads = threads
        self.max_iter = max_iter
        self.rho = rho
        self.tol = tol
        self.samples_order = None
        self.executor = ProcessPoolExecutor(max_workers=self.threads)

    def construct_prob(self, samples, roi, thread):
        n = self.system.dim

        P_dim = ((self.der + 1) * n) ** 2
        P = cvx.Variable(((self.der + 1) * n, (self.der + 1) * n), symmetric=True)

        samples_roi = samples[:, roi, :]

        if self.samples_order is None:
            import random
            self.samples_order = [i for i in range(samples_roi.shape[1])]
            random.shuffle(self.samples_order)
        samples_roi = samples_roi[:, self.samples_order, :]

        num_roi = samples_roi.shape[1]
        num_not_roi = len(roi) - num_roi

        alpha = cvx.Variable(num_roi + self.max_iter, nonneg=True)
        constraints = []

        u = cvx.Parameter(P_dim + num_roi + self.max_iter, value=np.zeros(P_dim + num_roi + self.max_iter))
        z = cvx.Parameter(P_dim + num_roi + self.max_iter, value=np.zeros(P_dim + num_roi + self.max_iter))
        rho = self.rho  # cvx.Parameter(value=self.rho, pos=True)

        not_roi = [not elem for elem in roi]
        samples_not_roi = samples[:, not_roi, :]

        # Split samples_roi to threads
        # samples_roi = np.array_split(samples_roi, self.threads, axis=1)
        if thread == 1:
            samples_roi = samples_roi[:, :-10, :]
        else:
            samples_roi = samples_roi[:, 10:, :]
        alpha_idx_shift = 0  # sum([arr.shape[1] for arr in samples_roi[:thread]])
        # samples_roi = samples_roi[thread]
        # roi = np.array_split(np.array(roi), self.threads)[thread].tolist()
        # Split samples_roi for tractable memory allocation
        samples_roi_lst = np.array_split(samples_roi, self.slices, axis=1)

        # num_roi = samples_roi.shape[1]

        len_slices = np.hstack((0, np.cumsum([elem.shape[1] for elem in samples_roi_lst])))

        # \dot{V}(x) < alpha within the ROI
        for slice in range(self.slices):
            start_idx = len_slices[slice]
            end_idx = len_slices[slice + 1]
            dim = end_idx - start_idx
            A1 = np.zeros((dim, P_dim + dim))
            b1 = np.zeros(dim)
            I = np.eye(dim)

            for i in range(start_idx, end_idx):
                vec1 = np.reshape(samples_roi[:, i, 0:self.der + 1].T, (-1,))
                vec2 = np.reshape(samples_roi[:, i, 1:self.der + 2].T, (-1,))
                A1[i - start_idx, :] = np.hstack((np.kron(vec1, vec2) + np.kron(vec2, vec1), -I[i - start_idx, :]))
                b1[i - start_idx] = -self.eps * samples_roi[:, i, 0].T @ samples_roi[:, i, 0]
            constraints += [A1 @ cvx.hstack((cvx.vec(P), alpha[alpha_idx_shift+start_idx:alpha_idx_shift+end_idx])) <= b1]

        # V(x) > 1 + \delta outside the roi
        A2 = np.zeros((num_not_roi, P_dim))
        for i in range(num_not_roi):
            vec = np.reshape(samples_not_roi[:, i, 0:self.der + 1].T, (-1,))
            A2[i, :] = -np.kron(vec, vec)
        b2 = -(1 + self.delta) * np.ones(num_not_roi)
        constraints += [A2 @ cvx.vec(P) <= b2]

        # V(x) < 1 + \alpha within the roi
        for slice in range(self.slices):
            start_idx = len_slices[slice]
            end_idx = len_slices[slice + 1]
            dim = end_idx - start_idx
            A3 = np.zeros((dim, P_dim + dim))
            I = np.eye(dim)
            for i in range(start_idx, end_idx):
                vec = np.reshape(samples_roi[:, i, 0:self.der + 1].T, (-1,))
                A3[i - start_idx, :] = np.hstack((np.kron(vec, vec), -I[i - start_idx, :]))
            b3 = np.ones(dim)
            constraints += [A3 @ cvx.hstack((cvx.vec(P), alpha[alpha_idx_shift+start_idx:alpha_idx_shift+end_idx])) <= b3]

            # V(x) > 0 within the roi
            constraints += [-A3[:, :P_dim] @ cvx.vec(P) <= 0]

        A_ext_roi, b_ext_roi, A_ext_n_roi, b_ext_n_roi, constraints_ext = self.construct_ext_prob(num_roi, P, alpha)
        constraints += constraints_ext
        ext_con = {"A_r": A_ext_roi, "b_r": b_ext_roi, "A_nr": A_ext_n_roi, "b_nr": b_ext_n_roi}

        # Cost: minimize L1 norm of \alpha
        vars = cvx.hstack((cvx.vec(P), 1e-4*alpha))
        obj = cvx.sum(alpha) + rho*cvx.sum_squares(vars - z + u)
        prob = cvx.Problem(cvx.Minimize(obj), constraints)

        return vars, z, u, rho, ext_con, prob

    def construct_ext_prob(self, num_roi, P, alpha):
        n = self.system.dim

        P_dim = ((self.der + 1) * n) ** 2

        A_ext_roi = cvx.Parameter((2 * self.max_iter, P_dim + self.max_iter),
                                  value=np.zeros((2 * self.max_iter, P_dim + self.max_iter)))
        b_ext_roi = cvx.Parameter(2 * self.max_iter, value=np.zeros(2 * self.max_iter))
        A_ext_n_roi = cvx.Parameter((2 * self.max_iter, P_dim), value=np.zeros((2 * self.max_iter, P_dim)))
        b_ext_n_roi = cvx.Parameter(2 * self.max_iter, value=np.zeros(2 * self.max_iter))

        constraints = []
        # \dot V(x) < alpha, V(x) < 1 + alpha within the roi
        constraints += [A_ext_roi @ cvx.hstack((cvx.vec(P), alpha[num_roi:])) <= b_ext_roi]
        # V(x) > 1 outside the roi
        constraints += [A_ext_n_roi @ cvx.vec(P) <= b_ext_n_roi]
        return A_ext_roi, b_ext_roi, A_ext_n_roi, b_ext_n_roi, constraints

    def find_lyapunov_lasso_precompile_slicing(self, samples, roi, slices, max_iter):
        n = self.system.dim

        P_dim = ((self.der + 1) * n) ** 2
        P = cvx.Variable(((self.der + 1) * n, (self.der + 1) * n), symmetric=True)

        A_ext_roi = cvx.Parameter((2 * max_iter, P_dim + max_iter))
        b_ext_roi = cvx.Parameter(2 * max_iter)
        A_ext_n_roi = cvx.Parameter((2 * max_iter, P_dim))
        b_ext_n_roi = cvx.Parameter(2 * max_iter)

        alpha_bound_ext = cvx.Parameter(max_iter)

        samples_roi = samples[:, roi, :]
        not_roi = [not elem for elem in roi]
        samples_not_roi = samples[:, not_roi, :]

        num_roi = samples_roi.shape[1]
        num_not_roi = len(roi) - num_roi
        alpha = cvx.Variable(num_roi + max_iter, nonneg=True)
        # constraints = [P >= np.zeros(((self.der + 1) * n, (self.der + 1) * n))]
        constraints = []

        # Split samples_roi for tractable memory allocation
        samples_roi_lst = np.array_split(samples_roi, slices, axis=1)
        roi_lst = [elem.tolist() for elem in np.array_split(np.array(roi), slices)]

        len_slices = np.hstack((0, np.cumsum([elem.shape[1] for elem in samples_roi_lst])))

        # \dot{V}(x) < alpha within the ROI
        for slice in range(slices):
            start_idx = len_slices[slice]
            end_idx = len_slices[slice+1]
            dim = end_idx - start_idx
            A1 = np.zeros((dim, P_dim + dim))
            b1 = np.zeros(dim)
            I = np.eye(dim)

            for i in range(start_idx, end_idx):
                vec1 = np.reshape(samples_roi[:, i, 0:self.der + 1].T, (-1,))
                vec2 = np.reshape(samples_roi[:, i, 1:self.der + 2].T, (-1,))
                A1[i-start_idx, :] = np.hstack((np.kron(vec1, vec2) + np.kron(vec2, vec1), -I[i-start_idx, :]))
                b1[i-start_idx] = -self.eps * samples_roi[:, i, 0].T @ samples_roi[:, i, 0]
            constraints += [A1 @ cvx.hstack((cvx.vec(P), alpha[start_idx:end_idx])) <= b1]


        # V(x) > 1 + \delta outside the roi
        A2 = np.zeros((num_not_roi, P_dim))
        for i in range(num_not_roi):
            vec = np.reshape(samples_not_roi[:, i, 0:self.der + 1].T, (-1,))
            A2[i, :] = -np.kron(vec, vec)
        b2 = -(1 + self.delta) * np.ones(num_not_roi)
        constraints += [A2 @ cvx.vec(P) <= b2]

        # V(x) < 1 + \alpha within the roi
        for slice in range(slices):
            start_idx = len_slices[slice]
            end_idx = len_slices[slice + 1]
            dim = end_idx - start_idx
            A3 = np.zeros((dim, P_dim + dim))
            I = np.eye(dim)
            for i in range(start_idx, end_idx):
                vec = np.reshape(samples_roi[:, i, 0:self.der + 1].T, (-1,))
                A3[i - start_idx, :] = np.hstack((np.kron(vec, vec), -I[i - start_idx, :]))
            b3 = np.ones(dim)
            constraints += [A3 @ cvx.hstack((cvx.vec(P), alpha[start_idx:end_idx])) <= b3]

            # V(x) > 0 within the roi
            constraints += [-A3[:, :P_dim] @ cvx.vec(P) <= 0]


        # Unused alphas are zero
        constraints += [alpha[num_roi:] <= alpha_bound_ext]
        # \dot V(x) < alpha, V(x) < 1 + alpha within the roi
        constraints += [A_ext_roi @ cvx.hstack((cvx.vec(P), alpha[num_roi:])) <= b_ext_roi]
        # V(x) > 1 outside the roi
        constraints += [A_ext_n_roi @ cvx.vec(P) <= b_ext_n_roi]

        # Cost: minimize L1 norm of \alpha
        alpha_coef = cvx.Parameter(num_roi + max_iter)
        prob = cvx.Problem(cvx.Minimize(alpha_coef @ alpha), constraints)

        return P, alpha, prob, A_ext_roi, b_ext_roi, A_ext_n_roi, b_ext_n_roi, alpha_bound_ext, alpha_coef

    def run_admm(self, vars, zs, us, rhos, prob):
        iter = 0
        while True:
            ## set parameter values
            futures = [self.executor.submit(solve, problem, self.verbose, var) for
                       (problem, var) in zip(prob, vars)]

            # collect results
            var_num = []
            for future in as_completed(futures):
                var_num_, sol_time = future.result()
                var_num.append(var_num_)
                self.solve_time += sol_time


            x_bar = 1 / self.threads * sum([x + u.value for x, u in zip(var_num, us)])
            z_prev = []
            for i in range(self.threads):
                z_prev.append(zs[i].value)
                zs[i].value = x_bar
                us[i].value = us[i].value + var_num[i] - zs[i].value
            r = np.hstack(var_num) - np.tile(x_bar, self.threads)
            s = rhos[0] * (np.hstack(z_prev) - np.tile(x_bar, self.threads))

            r_norm = np.linalg.norm(r)
            s_norm = np.linalg.norm(s)
            print(f'Iteration: {iter}')
            iter += 1
            print(r_norm)
            print(s_norm)

            if r_norm < 10*self.tol and s_norm < self.tol:
                break

            # if r_norm > 10 * s_norm:
            #     for i in range(self.threads):
            #         rhos[i].value *= 2
            # elif s_norm > 10 * r_norm:
            #     for i in range(self.threads):
            #         rhos[i].value /= 2
        P = np.reshape(x_bar[:((self.der + 1) * self.system.dim)**2], ((self.der + 1) * self.system.dim,
                                                                       (self.der + 1) * self.system.dim))
        return P

    def construct_iteratively(self, max_iter):
        states, roi = self.simulate_convergence()
        samples = self.compute_derivatives(states)
        # Construct Lyapunov Function
        prob_lst = [self.construct_prob(samples, roi, thread) for thread in range(self.threads)]
        vars, zs, us, rhos, ext_cons, prob = tuple(map(list, zip(*prob_lst)))

        P = self.run_admm(vars, zs, us, rhos, prob)

        obj, x = self.verify(P)
        num_iter = 0
        ext_r_idx = 0
        ext_nr_idx = 0
        I = np.eye(self.max_iter)
        ext_states = []
        while obj > 1e-4 and num_iter < self.max_iter:
            # Verification not successful: append states to sample set and rerun the optimization
            ext_states += [x]
            cur_ders = np.zeros((self.system.dim, self.der+2))
            cur_ders[:, 0] = x
            for der in range(self.der + 1):
                cur_ders[:, der + 1] = np.array(self.system.derivs[der](x)).flatten()
            cur_level = np.reshape(cur_ders[:, 0:self.der + 1].T, (-1,)).T @ \
                        (P @ np.reshape(cur_ders[:, 0:self.der + 1].T, (-1,)))
            print(cur_level)
            roi_cur = cur_level < 0.99
            # roi += [roi_cur]
            if roi_cur:
                # \dot V(x) < 0
                vec1 = np.reshape(cur_ders[:, 0:self.der + 1].T, (-1,))
                vec2 = np.reshape(cur_ders[:, 1:self.der + 2].T, (-1,))
                alpha_idx = int(np.floor(ext_r_idx/2))
                for ext_con in ext_cons:
                    ext_con["A_r"].value[ext_r_idx, :] = np.hstack((np.kron(vec1, vec2) + np.kron(vec2, vec1), -I[alpha_idx, :]))
                    ext_con["b_r"].value[ext_r_idx] = -self.eps * x.T @ x
                ext_r_idx += 1
                # V(x) < 1 + \alpha
                for ext_con in ext_cons:
                    ext_con["A_r"].value[ext_r_idx, :] = np.hstack((np.kron(vec1, vec1), -I[alpha_idx, :]))
                    ext_con["b_r"].value[ext_r_idx] = 1
                ext_r_idx += 1
            else:
                # V(x) > 1
                vec = np.reshape(cur_ders[:, 0:self.der + 1].T, (-1,))
                for ext_con in ext_cons:
                    ext_con["A_nr"].value[ext_nr_idx, :] = -np.kron(vec, vec)
                    ext_con["b_nr"].value[ext_nr_idx] = -(1+self.delta)
                ext_nr_idx += 1
            P = self.run_admm(vars, zs, us, rhos, prob)

            obj, x = self.verify(P)
            num_iter += 1
        print(repr(P))
        print(format_matrix(P, "bmatrix"))
        print(f'Number of iterations: {num_iter}')
        print(f'Solve time: {self.solve_time}')
        # if self.system.dim <= 2:
        #     self.plot_invariant_set_2d(states, roi, P)
        # elif self.system.dim == 3:
        #     self.plot_invariant_set_3d_v2(P)
        # else:
        #     plot_dims = [[0, 2], [1, 3]]  # [[0, 1], [2, 3], [4, 5], [6, 7], [7, 8]]
        #     for dims in plot_dims:
        #         self.plot_invariant_set_multi(P, dims)
        # plt.show()


def solve(problem, verbose, vars):
    problem.solve(solver='MOSEK', verbose=verbose, ignore_dpp=True)
    return vars.value, problem.solution.attr['solve_time']


if __name__ == "__main__":
    system = Example5D()
    lyapunov = ADMM(der=1, num_samples=9, system=system, eps=1e-2, delta=0.5, verbose=False, slices=1, threads=2,
                    max_iter=40, rho=1, tol=0.001)
    lyapunov.construct_iteratively(42)
