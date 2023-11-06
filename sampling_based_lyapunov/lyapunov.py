import matplotlib.pyplot as plt
import cvxpy as cvx
from sampling_based_lyapunov.util import format_matrix
import numpy as np
import casadi as ca
from sampling_based_lyapunov.system_models import System, VanDerPol, Trophino143, Trophino144, Trophino137, Quadrotor,\
    AnghelA, Trophino145, Example4D, Example5D
import os
import pickle
import time


class Lyapunov:
    def __init__(self, der, num_samples, system: System, eps, delta, verbose):
        self.der = der  # no. of derivatives in LF (der=0: V(x) = x.T @ P @ x)
        self.num_samples = num_samples  # no. of samples along each dimension
        self.system = system
        self.eps = eps
        self.delta = delta
        self.verbose = verbose
        self.solve_time = 0

    def simulate_convergence(self):
        '''
        Simulates the system dynamics on a grid to obtain the set of stable initial conditions
        '''
        # If simulation is already saved, just read the pickle
        abs_path = os.path.dirname(os.path.abspath(__file__))
        pickle_path = os.path.join(abs_path, "..", "pickle")
        filename = self.system.name + f'_{self.num_samples}' + '.pickle'
        filename = os.path.join(pickle_path, filename)
        if os.path.exists(filename):
            with open(filename, 'rb') as file:
                data = pickle.load(file)
            states = data[0]
            roi = data[1]
            self.roi = data[2]
            if self.system.dim == 2:
                roi = [False if idx < 2 * self.num_samples or
                                idx % self.num_samples == 0 or
                                (idx + 1) % self.num_samples == 0 or
                                (idx + 2) % self.num_samples == 0 or
                                (idx + self.num_samples - 1) % self.num_samples == 0 or
                                idx > self.num_samples * (self.num_samples - 2) else elem for idx, elem in
                       enumerate(roi)]
            else:
                eps = 1e-5
                roi = [False if any(np.abs(np.abs(states[:, idx]) - self.system.bounds[1, :]) < eps) else elem for idx, elem in
                       enumerate(roi)]
            return states, roi

        # X1, X2 = np.meshgrid(np.linspace(-4, 4, 30), np.linspace(-10, 10, 30), indexing='ij')
        # states = np.stack((X1, X2), axis=-1)
        # states = states.reshape(-1, 2).T

        X = np.meshgrid(*[np.linspace(self.system.bounds[0, i], self.system.bounds[1, i], self.num_samples)
                            for i in range(self.system.dim)], indexing='ij')

        states = np.stack(X, axis=-1)
        states = states.reshape(-1, self.system.dim).T
        roi = [np.linalg.norm(self.system.simulate(x0).y[:, -1]) < 0.01 for x0 in states.T]
        self.roi = roi
        if self.system.dim == 2:
            roi = [False if idx < 2 * self.num_samples or
                            idx % self.num_samples == 0 or
                            (idx + 1) % self.num_samples == 0 or
                            (idx + 2) % self.num_samples == 0 or
                            (idx + self.num_samples - 1) % self.num_samples == 0 or
                            idx > self.num_samples * (self.num_samples - 2) else elem for idx, elem in
                   enumerate(roi)]
        else:
            eps = 1e-5
            roi = [False if any(np.abs(np.abs(states[:, idx]) - self.system.bounds[1, :]) < eps) else elem for idx, elem in
                   enumerate(roi)]
        not_roi = [not elem for elem in roi]

        plt.figure()
        plt.plot(states[0, roi], states[1, roi], 'go')
        plt.plot(states[0, not_roi], states[1, not_roi], 'ro')
        plt.show()

        with open(filename, 'wb') as file:
            pickle.dump([states, roi, self.roi], file)

        return states, roi
    
    def simulate_convergence_v2(self, dim):
        self.num_samples = 50
        X = np.meshgrid(*[np.linspace(self.system.bounds[0, i], self.system.bounds[1, i], self.num_samples)
                            for i in dim], indexing='ij')

        states = np.stack(X, axis=-1)
        states = states.reshape(-1, len(dim)).T
        states_full = np.zeros((self.system.dim, states.shape[1]))
        states_full[dim, :] = states
        roi = [np.linalg.norm(self.system.simulate(x0).y[:, -1]) < 0.01 for x0 in states_full.T]
        self.roi = roi
        not_roi = [not elem for elem in roi]

        return states, roi

    def compute_derivatives(self, states):
        samples = np.zeros((self.system.dim, states.shape[1], self.der + 2))
        samples[:, :, 0] = states
        for der in range(self.der + 1):
            samples[:, :, der + 1] = np.array(self.system.derivs[der](states))
        return samples

    def find_lyapunov_lasso(self, samples, roi):
        n = self.system.dim

        P_dim = ((self.der + 1) * n) ** 2
        P = cvx.Variable(((self.der + 1) * n, (self.der + 1) * n), symmetric=True)

        samples_roi = samples[:, roi, :]
        not_roi = [not elem for elem in roi]
        samples_not_roi = samples[:, not_roi, :]

        num_roi = samples_roi.shape[1]
        num_not_roi = len(roi) - num_roi
        alpha = cvx.Variable(num_roi)

        # \dot{V}(x) < 0 within the ROI
        A1 = np.zeros((num_roi, P_dim + num_roi))
        b1 = np.zeros(num_roi)
        I = np.eye(num_roi)

        for i in range(num_roi):
            vec1 = np.reshape(samples_roi[:, i, 0:self.der + 1].T, (-1,))
            vec2 = np.reshape(samples_roi[:, i, 1:self.der + 2].T, (-1,))
            A1[i, :] = np.hstack((np.kron(vec1, vec2) + np.kron(vec2, vec1), -I[i, :]))
            b1[i] = -self.eps * samples_roi[:, i, 0].T @ samples_roi[:, i, 0]

        # V(x) > 1 outside the roi
        A2 = np.zeros((num_not_roi, P_dim))
        for i in range(num_not_roi):
            vec = np.reshape(samples_not_roi[:, i, 0:self.der + 1].T, (-1,))
            A2[i, :] = -np.kron(vec, vec)
        b2 = -(1+self.delta)*np.ones(num_not_roi)

        # A = np.vstack((A1, A2))
        # b = np.hstack((b1, b2))

        constraints = [A2 @ cvx.vec(P) <= b2]

        # V(x) < 1 + \alpha within the roi
        A3 = np.zeros((num_roi, P_dim + num_roi))
        I = np.eye(num_roi)
        for i in range(num_roi):
            vec = np.reshape(samples_roi[:, i, 0:self.der + 1].T, (-1,))
            A3[i, :] = np.hstack((np.kron(vec, vec), -I[i, :]))
        b3 = np.ones(num_roi)

        A = np.vstack((A1, A3))
        b = np.hstack((b1, b3))

        constraints += [A @ cvx.hstack((cvx.vec(P), alpha)) <= b]

        # V(x) > 0 within the roi
        constraints += [-A3[:, :P_dim] @ cvx.vec(P) <= 0]

        # alpha >= 0
        constraints += [alpha >= np.zeros(num_roi)]

        # Cost: minimize L1 norm of \alpha
        prob = cvx.Problem(cvx.Minimize(np.ones(num_roi) @ alpha), constraints)

        prob.solve(solver='MOSEK', verbose=self.verbose)
        return P.value, alpha.value

    def find_lyapunov_lasso_slicing(self, samples, roi, slices):
        n = self.system.dim

        P_dim = ((self.der + 1) * n) ** 2
        P = cvx.Variable(((self.der + 1) * n, (self.der + 1) * n), symmetric=True)

        samples_roi = samples[:, roi, :]
        not_roi = [not elem for elem in roi]
        samples_not_roi = samples[:, not_roi, :]

        num_roi = samples_roi.shape[1]
        num_not_roi = len(roi) - num_roi
        alpha = cvx.Variable(num_roi)
        constraints = [P >= np.zeros(((self.der + 1) * n, (self.der + 1) * n))]
        # constraints = []

        # Split samples_roi for tractable memory allocation
        samples_roi_lst = np.array_split(samples_roi, slices, axis=1)
        roi_lst = [elem.tolist() for elem in np.array_split(np.array(roi), slices)]

        len_slices = np.hstack((0, np.cumsum([elem.shape[1] for elem in samples_roi_lst])))

        # \dot{V}(x) < 0 within the ROI
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
        b2 = -(1+self.delta)*np.ones(num_not_roi)
        constraints += [A2 @ cvx.vec(P) <= b2]

        # V(x) < 1 + \alpha within the roi
        for slice in range(slices):
            start_idx = len_slices[slice]
            end_idx = len_slices[slice+1]
            dim = end_idx - start_idx
            A3 = np.zeros((dim, P_dim + dim))
            I = np.eye(dim)
            for i in range(start_idx, end_idx):
                vec = np.reshape(samples_roi[:, i, 0:self.der + 1].T, (-1,))
                A3[i-start_idx, :] = np.hstack((np.kron(vec, vec), -I[i-start_idx, :]))
            b3 = np.ones(dim)
            constraints += [A3 @ cvx.hstack((cvx.vec(P), alpha[start_idx:end_idx])) <= b3]

            # V(x) > 0 within the roi
            constraints += [-A3[:, :P_dim] @ cvx.vec(P) <= 0]

        # alpha >= 0
        constraints += [alpha >= np.zeros(num_roi)]

        # Cost: minimize L1 norm of \alpha
        prob = cvx.Problem(cvx.Minimize(np.ones(num_roi) @ alpha), constraints)

        prob.solve(solver='MOSEK', verbose=self.verbose)
        return P.value, alpha.value

    def find_lyapunov_lasso_precompile(self, samples, roi, max_iter):
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
        alpha = cvx.Variable(num_roi + max_iter)
        # constraints = [P >= np.zeros(((self.der + 1) * n, (self.der + 1) * n))]
        constraints = []

        # \dot{V}(x) < alpha within the ROI
        A1 = np.zeros((num_roi, P_dim + num_roi))
        b1 = np.zeros(num_roi)
        I = np.eye(num_roi)

        for i in range(num_roi):
            vec1 = np.reshape(samples_roi[:, i, 0:self.der + 1].T, (-1,))
            vec2 = np.reshape(samples_roi[:, i, 1:self.der + 2].T, (-1,))
            A1[i, :] = np.hstack((np.kron(vec1, vec2) + np.kron(vec2, vec1), -I[i, :]))
            b1[i] = -self.eps * samples_roi[:, i, 0].T @ samples_roi[:, i, 0]

        # V(x) > 1 + \delta outside the roi
        A2 = np.zeros((num_not_roi, P_dim))
        for i in range(num_not_roi):
            vec = np.reshape(samples_not_roi[:, i, 0:self.der + 1].T, (-1,))
            A2[i, :] = -np.kron(vec, vec)
        b2 = -(1 + self.delta) * np.ones(num_not_roi)

        constraints += [A2 @ cvx.vec(P) <= b2]

        # V(x) < 1 + \alpha within the roi
        A3 = np.zeros((num_roi, P_dim + num_roi))
        I = np.eye(num_roi)
        for i in range(num_roi):
            vec = np.reshape(samples_roi[:, i, 0:self.der + 1].T, (-1,))
            A3[i, :] = np.hstack((np.kron(vec, vec), -I[i, :]))
        b3 = np.ones(num_roi)

        A = np.vstack((A1, A3))
        b = np.hstack((b1, b3))

        constraints += [A @ cvx.hstack((cvx.vec(P), alpha[:num_roi])) <= b]

        # V(x) > 0 within the roi
        constraints += [-A3[:, :P_dim] @ cvx.vec(P) <= 0]

        # alpha >= 0
        constraints += [alpha >= np.zeros(num_roi + max_iter)]

        # Unused alphas are zero
        constraints += [alpha[num_roi:] <= alpha_bound_ext]
        # \dot V(x) < alpha, V(x) < 1 + alpha within the roi
        constraints += [A_ext_roi @ cvx.hstack((cvx.vec(P), alpha[num_roi:])) <= b_ext_roi]
        # V(x) > 1 outside the roi
        constraints += [A_ext_n_roi @ cvx.vec(P) <= b_ext_n_roi]

        # Cost: minimize L1 norm of \alpha
        prob = cvx.Problem(cvx.Minimize(np.ones(num_roi+max_iter) @ alpha), constraints)

        return P, alpha, prob, A_ext_roi, b_ext_roi, A_ext_n_roi, b_ext_n_roi, alpha_bound_ext

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

    def verify(self, P):
        opti = ca.Opti()
        x = opti.variable(self.system.dim)

        vec1 = ca.vertcat(*([x] + [self.system.derivs[i](x) for i in range(self.der)]))
        vec2 = ca.vertcat(*[self.system.derivs[i](x) for i in range(self.der+1)])
        opti.subject_to([vec1.T @ P @ vec1 <= 1.0])
        T = np.diag(1/self.system.bounds[1, :])**2
        # opti.subject_to([x.T @ T @ x >= 0.004])
        obj = vec1.T @ P @ vec2 + vec2.T @ P @ vec1
        opti.minimize(-obj)
        # opti.set_initial(x, -0.03)
        opti.set_initial(x[0], -0.2)
        opti.set_initial(x[1], 0.2)
        if self.verbose:
            opts = {}
        else:
            opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.sb': 'yes'}  # Supress output
        opti.solver("ipopt", opts)
        try:
            start_time = time.time()
            sol = opti.solve()
            end_time = time.time()
            self.solve_time += end_time - start_time
        except RuntimeError:
            print('Verifier failed for P=')
            print(repr(P))
            return -1, 42
        else:
            obj2 = vec1.T @ P @ vec1
            opti.minimize(obj2)
            sol2 = opti.solve()
            print(f'Objective value (if negative, then valid LF): {sol.value(obj)}')
            print(sol.value(x))
            print(f'Minimum of V(x): {sol2.value(obj2)}')
            # print(sol.value(x).T @ P @ sol.value(x).T)
            return sol.value(obj), sol.value(x)

    def plot_invariant_set_2d(self, states, roi, P):
        '''
        Plot stable and unstable states together with the 1 Level set of the LF. Currently works only for 2 dimensional
        systems
        '''
        plt.rcParams.update({'font.size': 10, 'lines.linewidth': 1.2, 'lines.markersize': 1.5})
        fig = plt.figure(1, figsize=(3, 3))

        roi = self.roi

        # Plot grid points inside and outside roi
        not_roi = [not elem for elem in roi]
        stable_states = states[:, roi]
        unstable_states = states[:, not_roi]
        plt.plot(stable_states[0, :], stable_states[1, :], 'go', label='$\mathcal{X}_0$')
        plt.plot(unstable_states[0, :], unstable_states[1, :], 'ro', label='$\mathcal{X}_\infty$')

        # Plot 1 level set of LF
        X1, X2 = np.meshgrid(*[np.linspace(1.1*self.system.bounds[0, i], 1.1*self.system.bounds[1, i], 300)
                               for i in range(self.system.dim)], indexing='ij')

        it = np.nditer([X1, X2], flags=['multi_index'])
        z = np.zeros_like(X1)

        while not it.finished:
            idx = it.multi_index
            # x = np.array([x[idx] for x in X])
            x = np.array([X1[idx], X2[idx]])
            vec = np.hstack([x] + [np.array(self.system.derivs[i](x)).flatten() for i in range(self.der)])
            z[idx] = vec @ (P @ vec)
            it.iternext()

        # levels = [0.01, 0.1, 0.3, 0.5, 0.7, 0.95, 1]
        levels = [1]
        CS = plt.contour(X1, X2, z, levels, colors='k')
        # CS.collections[0].set_label('$\mathcal{V}(x) = 1$')
        # plt.clabel(CS, inline=True, fontsize=10)
        # Thicken the one contour.
        # zc = CS.collections[-1]
        # plt.setp(zc, linewidth=2)
        plt.xlabel('$x_1$')
        plt.ylabel('$x_2$')

        fig.subplots_adjust(left=0.2,
                            bottom=0.2,
                            right=0.9,
                            top=0.9
                            )
        # plt.show()

    def plot_invariant_set_3d(self, states, roi, P):
        '''
        Plot stable and unstable states together with the 1 Level set of the LF. Currently works only for 2 dimensional
        systems
        '''
        plt.rcParams.update({'font.size': 10, 'lines.linewidth': 1.2, 'lines.markersize': 1.5})
        fig = plt.figure(1)

        ax = fig.add_subplot(111, projection='3d')

        # Plot grid points inside and outside roi
        not_roi = [not elem for elem in roi]
        stable_states = states[:, roi]
        unstable_states = states[:, not_roi]
        ax.scatter(stable_states[0, :], stable_states[1, :], 0*stable_states[0, :], s=1.5, c='g')
        ax.scatter(unstable_states[0, :], unstable_states[1, :], 0*unstable_states[0, :], s=1.5, c='r')

        # Plot 1 level set of LF
        X1, X2 = np.meshgrid(*[np.linspace(self.system.bounds[0, i], self.system.bounds[1, i], 100)
                               for i in range(self.system.dim)], indexing='ij')

        it = np.nditer([X1, X2], flags=['multi_index'])
        z1 = np.zeros_like(X1)
        z2 = np.zeros_like(X1)

        while not it.finished:
            idx = it.multi_index
            # x = np.array([x[idx] for x in X])
            x = np.array([X1[idx], X2[idx]])
            vec = np.hstack([x] + [np.array(self.system.derivs[i](x)).flatten() for i in range(self.der)])
            vec_der = np.hstack([np.array(self.system.derivs[i](x)).flatten() for i in range(self.der+1)])
            z1[idx] = vec @ (P @ vec)
            z2[idx] = vec @ (P @ vec_der) + vec_der @ (P @ vec)
            if z1[idx] > 1.05:
                z1[idx] = np.nan
                z2[idx] = np.nan
            it.iternext()

        ax.plot_surface(X1, X2, z1, cmap='rainbow')

        CS = plt.contour(X1, X2, z1, levels=[0.01, 0.1, 0.3, 0.5, 0.7, 1], cmap='rainbow', zdir='z', offset=0)

        # Thicken the one contour.
        # zc = CS.collections[-1]
        # plt.setp(zc, linewidth=2)

        # plt.clabel(CS, inline=True, fontsize=10)
        # plt.colorbar(CS)
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        ax.set_zlabel('V(x)')

        # Plot Lie derivative
        # fig2 = plt.figure(2)
        # ax2 = fig2.add_subplot(111, projection='3d')
        # ax2.plot_surface(X1, X2, z2, cmap='viridis')

        plt.show()

    def plot_invariant_set_multi(self, P, dims, fig_num):
        '''
        Plot the 1 Level set of the LF. Works for more than 2 dimensional systems
        '''
        plt.rcParams.update({'font.size': 10, 'lines.linewidth': 1.2, 'lines.markersize': 1.5})
        fig = plt.figure(fig_num, figsize=(3, 3))
        # Plot 1 level set of LF
        X1, X2 = np.meshgrid(*[np.linspace(1.1 * self.system.bounds[0, i], 1.1 * self.system.bounds[1, i], 300)
                               for i in dims], indexing='ij')
        it = np.nditer([X1, X2], flags=['multi_index'])
        z = np.zeros_like(X1)

        while not it.finished:
            idx = it.multi_index
            x = np.zeros(self.system.dim)
            x[dims] = np.array([X1[idx], X2[idx]])
            vec = np.hstack([x] + [np.array(self.system.derivs[i](x)).flatten() for i in range(self.der)])
            z[idx] = vec @ (P @ vec)
            it.iternext()

        # levels = [0.01, 0.1, 0.3, 0.5, 0.7, 0.95, 1]
        levels = [1]
        CS = plt.contour(X1, X2, z, levels, colors='k')
        # plt.clabel(CS, inline=True, fontsize=10)
        # Thicken the one contour.
        # zc = CS.collections[-1]
        # plt.setp(zc, linewidth=2)
        plt.xlabel(f'$x_{dims[0]+1}$')
        plt.ylabel(f'$x_{dims[1]+1}$')

        fig.subplots_adjust(left=0.2,
                            bottom=0.2,
                            right=0.9,
                            top=0.9
                            )
        # plt.show()

    def plot_invariant_set_3d_v2(self, P):
        # Plot 1 level set of LF
        X1, X2, X3 = np.meshgrid(*[np.linspace(1.1*self.system.bounds[0, i], 1.1*self.system.bounds[1, i], 100)
                               for i in range(self.system.dim)], indexing='ij')
        spacing = [(1.1*self.system.bounds[1, i] - 1.1*self.system.bounds[0, i])/99 for i in range(self.system.dim)]

        it = np.nditer([X1, X2, X3], flags=['multi_index'])
        z = np.zeros_like(X1)

        # Calculate the corresponding 4D function values
        while not it.finished:
            idx = it.multi_index
            x = np.array([X1[idx], X2[idx], X3[idx]])
            vec = np.hstack([x] + [np.array(self.system.derivs[i](x)).flatten() for i in range(self.der)])
            z[idx] = vec @ (P @ vec)
            it.iternext()

        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        from skimage import measure

        # Use marching cubes to obtain the surface mesh of these ellipsoids
        verts, faces, normals, values = measure.marching_cubes(z, 1, spacing=tuple(spacing), step_size=6)
        verts = verts + 1.1 * self.system.bounds[0, :]

        # Display resulting triangular mesh using Matplotlib
        fig = plt.figure(1, figsize=(3, 3))
        ax = fig.add_subplot(111, projection='3d')

        # Fancy indexing: `verts[faces]` to generate a collection of triangles
        from matplotlib.colors import LightSource
        ls = LightSource(azdeg=150, altdeg=65)
        mesh = Poly3DCollection(verts[faces], shade=True, facecolors=[0.0, 0.6, 0.0, 0.75], linewidths=(0.3,), lightsource=ls)
        mesh.set_edgecolor('k')
        ax.add_collection3d(mesh)

        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")
        ax.set_zlabel("$x_3$")

        ax.set_xlim(1.1*self.system.bounds[0, 0], 1.1*self.system.bounds[1, 0])
        ax.set_ylim(1.1*self.system.bounds[0, 1], 1.1*self.system.bounds[1, 1])
        ax.set_zlim(1.1*self.system.bounds[0, 2], 1.1*self.system.bounds[1, 2])

        ax.view_init(elev=15, azim=250)
        ax.set_box_aspect([1, 1, 1])
        # ax.legend()
        plt.tight_layout()
        # plt.show()

    def construct(self):
        states, roi = self.simulate_convergence()
        samples = self.compute_derivatives(states)
        P, alpha = self.find_lyapunov_lasso(samples, roi)
        self.verify(P)
        print(repr(P))
        # print(repr(np.linalg.eigvals(P)))
        # print(repr(alpha))

        if self.system.dim <= 2:
            self.plot_invariant_set_2d(states, roi, P)
        else:
            plot_dims = [[0, 1], [2, 3], [4, 5], [6, 7], [7, 8]]
            for dims in plot_dims:
                self.plot_invariant_set_multi(P, dims)
        plt.show()

    def construct_iteratively(self, max_iter):
        states, roi = self.simulate_convergence()
        samples = self.compute_derivatives(states)
        # Construct Lyapunov Function
        slices = 1
        P_opt, alpha_opt, prob, A_r, b_r, A_nr, b_nr, alpha_bound, alpha_coef = (
            self.find_lyapunov_lasso_precompile_slicing(samples, roi, slices, max_iter))
        A_init = np.zeros((2 * max_iter, max_iter + (self.system.dim * (self.der + 1))**2))
        b_init = np.zeros(2 * max_iter)
        A_r.value = A_init
        b_r.value = b_init
        A_nr.value = A_init[:, :(self.system.dim * (self.der + 1))**2]
        b_nr.value = b_init
        alpha_bound.value = np.zeros(max_iter)
        alpha_idx = 0
        alpha_coef.value = np.hstack((np.ones(sum(roi)+alpha_idx), np.zeros(max_iter-alpha_idx)))
        prob.solve(solver='MOSEK', verbose=self.verbose)
        self.solve_time += prob.solution.attr['solve_time']
        # Verify
        P = P_opt.value
        obj, x = self.verify(P)
        num_iter = 0
        ext_r_idx = 0
        ext_nr_idx = 0
        I = np.eye(max_iter)
        ext_states = []
        while obj > 0 and num_iter < max_iter:
            # Verification not successful: append states to sample set and rerun the optimization
            # samples = np.append(samples, np.zeros((self.system.dim, 1, self.der+2)), axis=1)
            # states = np.append(states, np.expand_dims(x, axis=1), axis=1)
            # samples[:, -1, 0] = x
            ext_states += [x]
            cur_ders = np.zeros((self.system.dim, self.der+2))
            cur_ders[:, 0] = x
            for der in range(self.der + 1):
                cur_ders[:, der + 1] = np.array(self.system.derivs[der](x)).flatten()
            cur_level = np.reshape(cur_ders[:, 0:self.der + 1].T, (-1,)).T @ \
                      (P @ np.reshape(cur_ders[:, 0:self.der + 1].T, (-1,)))
            print(cur_level)
            roi_cur = cur_level < 0.9
            # roi += [roi_cur]
            if roi_cur:
                # \dot V(x) < 0
                vec1 = np.reshape(cur_ders[:, 0:self.der + 1].T, (-1,))
                vec2 = np.reshape(cur_ders[:, 1:self.der + 2].T, (-1,))
                alpha_idx = int(np.floor(ext_r_idx/2))
                alpha_coef.value = np.hstack((np.ones(sum(roi) + alpha_idx), np.zeros(max_iter - alpha_idx)))
                A_r.value[ext_r_idx, :] = np.hstack((np.kron(vec1, vec2) + np.kron(vec2, vec1), -I[alpha_idx, :]))
                b_r.value[ext_r_idx] = -self.eps * x.T @ x
                alpha_bound.value[alpha_idx] = 1e2  # kind of dummy, bigger enough than zero
                ext_r_idx += 1
                # V(x) < 1 + \alpha
                A_r.value[ext_r_idx, :] = np.hstack((np.kron(vec1, vec1), -I[alpha_idx, :]))
                b_r.value[ext_r_idx] = 1
                ext_r_idx += 1
            else:
                # V(x) > 1
                vec = np.reshape(cur_ders[:, 0:self.der + 1].T, (-1,))
                A_nr.value[ext_nr_idx, :] = -np.kron(vec, vec)
                b_nr.value[ext_nr_idx] = -(1+self.delta)
                ext_nr_idx += 1
            prob.solve(solver='MOSEK', warm_start=True, verbose=self.verbose)
            self.solve_time += prob.solution.attr['solve_time']
            P = P_opt.value
            # P, alpha = self.find_lyapunov_lasso(samples, roi)

            obj, x = self.verify(P)
            num_iter += 1
        
        print(repr(P))
        # print(repr(np.linalg.eigvals(P)))
        # print(repr(alpha_opt.value))

        print(format_matrix(P, "bmatrix"))
        print(f'Number of iterations: {num_iter}')
        print(f'Solve time: {self.solve_time}')
        self.P = P
        if self.system.dim <= 2:
            self.plot_invariant_set_2d(states, roi, P)
        elif self.system.dim == 3:
            self.plot_invariant_set_3d_v2(P)
        else:
            plot_dims = [[0, 1], [2, 3], [2, 4]]  # [[0, 1], [2, 3], [4, 5], [6, 7], [7, 8]]
            # for dims in plot_dims:
            #     self.plot_invariant_set_multi(P, dims)
        # plt.show()

    def find_volume_doa(self):
        self.num_samples = 9
        states, roi = self.simulate_convergence()

        vol_roi = np.prod(np.array([elem[1] - elem[0] for elem in self.system.bounds.T]))
        vol_doa = sum(roi) / len(roi) * vol_roi
        print(f'Volume of DOA: {vol_doa}')

    def plot_doa(self, dim=None, fig_num=1):
        if dim is not None:
            states, roi = self.simulate_convergence_v2(dim)
            dim = 2
        else:
            states, roi = self.simulate_convergence()
            dim = self.system.dim
        if dim == 2:
            fig = plt.figure(fig_num)
            levels = [0.5]
            states = np.reshape(states.T, (self.num_samples, self.num_samples, 2))
            X1 = states[:, :, 0]
            X2 = states[:, :, 1]
            z = np.reshape(np.array(self.roi), (self.num_samples, self.num_samples))
            CS = plt.contour(X1, X2, z, levels, colors='b', linestyles='--')
            # CS.collections[0].set_label('DOA')
            # plt.clabel(CS, inline=True, fontsize=10)
            # Thicken the one contour.
            # zc = CS.collections[-1]
            # plt.setp(zc, linewidth=2)

            plt.plot([0, 0], [0, 0], 'k-', label='$\mathcal{V}(x) = 1$')
            plt.plot([0, 0], [0, 0], 'b--', label='DOA')
            plt.xlabel('$x_1$')
            plt.ylabel('$x_2$')
            # plt.legend()

            fig.subplots_adjust(left=0.2,
                                bottom=0.2,
                                right=0.9,
                                top=0.9
                                )
            # plt.show()
        elif dim == 3:
            # states = np.reshape(states.T, (self.num_samples, self.num_samples, self.num_samples, 3))
            # X1 = states[:, :, :, 0]
            # X2 = states[:, :, :, 1]
            # X3 = states[:, :, :, 2]
            z = np.reshape(np.array(roi), (self.num_samples, self.num_samples, self.num_samples))

            spacing = [(self.system.bounds[1, i] - self.system.bounds[0, i]) / (self.num_samples-1) for i in
                       range(self.system.dim)]

            from mpl_toolkits.mplot3d.art3d import Poly3DCollection
            from skimage import measure

            # Use marching cubes to obtain the surface mesh of these ellipsoids
            verts, faces, normals, values = measure.marching_cubes(z, 0.5, spacing=tuple(spacing), step_size=3)
            verts = verts + self.system.bounds[0, :]

            # Display resulting triangular mesh using Matplotlib
            fig = plt.figure(1)
            ax = fig.get_axes()[0]

            # Fancy indexing: `verts[faces]` to generate a collection of triangles
            from matplotlib.colors import LightSource
            ls = LightSource(azdeg=150, altdeg=65)
            mesh = Poly3DCollection(verts[faces], shade=True, facecolors=[0.0, 0.0, 0.0, 0.0], linewidths=(0.3,),
                                    lightsource=ls)
            mesh.set_edgecolor('b')
            ax.add_collection3d(mesh)

            # ax.set_xlabel("$x_1$")
            # ax.set_ylabel("$x_2$")
            # ax.set_zlabel("$x_3$")
            #
            # ax.set_xlim(1.1 * self.system.bounds[0, 0], 1.1 * self.system.bounds[1, 0])
            # ax.set_ylim(1.1 * self.system.bounds[0, 1], 1.1 * self.system.bounds[1, 1])
            # ax.set_zlim(1.1 * self.system.bounds[0, 2], 1.1 * self.system.bounds[1, 2])

            # ax.view_init(elev=15, azim=250)
            # ax.set_box_aspect([1, 1, 1])
            # # ax.legend()
            # plt.tight_layout()
            # plt.show()

    def find_volume_level_set(self):
        X = np.meshgrid(*[np.linspace(self.system.bounds[0, i], self.system.bounds[1, i], 25)
                               for i in range(self.system.dim)], indexing='ij')

        it = np.nditer(X, flags=['multi_index'])
        z = np.zeros_like(X[0])

        while not it.finished:
            idx = it.multi_index
            x = np.array([x[idx] for x in X])
            # x = np.array([X1[idx], X2[idx]])
            vec = np.hstack([x] + [np.array(self.system.derivs[i](x)).flatten() for i in range(self.der)])
            z[idx] = vec @ (self.P @ vec)
            it.iternext()

        vol_roi = np.prod(np.array([elem[1] - elem[0] for elem in self.system.bounds.T]))
        vol_level = z[z<=1].size / z.size * vol_roi
        print(f'Volume of sublevel set: {vol_level}')

if __name__ == "__main__":
    system = VanDerPol()
    lyapunov = Lyapunov(der=2, num_samples=30, system=system, eps=1e-3, delta=0.15, verbose=False)
    lyapunov.construct_iteratively(10)
    lyapunov.plot_doa()
    plt.show()
