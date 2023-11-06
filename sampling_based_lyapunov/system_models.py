import numpy as np
import casadi as ca
import scipy as sp
from sampling_based_lyapunov.util import quaternion_multiply
import os
import pickle
import matplotlib.pyplot as plt


class System:
    def __init__(self, simtime, bounds):
        self.dim = 0  # no. of states
        self.derivs: [ca.Function, ca.Function, ca.Function, ca.Function] = self.model()
        self.simtime = simtime  # simulation horizon
        self.bounds = bounds  # bounds of sampled domain, 2 x dim
        self.name = 'general_system'

    def model(self):
        '''
        Construct model: f(x) and its first 3 derivatives. Set dimension of the state space
        '''
        self.dim = 0
        return 4 * []

    def simulate(self, x0, equilibrium=None, t_eval=None):
        if equilibrium is None:
            equilibrium = np.zeros(self.dim)
        f = self.derivs[0]
        dxdt = lambda t, x: np.array(f(x)).flatten()
        diverged = lambda t, x: (x - equilibrium).T @ (x - equilibrium) - 100
        diverged.terminal = True
        diverged.direction = 1
        converged = lambda t, x: (x - equilibrium).T @ (x - equilibrium) - 0.0095**2
        converged.terminal = True
        converged.direction = -1

        sol = sp.integrate.solve_ivp(dxdt, [0, self.simtime], x0, method='RK45', t_eval=t_eval,
                                     events=[diverged, converged])
        return sol

    def generate_grid(self, num_samples, num_cores):
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument('slice', type=int)
        args = parser.parse_args()
        num_slice = args.slice
        if not isinstance(num_samples, list):
            num_samples = self.dim * [num_samples]

        X = np.meshgrid(*[np.linspace(self.bounds[0, i], self.bounds[1, i], num_samples[i])
                          for i in range(self.dim)], indexing='ij')

        states = np.stack(X, axis=-1)
        states = states.reshape(-1, self.dim).T

        states = np.array_split(states, num_cores, axis=1)[num_slice]
        roi = states.shape[1] * [[]]
        for i in range(states.shape[1]):
            roi[i] = np.linalg.norm(self.simulate(states[:, i]).y[:, -1]) < 0.01
            if not i % 500:
                print(f'Progress: {i/states.shape[1]*100} %')

        abs_path = os.path.dirname(os.path.abspath(__file__))
        pickle_path = os.path.join(abs_path, "..", "pickle")
        filename = self.name + f'_{num_samples[0]}' + f'_{num_slice}' + '.pickle'
        filename = os.path.join(pickle_path, filename)

        with open(filename, 'wb') as file:
            pickle.dump([states, roi], file)

        print(len(roi))
        print(sum(roi))

    def merge_grid_files(self, num_samples, num_cores):
        path = self.name + f'_{num_samples}'
        states_lst = num_cores * [[]]
        roi = []
        abs_path = os.path.dirname(os.path.abspath(__file__))
        pickle_path = os.path.join(abs_path, "..", "pickle")
        for i in range(num_cores):
            filename = os.path.join(pickle_path, path + f'_{i}.pickle')
            with open(filename, 'rb') as file:
                data = pickle.load(file)
            states_, roi_ = data[0], data[1]
            states_lst[i] = states_
            roi += roi_

        states = np.hstack(states_lst)
        eps = 1e-5
        self.roi = roi
        roi = [False if any(np.abs(np.abs(states[:, idx]) - self.bounds[1, :]) < eps) else elem for idx, elem in
               enumerate(roi)]

        abs_path = os.path.dirname(os.path.abspath(__file__))
        pickle_path = os.path.join(abs_path, "..", "pickle")
        with open(os.path.join(pickle_path, path + '.pickle'), 'wb') as file:
            pickle.dump([states, roi, self.roi], file)

    def plot_grid_points(self, num_samples, dims):
        path = self.name + f'_{num_samples}'
        abs_path = os.path.dirname(os.path.abspath(__file__))
        pickle_path = os.path.join(abs_path, "..", "pickle")
        with open(os.path.join(pickle_path, path + '.pickle'), 'rb') as file:
            data = pickle.load(file)
        states, roi = data[0], data[1]
        not_roi = [not elem for elem in roi]

        if self.dim != 3:
            all_idx = [i for i in range(self.dim)]
            cur_idx = dims
            other_idx = [elem for elem in all_idx if elem not in cur_idx]
            idx_slice = np.sum(states[other_idx, :] == 0, axis=0) == self.dim - 2

            from itertools import compress
            roi = list(compress(roi, idx_slice))
            states = states[:, idx_slice]

            plt.figure()
            plt.plot(states[cur_idx[0], roi], states[cur_idx[1], roi], 'go')
            plt.plot(states[cur_idx[0], not_roi], states[cur_idx[1], not_roi], 'ro')
            plt.show()
        else:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

            # Plot grid points inside and outside roi
            stable_states = states[:, roi]
            unstable_states = states[:, not_roi]
            ax.scatter(stable_states[0, :], stable_states[1, :], stable_states[2, :], s=1.5, c='g')
            # ax.scatter(unstable_states[0, :], unstable_states[1, :], unstable_states[2, :], s=1.5, c='r')
            plt.show()


class VanDerPol(System):
    def __init__(self):
        super().__init__(simtime=15, bounds=np.array([[-4, -10], [4, 10]]))
        self.name = 'vanderpol'

    def model(self):
        self.dim = 2
        x1, x2 = ca.MX.sym('x1'), ca.MX.sym('x2')
        x = ca.vertcat(x1, x2)
        f_expr = ca.vertcat(x2, -2 * x1 - 3 * x2 + x1 * x1 * x2)
        df_expr = ca.jacobian(f_expr, x) @ f_expr
        ddf_expr = ca.jacobian(df_expr, x) @ f_expr
        dddf_expr = ca.jacobian(ddf_expr, x) @ f_expr
        f = ca.Function('f', [x], [f_expr])
        df = ca.Function('df', [x], [df_expr])
        ddf = ca.Function('ddf', [x], [ddf_expr])
        dddf = ca.Function('dddf', [x], [dddf_expr])
        return [f, df, ddf, dddf]


class Trophino143(System):
    def __init__(self):
        super().__init__(simtime=15, bounds=np.array([[-2.5, -2.5], [2.5, 2.5]]))
        self.name = 'trophino143'

    def model(self):
        self.dim = 2
        x1, x2 = ca.MX.sym('x1'), ca.MX.sym('x2')
        x = ca.vertcat(x1, x2)
        f_expr = ca.vertcat(-x1 + x1 * x2**2, x1 - x2 + x1**2 * x2 - x1 * x2**2)
        df_expr = ca.jacobian(f_expr, x) @ f_expr
        ddf_expr = ca.jacobian(df_expr, x) @ f_expr
        dddf_expr = ca.jacobian(ddf_expr, x) @ f_expr
        f = ca.Function('f', [x], [f_expr])
        df = ca.Function('df', [x], [df_expr])
        ddf = ca.Function('ddf', [x], [ddf_expr])
        dddf = ca.Function('dddf', [x], [dddf_expr])
        return [f, df, ddf, dddf]


class Trophino144(System):
    def __init__(self):
        super().__init__(simtime=10, bounds=np.array([[-7, -7], [7, 7]]))
        self.name = 'trophino144'

    def model(self):
        self.dim = 2
        x1, x2 = ca.MX.sym('x1'), ca.MX.sym('x2')
        x = ca.vertcat(x1, x2)
        f_expr = ca.vertcat(-x1 + 2 * (x1**2) * x2, -x2)
        df_expr = ca.jacobian(f_expr, x) @ f_expr
        ddf_expr = ca.jacobian(df_expr, x) @ f_expr
        dddf_expr = ca.jacobian(ddf_expr, x) @ f_expr
        f = ca.Function('f', [x], [f_expr])
        df = ca.Function('df', [x], [df_expr])
        ddf = ca.Function('ddf', [x], [ddf_expr])
        dddf = ca.Function('dddf', [x], [dddf_expr])
        return [f, df, ddf, dddf]


class Trophino137(System):
    def __init__(self):
        super().__init__(simtime=100, bounds=np.array([[-2, -7], [7, 2]]))
        self.name = 'trophino137'

    def model(self):
        self.dim = 2
        x1, x2 = ca.MX.sym('x1'), ca.MX.sym('x2')
        x = ca.vertcat(x1, x2)
        eps1 = 0.5
        eps2 = 0.5
        f_expr = ca.vertcat(x2 + eps1 * x1/(x2**2 + 1), -x1 - x2 + eps2 * x1**2)
        df_expr = ca.jacobian(f_expr, x) @ f_expr
        ddf_expr = ca.jacobian(df_expr, x) @ f_expr
        dddf_expr = ca.jacobian(ddf_expr, x) @ f_expr
        f = ca.Function('f', [x], [f_expr])
        df = ca.Function('df', [x], [df_expr])
        ddf = ca.Function('ddf', [x], [ddf_expr])
        dddf = ca.Function('dddf', [x], [dddf_expr])
        return [f, df, ddf, dddf]

    def test(self):
        x0 = np.array([2, -3])
        sim = self.simulate(x0)

        plt.figure()
        plt.plot(sim.t, sim.y.T)
        plt.show()


class Trophino145(System):
    def __init__(self):
        super().__init__(simtime=100, bounds=np.array([[-4, -5, -8.5], [4, 5, 7]]))
        self.name = 'trophino145'

    def model(self):
        self.dim = 3
        x1, x2, x3 = ca.MX.sym('x1'), ca.MX.sym('x2'), ca.MX.sym('x3')
        x = ca.vertcat(x1, x2, x3)
        eps1 = 0.5
        eps2 = 0.5
        eps3 = 0.5
        f_expr = ca.vertcat(x2 + eps3 * x3 + eps1 * x1/(x2**2 + 1), -x1 - x2 + eps2 * x1**2,
                            eps3 * (-2 * x1 - 2 * x3 - x1**2))
        df_expr = ca.jacobian(f_expr, x) @ f_expr
        ddf_expr = ca.jacobian(df_expr, x) @ f_expr
        dddf_expr = ca.jacobian(ddf_expr, x) @ f_expr
        f = ca.Function('f', [x], [f_expr])
        df = ca.Function('df', [x], [df_expr])
        ddf = ca.Function('ddf', [x], [ddf_expr])
        dddf = ca.Function('dddf', [x], [dddf_expr])
        return [f, df, ddf, dddf]

    def test(self):
        x0 = np.array([1, -1, -2])
        sim = self.simulate(x0)

        plt.figure()
        plt.plot(sim.t, sim.y.T)
        plt.show()


class Example4D(System):
    def __init__(self):
        super().__init__(simtime=100, bounds=np.array([[-4, -10, -2, -7], [4, 10, 7, 2]]))
        self.name = 'example4d'

    def model(self):
        self.dim = 4
        x1, x2, x3, x4 = ca.MX.sym('x1'), ca.MX.sym('x2'), ca.MX.sym('x3'), ca.MX.sym('x4')
        x = ca.vertcat(x1, x2, x3, x4)

        f1_expr = ca.vertcat(x2, -2 * x1 - 3 * x2 + x1 * x1 * x2 - x4)
        eps1 = 0.5
        eps2 = 0.5
        f2_expr = ca.vertcat(x4 + eps1 * x3 / (x4 ** 2 + 1), -x3 - x4 + eps2 * x3 ** 2)
        f_expr = ca.vertcat(f1_expr, f2_expr) 

        df_expr = ca.jacobian(f_expr, x) @ f_expr
        ddf_expr = ca.jacobian(df_expr, x) @ f_expr
        dddf_expr = ca.jacobian(ddf_expr, x) @ f_expr
        f = ca.Function('f', [x], [f_expr])
        df = ca.Function('df', [x], [df_expr])
        ddf = ca.Function('ddf', [x], [ddf_expr])
        dddf = ca.Function('dddf', [x], [dddf_expr])
        return [f, df, ddf, dddf]

    def test(self):
        x0 = np.array([2, -2, -1, 1])
        sim = self.simulate(x0)
 
        plt.figure()
        plt.plot(sim.t, sim.y.T)
        plt.show()


class Example5D(System):
    def __init__(self):
        super().__init__(simtime=100, bounds=np.array([[-4, -10, -4, -5, -8.5], [4, 10, 4, 5, 7]]))
        self.name = 'example5d'

    def model(self):
        self.dim = 5
        x1, x2, z1, z2, z3 = ca.MX.sym('x1'), ca.MX.sym('x2'), ca.MX.sym('x3'), ca.MX.sym('x4'), ca.MX.sym('x5')
        x = ca.vertcat(x1, x2, z1, z2, z3)

        f1_expr = ca.vertcat(x2, -2 * x1 - 3 * x2 + x1 * x1 * x2 - z2)
        eps1 = 0.5
        eps2 = 0.5
        eps3 = 0.5
        f2_expr = ca.vertcat(z2 + eps3 * z3 + eps1 * z1/(z2**2 + 1), -z1 - z2 + eps2 * z1**2,
                                eps3 * (-2 * z1 - 2 * z3 - z1**2))
        f_expr = ca.vertcat(f1_expr, f2_expr)

        df_expr = ca.jacobian(f_expr, x) @ f_expr
        ddf_expr = ca.jacobian(df_expr, x) @ f_expr
        dddf_expr = ca.jacobian(ddf_expr, x) @ f_expr
        f = ca.Function('f', [x], [f_expr])
        df = ca.Function('df', [x], [df_expr])
        ddf = ca.Function('ddf', [x], [ddf_expr])
        dddf = ca.Function('dddf', [x], [dddf_expr])
        return [f, df, ddf, dddf]

    def test(self):
        x0 = np.array([2, -2, -1, 1, 1])
        sim = self.simulate(x0)

        plt.figure()
        plt.plot(sim.t, sim.y.T)
        plt.show()


class AnghelA(System):
    def __init__(self):
        super().__init__(simtime=100, bounds=np.array([[-4, -0.75, -4, -0.75], [4, 0.75, 4, 0.75]]))
        self.name = 'anghela'

    def model(self):
        self.dim = 4
        x1_, x2_, x3_, x4_ = ca.MX.sym('x1'), ca.MX.sym('x2'), ca.MX.sym('x3'), ca.MX.sym('x4')
        x = ca.vertcat(x1_, x2_, x3_, x4_)
        xs = np.array([0.02, 0, 0.06, 0])
        x1 = x1_ + xs[0]
        x2 = x2_ + xs[1]
        x3 = x3_ + xs[2]
        x4 = x4_ + xs[3]
        from sampling_based_lyapunov.fit_poly import fit_poly_1d, fit_poly_2d
        deg = 9
        coeff_2d = fit_poly_2d(x=np.linspace(-4, 4, 200), y=np.linspace(-4, 4, 200), f=np.sin, deg=deg)
        sin_x1_x3 = ca.sum2(ca.sum1(ca.horzcat(*[ca.vertcat(*[coeff_2d[j, i] * x3**i * x1**j for i in range(deg)]) for j in range(deg)])))
        deg = 5
        coeff_1d = fit_poly_1d(x=np.linspace(-4, 4, 200), f=np.sin, deg=5)
        sin_x1 = ca.sum2(ca.horzcat(*[coeff_1d[i] * x1**i for i in range(deg+1)]))
        sin_x3 = ca.sum2(ca.horzcat(*[coeff_1d[i] * x3**i for i in range(deg+1)]))
        f_expr = ca.vertcat(x2, -sin_x1 - 0.5 * sin_x1_x3 - 0.4 * x2,
                        x4, -0.5 * sin_x3 - 0.5 * -sin_x1_x3 - 0.5 * x4 + 0.05)
        # f_expr = ca.vertcat(x2, -ca.sin(x1) - 0.5 * ca.sin(x1 - x3) - 0.4 * x2,
        #                     x4, -0.5 * ca.sin(x3) - 0.5 * ca.sin(x3 - x1) - 0.5 * x4 + 0.05)
        df_expr = ca.jacobian(f_expr, x) @ f_expr
        ddf_expr = ca.jacobian(df_expr, x) @ f_expr
        dddf_expr = ca.jacobian(ddf_expr, x) @ f_expr
        f = ca.Function('f', [x], [f_expr])
        df = ca.Function('df', [x], [df_expr])
        ddf = ca.Function('ddf', [x], [ddf_expr])
        dddf = ca.Function('dddf', [x], [dddf_expr])
        return [f, df, ddf, dddf]

    def test(self):
        x0 = np.array([1, -0.1, -1, 0.1])
        sim = self.simulate(x0)

        plt.figure()
        plt.plot(sim.t, sim.y.T)
        plt.show()


class QuadAttitude(System):
    '''
    States: three angles and three angular velocities in the body frame
    '''
    def __init__(self):
        upper_bounds = np.array([1.5, 1.5, 1.5, 10, 10, 10])
        bounds = np.reshape(np.hstack((-upper_bounds, upper_bounds)), (2, 6))
        super().__init__(simtime=100, bounds=bounds)
        self.name = 'quadattitude'

    def model(self):
        self.dim = 6
        J = np.diag([1.5e-3, 1.5e-3, 2.6e-3])
        J_inv = np.linalg.inv(J)
        qx, qy, qz, qw, omx, omy, omz, taux, tauy, tauz = ca.MX.sym('qx'), ca.MX.sym('qy'), ca.MX.sym('qz'), \
                                                          ca.MX.sym('qw'), ca.MX.sym('omx'), ca.MX.sym('omy'), \
                                                          ca.MX.sym('omz'), ca.MX.sym('taux'), ca.MX.sym('tauy'), \
                                                          ca.MX.sym('tauz')
        # \dot{q} = 1/2 q (x) w
        x_open = ca.vertcat(qw, qx, qy, qz, omx, omy, omz)
        om = ca.vertcat(omx, omy, omz)
        tau = ca.vertcat(taux, tauy, tauz)
        f1 = 1 / 2 * quaternion_multiply((qw, qx, qy, qz), (0, omx, omy, omz))
        f2 = J_inv @ tau - J_inv @ (ca.cross(om, J @ om))
        f_open = ca.vertcat(f1, f2)
        f_open_fn = ca.Function('f_ol', [ca.vertcat(x_open, tau)], [f_open])
        dfdx_ol = ca.Function('dfdx_ol', [ca.vertcat(x_open, tau)], [ca.jacobian(f_open, x_open)])
        dfdu_ol = ca.Function('dfdu_ol', [ca.vertcat(x_open, tau)], [ca.jacobian(f_open, tau)])
        # Linearize system around the origin
        x0 = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        A = np.array(dfdx_ol(x0))[1:, 1:]
        B = np.array(dfdu_ol(x0))[1:, :]
        # Compute LQR
        Q = np.eye(6)
        R = 100 * np.eye(3)
        S = sp.linalg.solve_continuous_are(A, B, Q, R)
        K = np.linalg.inv(R) @ B.T @ S
        K = np.hstack((np.zeros((3, 1)), K))

        # Construct closed loop
        x = ca.MX.sym('x_cl', 7)
        u = -K @ x
        u_max = 0.01
        u = ca.vertcat(*[ca.fmin(ca.fmax(u[i], -u_max), u_max) for i in range(3)])
        # u = u_max*(2/(1 + np.exp(-2*(u/u_max))) - 1)

        f_expr = f_open_fn(ca.vertcat(x, u))
        df_expr = ca.jacobian(f_expr, x) @ f_expr
        ddf_expr = ca.jacobian(df_expr, x) @ f_expr
        dddf_expr = ca.jacobian(ddf_expr, x) @ f_expr
        f = ca.Function('f', [x], [f_expr])
        df = ca.Function('df', [x], [df_expr])
        ddf = ca.Function('ddf', [x], [ddf_expr])
        dddf = ca.Function('dddf', [x], [dddf_expr])
        return [f, df, ddf, dddf]

    def simulate(self, x0, equilibrium=None, t_eval=None, ):
        if equilibrium is None:
            equilibrium = np.zeros(self.dim)
        f = self.derivs[0]
        dxdt = lambda t, x: np.array(f(x)).flatten()
        diverged = lambda t, x: (x - equilibrium).T @ (x - equilibrium) - 100
        diverged.terminal = True
        diverged.direction = 1
        converged = lambda t, x: (x - equilibrium).T @ (x - equilibrium) - 0.05**2
        converged.terminal = True
        converged.direction = -1

        quat_deform = lambda t, x: np.abs(x[:4].T @ x[:4] - 1) - 0.05
        quat_deform.terminal = True
        quat_deform.direction = 1

        sol_lst = []
        sol = sp.integrate.solve_ivp(dxdt, [0, self.simtime], x0, method='RK45', t_eval=t_eval,
                                     events=[diverged, converged, quat_deform])
        sol_lst += [sol]
        # If quaternion norm is far from 1, reset simulation
        while sol.t_events[2].shape[0] > 0:
            t0 = sol.t_events[2][0]
            x0 = sol.y_events[2][0]
            x0[0:4] *= 1 / np.linalg.norm(x0[0:4])
            sol = sp.integrate.solve_ivp(dxdt, [t0, self.simtime], x0, method='RK45', t_eval=t_eval,
                                         events=[diverged, converged, quat_deform])
            sol_lst += [sol]
        return sol_lst

    def test(self):
        q0 = np.random.rand(4)
        q0 *= 1 / np.linalg.norm(q0)
        x0 = np.hstack((q0, 10*np.random.rand(3)))
        sim_lst = self.simulate(x0, equilibrium=np.array([1, 0, 0, 0, 0, 0, 0]))

        t = np.hstack([sim.t for sim in sim_lst])
        y = np.hstack([sim.y for sim in sim_lst])
        plt.figure()
        plt.plot(t, y.T)
        plt.show()


class Quadrotor(System):
    '''
    States: position, velocity and Euler angles
    '''
    def __init__(self):
        upper_bounds = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.3, 0.3, 0.3])
        bounds = np.reshape(np.hstack((-upper_bounds, upper_bounds)), (2, 9))
        super().__init__(simtime=100, bounds=bounds)
        self.name = 'quadrotor'

    def model(self):
        self.dim = 9
        x, y, z, vx, vy, vz, phi, theta, psi, F, taux, tauy, tauz = ca.MX.sym('x'), ca.MX.sym('y'), ca.MX.sym('z'), \
                                                                    ca.MX.sym('vx'), ca.MX.sym('vy'), ca.MX.sym('vz'), \
                                                                    ca.MX.sym('phi'), ca.MX.sym('theta'), ca.MX.sym('psi'), \
                                                                    ca.MX.sym('F'), ca.MX.sym('taux'), ca.MX.sym('tauy'),\
                                                                    ca.MX.sym('tauz')
        # \dot{q} = 1/2 q (x) w
        x_open = ca.vertcat(x, y, z, vx, vy, vz, phi, theta, psi)
        tau = ca.vertcat(taux, tauy, tauz)

        R_lst = [[ca.cos(psi) * ca.cos(theta), ca.cos(psi) * ca.sin(phi) * ca.sin(theta) - ca.cos(phi) * ca.sin(psi),
                  ca.sin(phi) * ca.sin(psi) + ca.cos(phi) * ca.cos(psi) * ca.sin(theta)],
                 [ca.cos(theta) * ca.sin(psi), ca.cos(phi) * ca.cos(psi) + ca.sin(phi) * ca.sin(theta) * ca.sin(psi),
                  ca.cos(phi) * ca.sin(theta) * ca.sin(psi) - ca.cos(psi) * ca.sin(phi)],
                 [-ca.sin(theta), ca.sin(phi) * ca.cos(theta), ca.cos(phi) * ca.cos(theta)]]
        R = ca.vertcat(ca.horzcat(*R_lst[0]), ca.horzcat(*R_lst[1]), ca.horzcat(*R_lst[2]))

        invQ_lst = [[1, ca.sin(phi) * ca.tan(theta), ca.cos(phi) * ca.tan(theta)], [0, ca.cos(phi), -ca.sin(phi)],
                    [0, ca.sin(phi) / ca.cos(theta), ca.cos(phi) / ca.cos(theta)]]
        invQ = ca.vertcat(ca.horzcat(*invQ_lst[0]), ca.horzcat(*invQ_lst[1]), ca.horzcat(*invQ_lst[2]))

        f1 = ca.vertcat(vx, vy, vz)
        f2 = R @ ca.vertcat(0, 0, F) - np.array([0, 0, 9.81])
        f3 = invQ @ tau

        u_open = ca.vertcat(F, tau)

        f_open = ca.vertcat(f1, f2, f3)
        f_open_fn = ca.Function('f_ol', [ca.vertcat(x_open, u_open)], [f_open])
        dfdx_ol = ca.Function('dfdx_ol', [ca.vertcat(x_open, u_open)], [ca.jacobian(f_open, x_open)])
        dfdu_ol = ca.Function('dfdu_ol', [ca.vertcat(x_open, u_open)], [ca.jacobian(f_open, u_open)])
        # Linearize system around the origin
        x0 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 9.81, 0, 0, 0])
        A = np.array(dfdx_ol(x0))
        B = np.array(dfdu_ol(x0))
        # Compute LQR
        Q = np.eye(9)
        R = 10 * np.eye(4)
        S = sp.linalg.solve_continuous_are(A, B, Q, R)
        K = np.linalg.inv(R) @ B.T @ S

        # Construct closed loop
        x = ca.MX.sym('x_cl', 9)
        u0 = np.array([9.81, 0, 0, 0])
        u = u0 - K @ x

        # F_max = 30
        # u[0] = ca.fmin(ca.fmax(u[0], 0), F_max)
        # tau_max = 0.1
        # u = ca.vertcat(*([u[0]] + [ca.fmin(ca.fmax(u[i], -tau_max), tau_max) for i in range(1, 4)]))

        f_expr = f_open_fn(ca.vertcat(x, u))
        df_expr = ca.jacobian(f_expr, x) @ f_expr
        ddf_expr = ca.jacobian(df_expr, x) @ f_expr
        dddf_expr = ca.jacobian(ddf_expr, x) @ f_expr
        f = ca.Function('f', [x], [f_expr])
        df = ca.Function('df', [x], [df_expr])
        ddf = ca.Function('ddf', [x], [ddf_expr])
        dddf = ca.Function('dddf', [x], [dddf_expr])

        self.K = K

        return [f, df, ddf, dddf]

    def test(self):
        r0 = 0.1 * (np.random.rand(3) - 0.5)
        v0 = 0.1 * (np.random.rand(3) - 0.5)
        lam0 = 0.6 * (np.random.rand(3) - 0.5)
        x0 = np.hstack((r0, v0, lam0))
        sim = self.simulate(x0)

        plt.figure()
        plt.plot(sim.t, sim.y.T)

        u = - self.K @ sim.y
        u[0, :] = np.clip(u[0, :] - 9.81, 0, 30)
        u[1:, :] = np.clip(u[1:, :], -0.1, 0.1)*1000
        plt.figure()
        plt.plot(sim.t, u.T)
        plt.show()


if __name__ == "__main__":
    system = Example5D()
    # system.test()
    # system.generate_grid(num_samples=9, num_cores=16)  # needs to be run by 'run_grid_evaluation.sh'
    system.merge_grid_files(num_samples=9, num_cores=16)
    # system.plot_grid_points(num_samples=17, dims=[2, 3])