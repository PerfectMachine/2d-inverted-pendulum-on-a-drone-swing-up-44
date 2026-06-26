import numpy as np
import mujoco
from scipy.linalg import solve_continuous_are

G = 9.81

def wrap_pi(a):
    return (a + np.pi) % (2.0 * np.pi) - np.pi

class Controller:

    def __init__(self, model, data=None):
        self.model = model
        joint = lambda n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)
        self.px, self.dx = model.jnt_qposadr[joint('slide_x')], model.jnt_dofadr[joint('slide_x')]
        self.pz, self.dz = model.jnt_qposadr[joint('slide_z')], model.jnt_dofadr[joint('slide_z')]
        self.pth, self.dth = model.jnt_qposadr[joint('pitch_y')], model.jnt_dofadr[joint('pitch_y')]
        self.pph, self.dph = model.jnt_qposadr[joint('phi')], model.jnt_dofadr[joint('phi')]

        (self.mass, self.mass_d, self.mass_p, self.length,
         self.inertia_p, self.inertia_d, self.arm) = self.get_params(model)

        self.hover = self.mass * G / 2.0

        self.mgl = self.mass_p * G * self.length
        self.energy_top = 2.0 * self.mgl

        self.k_energy = 20.0
        self.accel_max = 15.0
        self.kp_z, self.kd_z = 4.0, 1.5
        self.kp_th, self.kd_th = 6.0, 0.5
        self.kp_x, self.kd_x = 0.6, 0.8
        self.swing_sign = 1.0

        self.gain = self.build_lqr()
        self.u_min = model.actuator_ctrlrange[:, 0].copy()
        self.u_max = model.actuator_ctrlrange[:, 1].copy()
        self.mode = 'swingup'

    @staticmethod
    def axis_inertia(inertia_diag, iquat, axis_body):
        conj = np.zeros(4); mujoco.mju_negQuat(conj, iquat)
        a = np.zeros(3); mujoco.mju_rotVecQuat(a, axis_body.astype(float), conj)
        return float(np.dot(inertia_diag, a * a))

    def get_params(self, model):
        body = lambda n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n)
        site = lambda n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, n)
        bd, bp = body('drone'), body('pendulum')

        mass_d = float(model.body_mass[bd])
        mass_p = float(model.body_mass[bp])
        mass = mass_d + mass_p

        axis = np.array([0.0, 1.0, 0.0])

        length = float(np.linalg.norm(model.body_ipos[bp]))
        inertia_p_com = self.axis_inertia(model.body_inertia[bp], model.body_iquat[bp], axis)
        inertia_p = inertia_p_com + mass_p * length * length

        inertia_d_com = self.axis_inertia(model.body_inertia[bd], model.body_iquat[bd], axis)
        ipos = model.body_ipos[bd]
        inertia_d = inertia_d_com + mass_d * (ipos[0] ** 2 + ipos[2] ** 2)

        arm = float(abs(model.site_pos[site('thrustR')][0]))
        return mass, mass_d, mass_p, length, inertia_p, inertia_d, arm

    def build_lqr(self):
        m, mp, l = self.mass, self.mass_p, self.length
        Jp, Id, d, g = self.inertia_p, self.inertia_d, self.arm, G
        det = m * Jp - mp**2 * l**2
        a = m * mp * g * l / det

        A = np.array([
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0],
            [-a,  a, 0.0, 0.0]
        ])
        B = np.array([
            [0.0, 0.0],
            [0.0, 0.0],
            [d/Id, -d/Id],
            [0.0, 0.0]
        ])

        Q = np.diag([4.0, 120.0, 1.0, 12.0])
        R = np.diag([0.01, 0.01])

        P = solve_continuous_are(A, B, Q, R)
        return np.linalg.solve(R, B.T @ P)

    def __call__(self, t, qpos, qvel):
        x, z = qpos[self.px], qpos[self.pz]
        th = qpos[self.pth]
        psi = qpos[self.pph]
        vx, vz = qvel[self.dx], qvel[self.dz]
        vth = qvel[self.dth]
        vpsi = qvel[self.dph]

        ang = th + psi
        vang = vth + vpsi

        err = abs(wrap_pi(ang - np.pi))

        if self.mode == 'swingup' and err < 0.15 and abs(vang) < 2.5:
            self.mode = 'lqr'
        elif self.mode == 'lqr' and err > 0.30:
            self.mode = 'swingup'

        if self.mode == 'lqr':
            d_ang = wrap_pi(ang - np.pi)
            s = np.array([th, d_ang, vth, vang])
            u = np.array([self.hover, self.hover]) - self.gain @ s
        else:
            u = self.swing_up(x, z, th, ang, vx, vz, vth, vang)

        return np.clip(u, self.u_min, self.u_max)

    def swing_up(self, x, z, th, ang, vx, vz, vth, vang):
        energy = 0.5 * self.inertia_p * vang * vang + self.mgl * (1.0 - np.cos(ang))

        ax = self.swing_sign * self.k_energy * (self.energy_top - energy) * vang * np.cos(ang)
        if abs(vang) < 0.05 and (self.energy_top - energy) > 0.05:
            ax += 1.5
        ax += -self.kp_x * x - self.kd_x * vx
        ax = np.clip(ax, -self.accel_max, self.accel_max)

        th_des = np.clip(np.arctan2(ax, G), -0.6, 0.6)

        az = self.kp_z * (0.0 - z) - self.kd_z * vz
        c = max(np.cos(th), 0.5)
        thrust = self.mass * (G + az) / c

        torque = self.kp_th * (th_des - th) - self.kd_th * vth
        fl = 0.5 * (thrust + torque / self.arm)
        fr = 0.5 * (thrust - torque / self.arm)
        return np.array([fl, fr])
