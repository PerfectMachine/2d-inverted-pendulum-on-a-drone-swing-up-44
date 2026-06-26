import os
import time
import numpy as np
import mujoco
import mujoco.viewer
import matplotlib.pyplot as plt

from controller import DronePendulumController

XML = os.path.join(os.path.dirname(__file__), "..", "sim", "dronePendulum.xml")

model = mujoco.MjModel.from_xml_path(XML)
data = mujoco.MjData(model)
mujoco.mj_resetData(model, data)

ctrl = DronePendulumController(model, data)

T_END = 20.0
log = {k: [] for k in ("t", "phi", "th", "uL", "uR", "mode")}

with mujoco.viewer.launch_passive(model, data) as viewer:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "drone")
    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    viewer.cam.azimuth = -90
    viewer.cam.elevation = 0
    viewer.cam.distance = 2.0
    while viewer.is_running() and data.time < T_END:
        step_start = time.time()

        u = ctrl(data.time, data.qpos.copy(), data.qvel.copy())
        data.ctrl[:] = u
        mujoco.mj_step(model, data)

        log["t"].append(data.time)
        log["phi"].append(data.qpos[ctrl.qph] + data.qpos[ctrl.qth])
        log["th"].append(data.qpos[ctrl.qth])
        log["uL"].append(u[0]); log["uR"].append(u[1])
        log["mode"].append(0 if ctrl.mode == "swingup" else 1)

        viewer.cam.lookat[:] = [data.xpos[body_id][0], 0.0, data.xpos[body_id][2]]
        viewer.sync()
        dt = model.opt.timestep - (time.time() - step_start)
        if dt > 0:
            time.sleep(dt)

t = np.array(log["t"])
fig, ax = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
ax[0].plot(t, np.degrees(log["phi"]), label="phi")
ax[0].axhline(180, ls="--", c="r", label="up")
ax[0].set_ylabel("pendulum, °"); ax[0].legend(); ax[0].grid()
ax[1].plot(t, np.degrees(log["th"]), label="pitch, °")
ax[1].set_ylabel("drone"); ax[1].legend(); ax[1].grid()
ax[2].plot(t, log["uL"], label="F_L"); ax[2].plot(t, log["uR"], label="F_R")
ax[2].fill_between(t, 0, 20, where=np.array(log["mode"]) == 1,
                   color="g", alpha=0.1, label="LQR")
ax[2].set_ylabel("thrust, Н"); ax[2].set_xlabel("t, с"); ax[2].legend(); ax[2].grid()
plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), "result.png"), dpi=120)
plt.show()