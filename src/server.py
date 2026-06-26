import os
import numpy as np
import mujoco
import zmq

from controller import Controller
from protocol import ENDPOINT, unpack_state, pack_ctrl

XML = os.path.join(os.path.dirname(__file__), "..", "sim", "dronePendulum.xml")

model = mujoco.MjModel.from_xml_path(XML)
ctrl = Controller(model)

ctx = zmq.Context.instance()
sock = ctx.socket(zmq.REP)
sock.bind(ENDPOINT)

print("controller server up on", ENDPOINT)
print("waiting for run.py ... (Ctrl+C to quit)")

try:
    while True:
        msg = sock.recv()
        if msg == b"BYE":
            sock.send(b"BYE")
            break
        t, qpos, qvel = unpack_state(msg)
        u = ctrl(t, np.array(qpos), np.array(qvel))
        sock.send(pack_ctrl(u))
except KeyboardInterrupt:
    print("\nstopped by user")
finally:
    sock.close(0)
    ctx.term()
