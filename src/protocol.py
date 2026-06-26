import struct

ENDPOINT = "tcp://127.0.0.1:5599"

STATE_FMT = "<d4d4d"
CTRL_FMT = "<2d"

def pack_state(t, qpos, qvel):
    return struct.pack(STATE_FMT, float(t), *(float(v) for v in qpos[:4]), *(float(v) for v in qvel[:4]))

def unpack_state(buf):
    vals = struct.unpack(STATE_FMT, buf)
    return vals[0], list(vals[1:5]), list(vals[5:9])

def pack_ctrl(u):
    return struct.pack(CTRL_FMT, float(u[0]), float(u[1]))

def unpack_ctrl(buf):
    return struct.unpack(CTRL_FMT, buf)
