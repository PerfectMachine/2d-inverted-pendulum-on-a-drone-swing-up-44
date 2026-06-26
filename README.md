# Drone with Pendulum in 2D

A planar two-rotor drone. The simulation and the controller run as two
separate processes communicating over **ZeroMQ**.

Control is a cascade of two laws: energy-based **swing-up** and an analytic
**LQR** near the upright position.

## Structure

```
drone_pendulum/
├── sim/
│   └── dronePendulum.xml    MuJoCo model (drone + pendulum)
└── src/
    ├── controller.py        swing-up + LQR
    ├── protocol.py          ZeroMQ message format
    ├── server.py            REP server: state → thrust
    └── run.py               MuJoCo simulation, REQ client
```

## Model

The system is planar: the body has three degrees of freedom — `slide_x`,
`slide_z` (translation) and `pitch_y` (body tilt θ). The pendulum rotates on a
separate hinge `phi`. It starts hanging down (φ = 0); the goal is to bring it
to φ = π.

## Control

| Mode | When | What it does |
|------|------|--------------|
| **swing-up** | pendulum down | accelerates the drone along `x`, pumping energy up to `E = 2·mgl` |
| **LQR** | `\|φ−π\| < 0.15`, `\|φ̇\| < 2.5` | holds the top, `u = u₀ − K·s` |

Switching uses hysteresis: it falls back to swing-up when `|φ−π| > 0.30`.
LQR state vector: `s = [θ, φ−π, θ̇, φ̇]`.

## Protocol

REQ/REP exchange over `tcp://127.0.0.1:5599`. Each step the simulation sends the
state `(t, qpos, qvel)` and the controller replies with thrust `(F_L, F_R)`.
The controller never touches MuJoCo objects directly — only the data from the
socket.

## The results

The main goal of the project was achieved.

![The result](https://cdn.discordapp.com/attachments/1465441947517714560/1520049136550809610/pendulum.gif?ex=6a3fc793&is=6a3e7613&hm=9615237cb2ee0b89d3297124cdc99e174201405c191dd5acbc62b380c33359fc&)

## How to run?

```bash
pip install -r requirements.txt
```

In two terminals from the `src/` folder:

```bash
python server.py    # first — the controller
python run.py       # then — the simulation
```

`run.py` saves a `result.png` plot on exit.

## Dependencies

`mujoco`, `numpy`, `scipy`, `matplotlib`, `pyzmq`
