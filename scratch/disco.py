#!/usr/bin/env python3
"""
Displays robot fetch at a disco party.
"""
from mujoco_py import load_model_from_path, MjSim, MjViewer
from mujoco_py.modder import TextureModder
import os

model = load_model_from_path("../xmls/fetch/main.xml")
pool = MjSimPool(model)
modder = TextureModder(sim)

pool = MjSimPool([MjSim(model) for _ in range(20)])
for i, sim in enumerate(pool.sims):
    sim.data.qpos[:] = 0.0
    sim.data.qvel[:] = 0.0
    sim.data.ctrl[:] = i

# Advance all 20 simulations 100 times.
for _ in range(100):
    pool.step()

import ipdb; ipdb.set_trace()

for i, sim in enumerate(pool.sims):
    print("%d-th sim qpos=%s" % (i, str(sim.data.qpos)))


while True:
    for name in sim.model.geom_names:
        modder.rand_all(name)

    sim.forward()
    cam_img = sim.render(640, 360)[::-1, :, :]
    import ipdb; ipdb.set_trace()
