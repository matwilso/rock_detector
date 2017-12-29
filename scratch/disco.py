#!/usr/bin/env python3
"""
Displays robot fetch at a disco party.
"""
from mujoco_py import load_model_from_path, MjSim, MjViewer
from mujoco_py.modder import TextureModder
import os

model = load_model_from_path("../xmls/fetch/main.xml")
sim = MjSim(model)
modder = TextureModder(sim)


while True:
    for name in sim.model.geom_names:
        modder.rand_all(name)

    sim.forward()
    cam_img = sim.render(640, 360)[::-1, :, :]
    import ipdb; ipdb.set_trace()
