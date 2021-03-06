#!/usr/bin/env python3
import numpy as np


practice = np.array([
    [0.7, 1.0, 0.12, 1.4, -1.0, 0.14, 3.1, -2.0, 0.18],
    [0.9, 0.9, 0.09, 1.4, -1.3, 0.10, 3.1, -2.7, 0.19],
    ])

round1 = np.array([
    [0.0, 0.0, 0.0, 1.8, 0.5, 0.16, 3.0, -0.9, 0.14],
    [0.0, 0.0, 0.0, 2.4, 0.4, 0.18, 2.7, -0.8, 0.16],
    [0.0, 0.0, 0.0, 2.6, 0.4, 0.14, 2.8, -0.9, 0.13]
    ])


print(np.mean(practice, axis=0))
print(np.mean(round1, axis=0))

