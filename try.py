import numpy as np 
import torch
global_mlp_loss = np.array([0.0007631898624822497, 0.004195349756628275, 0.017014864832162857, 0.03371699899435043, 0.051272351294755936])
local_mlp_loss = np.array([0.0006829224876128137,  0.004097772762179375, 0.016549699008464813, 0.033134836703538895, 0.05059770494699478])
mean_y = np.array([0.1041, 0.1862, 0.3099, 0.5042, 0.8898])

global_mlp_loss = np.sqrt(global_mlp_loss) / mean_y
local_mlp_loss = np.sqrt(local_mlp_loss) / mean_y
T = [0.132, 115.623, 1794.756, 4477.035, 11168.001]

import matplotlib.pyplot as plt 
fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(T,global_mlp_loss)
ax.plot(T,local_mlp_loss)
ax.scatter(T,global_mlp_loss, s=20)
ax.scatter(T,local_mlp_loss, s=20)
ax.legend(['Global MLP', 'Local MLP'])
plt.show()