
import os
import sys
sys.path.append('./genCHAO')
import numpy as np
import matplotlib.pyplot as plt
from SR_main import load_uv_field, super_resolution

###################### Lecture file ######################

X, Y, U_matrix, V_matrix = load_uv_field('uv_beta0sigma8av_cv.txt')
print(U_matrix.shape, V_matrix.shape)

# Avec normalisation
#X, Y, U_matrix, V_matrix = load_uv_field('uv_beta0sigma8am_cv.txt', normalize=True)

print('field loaded, moving to super-resolution')

###################### Super-resolution ######################

Super_resolved, U_LR = super_resolution(U_matrix, factor = 32, k_injection = 5)

###################### Plotting ######################

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

ax1.imshow(U_matrix)
ax1.set_title('Original')

ax2.imshow(U_LR)
ax2.set_title('Low Resolution')

ax3.imshow(Super_resolved)
ax3.set_title('Super resolved')

plt.tight_layout()
plt.show()