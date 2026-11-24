import numpy as np
import matplotlib.pyplot as plt
from SR_main import load_uv_field, super_resolution

###################### Lecture file ######################

X, Y, U_matrix, V_matrix = load_uv_field('uv_beta0sigma8av_cv.txt')
print(U_matrix.shape, V_matrix.shape)

# Avec normalisation
#X, Y, U_matrix, V_matrix = load_uv_field('uv_beta0sigma8am_cv.txt', normalize=True)

###################### Super-resolution ######################

Super_resolved_test = super_resolution(U_matrix, factor = 32, k_injection = 5)

###################### Plotting ######################

plt.imshow(Super_resolved_test)