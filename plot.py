import matplotlib.pyplot as plt

gpu_time = [0.21, 1.17, 7.13, 53.99, 427.47]
cpu_time = [15.17, 115.21, 979.42, 30414.15, 244077.27]
htd_time = [0.18, 0.56, 1.63, 5.77, 19.22]
dth_time = [0.13, 0.70, 1.66, 7.57, 57.90]
matrix_sizes = [125, 250, 500, 1000, 2000]

# plt.plot(matrix_sizes, gpu_time)
plt.plot(matrix_sizes, cpu_time)
# plt.plot(matrix_sizes, htd_time)
# plt.plot(matrix_sizes, dth_time)
plt.show()
