import matplotlib.pyplot as plt
import numpy as np

nfeature = 5

org_arr = np.loadtxt("Kalman_project/Data/data_org_harris.csv", delimiter = ',')
id_arr = np.loadtxt("Kalman_project/Data/data_id_harris.csv", delimiter = ',')
mean_arr = np.loadtxt("Kalman_project/Data/data_mean_harris.csv", delimiter = ',')
kalman_arr = np.loadtxt("Kalman_project/Data/data_kalman_harris.csv", delimiter = ',')



id_arr = np.reshape(id_arr, (int(len(id_arr)/nfeature), nfeature))
id_arr = id_arr.T

for i in range(nfeature):
    id_arr[i] = id_arr[i][0]-id_arr[i]

time1 = np.array(range((len(id_arr[0])-len(org_arr)),len(id_arr[0])))
time1 = time1/30


time2 = np.array(range(np.size(id_arr[0])))
time2 = time2/30

fig = plt.figure()
# for i in range(nfeature):
#     plt.plot(time2, id_arr[i], color='r', linewidth=1.5)
plt.plot(time1, org_arr, label = 'standard deviation threshold', color='b', linewidth=1.5)
plt.plot(time2, kalman_arr, label = 'kalmen threshold', color='r', linewidth=1.5)
plt.plot(time2, mean_arr, label = 'mean value', color='g', linewidth=1.5)
plt.legend()
plt.show()
