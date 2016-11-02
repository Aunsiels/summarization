import matplotlib.pyplot as plt
import numpy as np
f = open("loss.csv")
rouge1 = []
rouge2 = []
rouge3 = []
rougel = []
rouges = []
loss = []
i = 0
N = 50

for j in f:
    if i%6 == 0:
        rouge1 += list(map(float, j.split(',')))
    elif i%6 == 1:
        rouge2 += list(map(float, j.split(',')))
    elif i%6 == 2:
        rouge3 += list(map(float, j.split(',')))
    elif i%6 == 3:
        rougel += list(map(float, j.split(',')))
    elif i%6 == 4:
        rouges += list(map(float, j.split(',')))
    elif i%6 == 5:
        loss += list(map(float, j.split(',')))
    i += 1

rouge1 = np.convolve(rouge1, np.ones((N,))/N, mode='valid')
rouge2 = np.convolve(rouge2, np.ones((N,))/N, mode='valid')
rouge3 = np.convolve(rouge3, np.ones((N,))/N, mode='valid')
rougel = np.convolve(rougel, np.ones((N,))/N, mode='valid')
rouges = np.convolve(rouges, np.ones((N,))/N, mode='valid')
loss = np.convolve(loss, np.ones((N,))/N, mode='valid')
plt.plot(rouge1, label="rouge1")
plt.plot(rouge2, label="rouge2")
plt.plot(rouge3, label="rouge3")
plt.plot(rougel, label="rouge-L")
plt.plot(rouges, label="rouge-SU4")
plt.plot(loss, label="loss")
plt.title("loss")
plt.legend()
plt.show()
f.close()
