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
N = 1000
BATCHSIZE = 64

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

rouge1 = list(map(lambda x: x / BATCHSIZE, rouge1))
rouge2 = list(map(lambda x: x / BATCHSIZE, rouge2))
rouge3 = list(map(lambda x: x / BATCHSIZE, rouge3))
rougel = list(map(lambda x: x / BATCHSIZE, rougel))
rouges = list(map(lambda x: x / BATCHSIZE, rouges))

rouge1 = np.convolve(rouge1, np.ones((N,))/N, mode='valid')
rouge2 = np.convolve(rouge2, np.ones((N,))/N, mode='valid')
rouge3 = np.convolve(rouge3, np.ones((N,))/N, mode='valid')
rougel = np.convolve(rougel, np.ones((N,))/N, mode='valid')
rouges = np.convolve(rouges, np.ones((N,))/N, mode='valid')
loss = np.convolve(loss, np.ones((N,))/N, mode='valid')

print("rouge1 last :" + str(rouge1[-1] * 100))
print("rouge2 last :" + str(rouge2[-1] * 100))
print("rouge3 last :" + str(rouge3[-1] * 100))
print("rougel last :" + str(rougel[-1] * 100))
print("rouges last :" + str(rouges[-1] * 100))

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
