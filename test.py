import numpy as np
from sklearn.preprocessing import minmax_scale


def fifo_numpy_1D(fifo_numpy_data, numpy_append_data):
    fifo_numpy_data = np.delete(fifo_numpy_data, 0)
    fifo_numpy_data = np.append(fifo_numpy_data, numpy_append_data)
    return fifo_numpy_data


a = np.array([0.04, 0.02, 0.015, 0.014, 0.011])
# a = minmax_scale(a)
b = np.square(np.subtract(a, a.mean())).mean()
c = np.sqrt(np.mean(a ** 2))
z = b
y = c
print("====================")
print(b)
print(c)
print("====================")

a = np.array([0.02, 0.015, 0.014, 0.011, 0.01])
b = np.square(np.subtract(a, a.mean())).mean()
c = np.sqrt(np.mean(a ** 2))
print("====================")
print(b)
print(c)
print(z-b)
print(y-c)
z = b
y = c
print("====================")
a = np.array([0.015, 0.014, 0.011, 0.01, 0.01])
b = np.square(np.subtract(a, a.mean())).mean()
c = np.sqrt(np.mean(a ** 2))
print("====================")
print(b)
print(c)
print(z-b)
print(y-c)
z = b
y = c
print("====================")
a = np.array([0.014, 0.011, 0.01, 0.01, 0.01])
b = np.square(np.subtract(a, a.mean())).mean()
c = np.sqrt(np.mean(a ** 2))
print("====================")
print(b)
print(c)
print(z-b)
print(y-c)
z = b
y = c
print("====================")
print("====================")
a = np.array([0.011, 0.01, 0.01, 0.01, 0.01])
b = np.square(np.subtract(a, a.mean())).mean()
c = np.sqrt(np.mean(a ** 2))
print("====================")
print(b)
print(c)
print(z-b)
print(y-c)
z = b
y = c
print("====================")
a = np.array([0.01, 0.01, 0.01, 0.01])
b = np.square(np.subtract(a, a.mean())).mean()
c = np.sqrt(np.mean(a ** 2))
print("====================")
print(b)
print(c)
print(z-b)
print(y-c)
z = b
y = c
print("====================")
a = np.array([0.01, 0.01, 0.01, 0.01])
b = np.square(np.subtract(a, a.mean())).mean()
c = np.sqrt(np.mean(a ** 2))
print("====================")
print(b)
print(c)
print(z-b)
print(y-c)
z = b
y = c
print("====================")
a = np.array([0.01, 0.01, 0, 0])
b = np.square(np.subtract(a, a.mean())).mean()
c = np.sqrt(np.mean(a ** 2))
print("====================")
print(b)
print(c)
print(z-b)
print(y-c)
z = b
y = c
print("====================")

_loss_list_5times = np.ones(5, dtype=float)
_loss_list_5times = fifo_numpy_1D(_loss_list_5times, 0.11)