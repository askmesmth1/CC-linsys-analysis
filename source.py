import numpy as np
import csv

with open("output.csv", 'w', newline='') as file:
    writer = csv.writer(file)

with open("input.csv", newline='') as file:
    data = [list(map(int, rec)) for rec in csv.reader(file, delimiter=',')]
    X = np.array(data[0])
    U = np.array(data[1])
    N = np.array(data[2])
    Y = np.array(data[3])
    T = np.array(data[4])

A = np.random.sample(size=(X.size, X.size))
B = np.random.sample(size=(X.size, U.size))
E = np.random.sample(size=(X.size, N.size))
C = np.random.sample(size=(Y.size, X.size))
D = np.random.sample(size=(Y.size, U.size))
F = np.random.sample(size=(Y.size, N.size))

for i in range(T[0]):
    N = np.random.normal(size=(N.shape))
    X = np.matmul(A, X) + np.matmul(B, U) + np.matmul(E, N)
    N = np.random.normal(size=(N.shape))
    Y = np.matmul(C, X) + np.matmul(D, U) + np.matmul(F, N)
    print('state:', X)
    print('output:', Y)

    with open("output.csv", 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(Y)