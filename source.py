import numpy as np
import matplotlib.pyplot as plt
import csv
'''
with open("output.csv", 'w', newline='') as file:
    writer = csv.writer(file)

with open("input.csv", newline='') as file:
    data = [list(map(int, rec)) for rec in csv.reader(file, delimiter=',')]
    X = np.array(data[0])
    U = np.array(data[1])
    N = np.array(data[2])
    Y = np.array(data[3])
    T = np.array(data[4])
print(X, '\n', U, '\n', N, '\n', Y, '\n', T, '\n')
'''

def system_is_stable(a):
    result = set()
    plt.subplot(2, 1, 1)
    for x in range(len(a)):
        plt.plot([0, a[x].real], [0, a[x].imag], 'bo--')
        if abs(a[x].real) < 1 and abs(a[x].imag) < 1:
            result.add('stable')
        else:
            result.add('unstable')
    if 'unstable' in result:
        return False
    else:
        return True

'''Input'''
with open("input.txt", newline='', encoding='utf-8') as file:
    data = [rec.replace(' ', '')[3:-1] for rec in file.readlines()]
    X = np.fromstring(data[0], dtype=float, sep=',')
    U = np.fromstring(data[1], dtype=float, sep=',')
    N = np.fromstring(data[2], dtype=float, sep=',')
    Y = np.fromstring(data[3], dtype=float, sep=',')
    T = int(data[4])

'''Matrices'''
A = np.random.normal(size=(X.size, X.size))
A = A/5
#A = np.array([[-0.1, 0.3, 0.05, 0.6, 0.8, 0.06], [1, 0.1, -1, 1, 0.2, 1], [-1, -1, -0.6, 1, 0.3, 0.5], [0.07, -0.09, 1, 0.2, -0.5, -0.8], [1, -1, 0.7, 0.9, -0.9, -0.3], [1, 1, 0.8, 0.5, 0.4, 0.2]])
#A = A/3
B = np.random.normal(size=(X.size, U.size))
E = np.random.normal(size=(X.size, N.size))
C = np.random.normal(size=(Y.size, X.size))
D = np.random.normal(size=(Y.size, U.size))
F = np.random.normal(size=(Y.size, N.size))

'''Eigenvalues of square matrix A'''
p = np.linalg.eigvals(A)

X_files, Y_files, Y_temp, time, max_val = [], [], [], [], 0.0

'''Main cycle'''
for i in range(T):
    N = np.random.normal(size=(N.shape))
    X = np.matmul(A, X) + np.matmul(B, U)# + np.matmul(E, N)
    N = np.random.normal(size=(N.shape))
    Y = np.matmul(C, X) + np.matmul(D, U)# + np.matmul(F, N)

    if Y[1] > max_val:
        max_val = Y[1]
    elif max_val == 0.0:
        max_val = Y[1]

    time.append(i)
    X_files.append(X)
    Y_files.append(Y)
    Y_temp.append(Y[1])
    #print('state:', X)
    #print('output:', Y)

'''Overshoot and setting time'''
overshoot = abs((max_val - Y_temp[-1]) / Y_temp[-1])
for element in reversed(Y_temp):
    if abs((element - Y_temp[-1]) / Y_temp[-1]) > 0.05:
        setting_time = Y_temp.index(element)
        break

'''Results'''
if system_is_stable(p):
    print('Система устойчива')
    print('overshoot:', overshoot * 100, '%', '\nmaximum value:', max_val, '\nfinal value:', Y_temp[-1], '\nsetting time:', setting_time)
else:
    print('Система неустойчива')
plt.subplot(2,1,2)
plt.plot(time, Y_temp)
plt.grid()
plt.show()

'''Output'''
'''with open('output.txt', 'w', encoding='utf-8') as file:
    for line in Y_files:
        file.write('Y = '+ str(line) + '\n')
'''
'''
    with open("output.csv", 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(Y)
'''