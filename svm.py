import matplotlib.pyplot as plt
import numpy as np

def generate_dataset():
    np.random.seed(3)
    class1 = []; class2 = []
    label1 = []; label2 =[]
    # define decision line
    def f(x):
        return 1*x
    for _ in range(100):
        x = np.random.rand() * 10
        y = np.random.rand() * 10
        if y-f(x) > 1:
            class1.append([x, y])
            label1.append(1)
        elif y-f(x) < -1:
            class2.append([x, y])
            label2.append(-1)
    x1, y1 = zip(*[data for data in class1])
    x2, y2 = zip(*[data for data in class2])
    plt.plot(x1, y1, 'ro')
    plt.plot(x2, y2, 'bo')
    plt.axis([0,1,0,1])
    plt.show()
    return class1+class2, label1+label2

def select_j(i, m):
    j = i
    while(j == i):
        j = np.random.randint(0, m)
    return j

def clip_alpha(aj, H: float, L:float):
    return min(max(L, aj), H)


def smo_simple(dataset, labels, C, max_iter):
    data = np.array(dataset, dtype=np.float)
    label = np.array(labels, dtype=np.float)
    b = 0
    m, n = np.shape(data)
    alphas = np.zeros(m, dtype=np.float)
    iter = 0
    while iter < max_iter:
        alpha_pair_changed = 0
        for i in range(m):
            x_i, y_i = data[i], label[i]
            fx_i = np.dot(alphas*label, np.dot(data, x_i)) + b
            e_i = fx_i - y_i
            j = select_j(iter, m)
            x_j, y_j = data[j], label[j]
            fx_j = np.dot(alphas*label, np.dot(data, x_j)) + b
            e_j = fx_j - y_j

            # calculate a_j_new
            eta = np.dot(x_i, x_i) + np.dot(x_j, x_j) - 2*np.dot(x_i, x_j)
            if eta <= 0:
                print("eta <= 0, continue")
                continue
            a_i, a_j = alphas[i], alphas[j]
            a_j_new = a_j + y_j * (e_i - e_j) / eta

            # limit a_j_new to [0, C]
            if y_i == y_j:
                L = max(0., a_i + a_j - C)
                H = min(a_i + a_j, C)
            else:
                L = max(0., a_j - a_i)
                H = min(C - a_i + a_j, C)
            if L < H:
                a_j_new = clip_alpha(a_j_new, H, L)
            else:
                print("L >= H. (L, H) =", (L, H))
                continue

            # judge if a_j moves an enough distance
            if abs(a_j_new - a_j) < 0.00001:
                print("a_j has not moved enough, a_j_new - a_j = %f" % (a_j_new - a_j))
                continue

            # calculate a_i_new and update a_i, a_j
            a_i_new = (a_j - a_j_new)*y_i*y_j + a_i
            alphas[i], alphas[j] = a_i_new, a_j_new
            alpha_pair_changed += 1

            #calculate b
            b_i = -e_i + (a_i - a_i_new)*y_i*np.dot(x_i, x_i) + (a_j - a_j_new)*y_j*np.dot(x_i, x_j) + b
            b_j = -e_j + (a_i - a_i_new)*y_i*np.dot(x_i, x_j) + (a_j - a_j_new)*y_j*np.dot(x_j, x_j) + b
            # b = b_i if b_i == b_j else (b_i + b_j)/2
            if 0 < a_i_new < C:
                b = b_i
            elif 0 < a_j_new < C:
                b = b_j
            else:
                b = (b_i + b_j)/2

            print("(a_i, a_j) moved from (%f, %f) to (%f, %f)." % (a_i, a_j, a_i_new, a_j_new))


        if alpha_pair_changed == 0:
            print("Iteration %d of max_iter %d" % (iter+1, max_iter))
            iter += 1
        else:
            iter = 0

    return alphas, b

def get_W(alphas, dataset, label):
    a, x, y = map(np.array, [alphas, dataset, label])
    W = np.dot(a * y, x)
    return W.tolist()


if __name__ == '__main__':
    dataset, label = generate_dataset()
    # print(label)
    alphas, b = smo_simple(dataset, label, C=6, max_iter=40)
    print(alphas)
    W = get_W(alphas, dataset, label)
    print(W, b)

    class1, class2 = [], []
    for data in zip(dataset, label):
        if data[1] == 1.0:
            class1.append(data[0])
        elif data[1] == -1.0:
            class2.append(data[0])
    x11, x12 = zip(*class1)
    x21, x22 = zip(*class2)
    plt.plot(x11, x12, 'ro')
    plt.plot(x21, x22, 'bo')

    x = np.linspace(0, 10, 50)
    y = -(W[0]*x + b)/W[1]
    plt.plot(x, y)
    for i in range(len(dataset)):
        if alphas[i] > 1e-3:
            xi_1, xi_2 = dataset[i][0], dataset[i][1]
            plt.scatter(xi_1, xi_2, s=150, c='none', linewidths=1.5, edgecolors='#1f77b4')
            x = np.linspace(0, 10, 50)
            y = -W[0]/W[1] * x + (xi_2 + W[0]/W[1] * xi_1)
            plt.plot(x, y, 'y--')
    plt.show()

