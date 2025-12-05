a = 4
w = 1.0
lr = 0.1

for step in range(5):
    grad = 2 * (w-a)
    w = w - lr * grad
    print(step, w)