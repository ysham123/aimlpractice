x,y = 3, 10
w1, b, lr = 1.0, 0.0, 0.01

for step in range(5):
    y_hat = w1 * x + b #forward 
    loss = (y_hat - y) ** 2 #loss

    #gradients 

    grad_yhat = 2 * (y_hat - y) #dl/dy_hat
    grad_w1 = grad_yhat * x #dl/dw1
    grad_b = grad_yhat * 1 #dl/db

    #update
    w1 = w1 - lr * grad_w1
    b = b - lr * grad_b

    print(f"step {step}, weight {w1}, bias {b}, loss {loss}")
