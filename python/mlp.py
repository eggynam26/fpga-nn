from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(0)

digits = load_digits()
x = digits.data
y = digits.target

print("X shape:", x.shape)
print("y shape:", y.shape) 
print("first label:", y[0])
print("first image values:", x[0])

#plt.imshow(digits.images[0], cmap='gray')
#plt.show()


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)


#print("x_train:", x_train.shape)
#print("x_test:", x_test.shape)
#print("y_train:", y_train.shape)
#print("y_test:", y_test.shape)


x_train = x_train / 16.0
x_test = x_test / 16.0

W1 = np.random.randn(64, 32) * 0.1
b1 = np.zeros(32)
W2 = np.random.randn(32, 10) * 0.1
b2 = np.zeros(10)

def forward(x):
    hidden = np.maximum(0, x @ W1 + b1)  # ReLU activation
    return hidden, hidden @ W2 + b2  # Output layer (no activation for logits)


hidden, scores = forward(x_test)
pred = np.argmax(scores, axis=1)
print(np.mean(pred == y_test))  # Print accuracy

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / np.sum(e_x, axis=1, keepdims=True)

def loss(probs, y):
    N = y.shape[0]
    correct_logprobs = -np.log(probs[range(N), y])
    return np.sum(correct_logprobs) / N

lr = 0.5
N = x_train.shape[0]

for step in range(501):
    # forward pass
    hidden, scores = forward(x_train)
    probs = softmax(scores)

    # backward pass: how wrong, and how to nudge each parameter
    dscores = probs.copy()
    dscores[range(N), y_train] -= 1      # error at the output
    dscores /= N                         # average over images

    dW2 = hidden.T @ dscores
    db2 = dscores.sum(axis=0)

    dhidden = dscores @ W2.T             # send error back through W2
    dhidden[hidden <= 0] = 0             # ReLU blocked these units

    dW1 = x_train.T @ dhidden
    db1 = dhidden.sum(axis=0)

    # update: step each parameter downhill
    W1 -= lr * dW1
    b1 -= lr * db1
    W2 -= lr * dW2
    b2 -= lr * db2

    if step % 50 == 0:
        print(step, loss(probs, y_train))

_, scores = forward(x_test)
print("test accuracy:", np.mean(np.argmax(scores, axis=1) == y_test))