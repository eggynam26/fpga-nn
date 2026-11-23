from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt



digits = load_digits()
x = digits.data
y = digits.target

print("X shape:", x.shape)
print("y shape:", y.shape) 
print("first label:", y[0])
print("first image values:", x[0])

plt.imshow(digits.images[0], cmap='gray')
plt.show()


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)


print("x_train:", x_train.shape)
print("x_test:", x_test.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)