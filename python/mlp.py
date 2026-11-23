from sklearn.datasets import load_digits
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