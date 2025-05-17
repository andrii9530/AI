import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

X = np.linspace(-20, 20, 400).reshape(-1, 1)
y = np.sin(X).flatten() + 0.1 * X.flatten()**2

model = LinearRegression()
model.fit(X, y)

y_pred = model.predict(X)

plt.figure(figsize=(10, 6))
plt.plot(X, y, label='Реальна функція: sin(x) + 0.1 * x²', color='red')
plt.plot(X, y_pred, label='Модель лінійної регресії', color='blue', linestyle='--')
plt.xlabel("x")
plt.ylabel("f(x)")
plt.title("Передбачення функції f(x) = sin(x) + 0.1 * x²")
plt.legend()
plt.grid(True)
plt.show()

mae = mean_absolute_error(y, y_pred)
mse = mean_squared_error(y, y_pred)

print(f"MAE (середня абсолютна помилка): {mae:.4f}")
print(f"MSE (середня квадратична помилка): {mse:.4f}")
