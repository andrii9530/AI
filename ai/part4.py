import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# 1. Генерація даних
X = np.linspace(-20, 20, 400).reshape(-1, 1)
y = np.sin(X).flatten() + 0.1 * X.flatten()**2

# 2. Навчання моделі
model = LinearRegression()
model.fit(X, y)

# 3. Прогноз
y_pred = model.predict(X)

# 4. Візуалізація
plt.figure(figsize=(10, 6))
plt.plot(X, y, label='Реальна функція: sin(x) + 0.1 * x²', color='blue')
plt.plot(X, y_pred, label='Модель лінійної регресії', color='red', linestyle='--')
plt.xlabel("x")
plt.ylabel("f(x)")
plt.title("Передбачення функції f(x) = sin(x) + 0.1 * x²")
plt.legend()
plt.grid(True)
plt.show()

# 5. Метрики якості
mae = mean_absolute_error(y, y_pred)
mse = mean_squared_error(y, y_pred)

print(f"📏 MAE (середня абсолютна помилка): {mae:.4f}")
print(f"📐 MSE (середня квадратична помилка): {mse:.4f}")
