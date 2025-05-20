import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error


days = np.arange(1, 366).reshape(-1, 1)


consumption = 300 + 100 * np.cos((days.flatten() - 20) * 2 * np.pi / 365) + np.random.normal(0, 10, size=365)


X = np.hstack([
    np.sin(2 * np.pi * days / 365),
    np.cos(2 * np.pi * days / 365)
])




model = MLPRegressor(
    hidden_layer_sizes=(32, 16),
    activation='relu',
    solver='adam',
    max_iter=3000,
    early_stopping=True,
    random_state=42
)


model.fit(X, consumption)



plt.figure(figsize=(12, 6))
plt.plot(days, consumption, label='Реальне споживання', color='gray')
plt.plot(days, model.predict(X), label='Передбаченна сподиванність', color='blue')
plt.xlabel("День року")
plt.ylabel("Споживання електроенергії (кВт·год)")
plt.title("Передбачення споживання електро-енергії в різні дні року")
plt.grid(True)
plt.legend()
plt.show()
