import numpy as np 
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error


X = np.linspace(0, 24, 500).reshape(-1, 1)


y = 20 + 10 * np.sin((X.flatten() - 8) * np.pi / 12)**2 + 5 * np.sin((X.flatten() - 17) * np.pi / 8)**2


scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

nn_model = MLPRegressor(hidden_layer_sizes=(32, 16), max_iter=10000, random_state=42)
nn_model.fit(X_scaled, y)


poly_model = make_pipeline(PolynomialFeatures(degree=5), LinearRegression())
poly_model.fit(X_scaled, y)


def time_to_decimal(hhmm):
    h, m = map(int, hhmm.split(':'))
    return h + m / 60


times = ['10:30', '00:00', '02:40']
print(" Передбачення для заданих моментів часу:")
for t in times:
    dec_time = np.array([[time_to_decimal(t)]])
    scaled_time = scaler.transform(dec_time)
    nn_pred = nn_model.predict(scaled_time)[0]
    poly_pred = poly_model.predict(scaled_time)[0]
    print(f" {t} → NN: {nn_pred:.2f} хв | PolyReg: {poly_pred:.2f} хв")


X_test = np.linspace(0, 24, 1000).reshape(-1, 1)
X_test_scaled = scaler.transform(X_test)

plt.figure(figsize=(10, 6))
plt.plot(X, y, label='Реальна тривалість', color='gray', linestyle='--')
plt.plot(X_test, nn_model.predict(X_test_scaled), label='Нейронна мережа', color='blue')
plt.plot(X_test, poly_model.predict(X_test_scaled), label='Поліном. регресія', color='green')
plt.xlabel('Час доби')
plt.ylabel('Тривалість поїздки (хв)')
plt.title('Передбачення тривалості поїздки (NN vs PolyReg)')
plt.grid(True)
plt.legend()
plt.show()

nn_mae = mean_absolute_error(y, nn_model.predict(X_scaled))
poly_mae = mean_absolute_error(y, poly_model.predict(X_scaled))

print(f"\n📏 MAE (Нейромережа): {nn_mae:.4f}")
print(f"📐 MAE (Поліноміальна регресія): {poly_mae:.4f}")
