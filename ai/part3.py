import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


X = np.linspace(-10, 10, 1000) 
y = X** 2 * np.sin(X) 

plt.plot(X, y)
plt.title('lox')
plt.xlabel('X')
plt.ylabel('san(X)')
plt.grid(True)
plt.show()




data = np.random.normal(loc=5, scale=2, size=1000)

plt.hist(data, bins=30, color='skyblue', edgecolor='black')
plt.title('Гістограма нормального розподілу (μ=5, σ=2)')
plt.xlabel('Значення')
plt.ylabel('Частота')
plt.grid(True)
plt.show()




labels = ['Їсти', 'Грати', 'гуляти', 'займатись дурнею']
sizes = [20, 25, 25, 30]  
colors = ['orange', 'red','green','yellow' ]
explode = (0, 0, 0, 0.1)  


plt.pie(sizes, labels=labels, colors=colors, explode=explode,
        autopct='%1.1f%%', shadow=True, startangle=140)

plt.title('Улюблені хобі')
plt.axis('equal') 
plt.show()




np.random.seed(42)

apples = np.random.normal(loc=150, scale=10, size=100)  
bananas = np.random.normal(loc=120, scale=15, size=100)
oranges = np.random.normal(loc=130, scale=12, size=100)
kiwis = np.random.normal(loc=80, scale=8, size=100)      

data = [apples, bananas, oranges, kiwis]
labels = ['Яблука', 'Банани', 'Апельсини', 'Ківі']

plt.boxplot(data, labels=labels, patch_artist=True)


plt.title("Box-Plot маси фруктів (100 випадкових значень)")
plt.ylabel("Маса (г)")
plt.grid(True)
plt.show()