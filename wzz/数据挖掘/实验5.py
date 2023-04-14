import numpy as np
import matplotlib.pyplot as plt

# 1.
np.random.seed(0)
n_samples = 100
x = np.random.rand(n_samples, 1) * 10
y = 2 * x + 1 + np.random.randn(n_samples, 1)

# 2.
for i in range(5):
    print(f"({x[i]},{y[i]})")

plt.scatter(x,y,c='blue')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()

# 3.
from sklearn import linear_model

model = linear_model.LinearRegression()
model.fit(x,y)
y_ = model.predict(x)
plt.scatter(x,y)
plt.plot(x,y_,color='red',linewidth=3.0,linestyle='-')
plt.show()

# 4.
from sklearn import linear_model
model = linear_model.SGDRegressor()
model.fit(x,y)
y_ = model.predict(x)
plt.scatter(x,y)
plt.plot(x,y_,color='red',linewidth=3.0,linestyle='-')
plt.show()



