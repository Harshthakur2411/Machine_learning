from sklearn.linear_model import LinearRegression
import numpy as np

# Input: Area (in 100 sq.ft)
X = np.array([[1], [2], [3], [4], [5], [6]])
# Output: Price (in lakhs)
y = np.array([10, 20, 30, 40, 50, 60])

model = LinearRegression()
model.fit(X, y)

print(model.predict([[7]]))  # 50
