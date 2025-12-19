from sklearn.linear_model import LogisticRegression
import numpy as np

x = [[0], [0], [2], [3]]
y = [0, 0, 1, 1]
model = LogisticRegression()
model.fit(x, y)
predictions = model.predict([[1.5], [0.5]]) 
print(predictions)  # Output: [0 1]
