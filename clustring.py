from sklearn.cluster import KMeans
import numpy as np

# Data: Age and Income
X = np.array([
    [25, 30000],
    [45, 90000],
    [23, 28000],
    [35, 60000],
    [52, 110000]
])

# Model
model = KMeans(n_clusters=2)
model.fit(X)

# Cluster prediction
data = np.array(model.labels_)
for i in data:
    if i==0:
        print("Rich")
    
    else:
        print("Poor")
    
