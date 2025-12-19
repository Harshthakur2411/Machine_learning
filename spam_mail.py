# Step 1: Import required libraries
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Step 2: Create dataset (Email + Label)
data = {
    'email': [
        'Win money now',
        'Limited offer claim prize',
        'Congratulations you won lottery',
        'Meeting tomorrow at office',
        'Project deadline extended',
        'Let us discuss the report'
    ],
    'label': [1, 1, 1, 0, 0, 0]   # 1 = Spam, 0 = Not Spam
}

df = pd.DataFrame(data)

# Step 3: Separate input and output
X = df['email']   # Email text
y = df['label']   # Spam / Not Spam

# Step 4: Convert text data into numerical form
vectorizer = CountVectorizer()
X_vectorized = vectorizer.fit_transform(X)

# Step 5: Train the model
model = MultinomialNB()
model.fit(X_vectorized, y)

# Step 6: Test with new email
new_email = ['Win cash prize now']
new_email_vector = vectorizer.transform(new_email)

prediction = model.predict(new_email_vector)

# Step 7: Output result
if prediction[0] == 1:
    print("Spam Email ❌")
else:
    print("Not Spam Email ✅")
