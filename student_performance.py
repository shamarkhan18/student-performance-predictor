from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = pd.read_csv("StudentsPerformance.csv")

# Encode text columns

le = LabelEncoder()

categorical_columns = [
    'gender',
    'race/ethnicity',
    'parental level of education',
    'lunch',
    'test preparation course'
]

for col in categorical_columns:
    df[col] = le.fit_transform(df[col])

# Features (Input)

X = df.drop('math score', axis=1)

# Target (Output)

y = df['math score']

# Split dataset

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train model

model = LinearRegression()

model.fit(X_train, y_train)

# Train model

model = LinearRegression()

model.fit(X_train, y_train)

print("Model trained successfully!")

# Make predictions

predictions = model.predict(X_test)

print("\nFirst 10 Predictions:")

for actual, predicted in zip(y_test.head(10), predictions[:10]):
    print(
        f"Actual: {actual} | Predicted: {predicted:.2f}"
    )

mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("\nModel Performance")
print("Mean Absolute Error:", round(mae, 2))
print("R2 Score:", round(r2, 2))

print("\n------ Predict New Student ------")

reading = float(input("Reading Score: "))
writing = float(input("Writing Score: "))

sample = pd.DataFrame({
    'gender':[1],
    'race/ethnicity':[2],
    'parental level of education':[3],
    'lunch':[1],
    'test preparation course':[1],
    'reading score':[reading],
    'writing score':[writing]
})

prediction = model.predict(sample)

print("\nPredicted Math Score:", round(prediction[0],2))