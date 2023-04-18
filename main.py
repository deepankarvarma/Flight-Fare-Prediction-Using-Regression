import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

# Load the dataset into a pandas dataframe
df = pd.read_csv('Cleaned_dataset.csv')

# Check for missing values
print(df.isnull().sum())

# Replace missing values with the mean
# df.fillna(df.mean(), inplace=True)

# # Check for outliers
# # You can use box plots or scatter plots to identify outliers
# # Remove outliers that are more than 3 standard deviations away from the mean
df = df[(df['Fare'] - df['Fare'].mean()).abs() < 3 * df['Fare'].std()]

# # Check for correlations
corr_matrix = df.corr()
print(corr_matrix['Fare'].sort_values(ascending=False))

 # Normalize the data using standard scaler
# scaler = StandardScaler()
# X = scaler.fit_transform(df.drop('Fare', axis=1))
# y = df['Fare']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df.drop({'Fare','Date_of_journey'}, axis=1), df['Fare'], test_size=0.2, random_state=42)

# Train a linear regression model on the training set
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate the model on the training set
y_pred = model.predict(X_train)
mse = mean_squared_error(y_train, y_pred)

# Print the MSE and model parameters
print("MSE on training set: ", mse)
print("Model intercept: ", model.intercept_)
print("Model coefficients: ", model.coef_)
