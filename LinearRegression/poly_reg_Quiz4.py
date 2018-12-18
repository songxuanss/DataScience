# TODO: Add import statements
import pandas as pd
import numpy as np
# Assign the data to predictor and outcome variables
# TODO: Load the data
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

train_data = pd.read_csv('data_poly.csv')

X = np.reshape(train_data[['Var_X']], (20,1))
y = train_data[['Var_Y']]
print X, y
# Create polynomial features
# TODO: Create a PolynomialFeatures object, then fit and transform the
# predictor feature
poly_feat = PolynomialFeatures(degree=4, interaction_only=True)
X_poly = poly_feat.fit_transform(X)

print X_poly

# Make and fit the polynomial regression model
# TODO: Create a LinearRegression object and fit it to the polynomial predictor
# features
poly_model = LinearRegression()

poly_model.fit(X_poly, y)

print poly_model.predict([[1, 0.02160]])

# Once you've completed all of the steps, select Test Run to see your model
# predictions against the data, or select Submit Answer to check if the degree
# of the polynomial features is the same as ours!