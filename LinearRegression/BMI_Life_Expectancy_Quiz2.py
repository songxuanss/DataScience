from sklearn.linear_model import LinearRegression
import pandas as pd

bmi_life_data = pd.read_csv("bmi_and_life_expectancy.csv")
bmi_life_model = LinearRegression()
# print bmi_life_data['BMI'].apply(lambda x: [x])
bmi_life_model.fit(bmi_life_data[['BMI']], bmi_life_data[['Life expectancy']])

laos_life_exp = bmi_life_model.predict([[21.07931]])
