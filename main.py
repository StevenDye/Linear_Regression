import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import random
from matplotlib import style
from sklearn.linear_model import LinearRegression
from statistics import mean

style.use('fivethirtyeight')

########## DEFINE FUNCTIONS ####################

#Creates random data
def create_dataset(how_many, variance, step=0, correlation=False):
	val = 1
	ys = []
	for i in range(how_many):
		y = val + random.randrange(-variance, variance)
		ys.append(y)
		if correlation == 'pos':
			val+=step
		if correlation == 'neg':
			val-=step
	xs = [i for i in range(len(ys))]
	return np.array(xs, dtype=np.float64), np.array(ys, dtype=np.float64)

#Finds slope and x intercept
def best_fit_slope_and_intercept(xs, ys):
	# This only approximates from a sample of the data using covariance
	m =  (mean(xs) * mean(ys) - mean(xs*ys)) / ((mean(xs)**2) - (mean(xs**2)))
	b = mean(ys) - m * mean(xs)
	return m, b

#Finds the square of the error
def squared_error(ys_orig, ys_line):
	return sum((ys_line - ys_orig)**2)

#Finds coefficient of squared error
def coefficient_of_determination(ys_orig, ys_line):
	y_mean_line = [mean(ys_orig) for y in ys_orig]
	squared_error_regr = squared_error(ys_orig, ys_line)
	squared_error_y_mean = squared_error(ys_orig, y_mean_line)
	return 1 - (squared_error_regr / squared_error_y_mean)


############ CREATE DATA ####################

data_amount = int(input('How many data points do you want to create? '))
noise = int(input('How much randomness do you want in the data? (insert a positive number): '))
print('If you want the data to be positively correlated, type "pos".')
print('If you want the data to be negatively correlated, type "neg".')
print('If you do not want the data correlated, type anything else. ')
cor = input()

if cor == 'pos' or cor == 'neg':
	steepness = int(input('How steep do you want the slope to be?: '))
else: steepness = 0

xs, ys = create_dataset(data_amount, noise, steepness, correlation=cor)


############### PERFORM CALCULATIONS #####################

m,b = best_fit_slope_and_intercept(xs,ys)
regression_line = [(m*x) + b for x in xs]
r_squared = coefficient_of_determination(ys, regression_line)

# Linear Regression model from sklearn
linear_regressor = LinearRegression()
X = xs.reshape(-1, 1) # Reshapes data to be two dimensional
Y = ys.reshape(-1, 1)
linear_regressor.fit(X, Y)
Y_pred = linear_regressor.predict(X)

#Makes a prediction
predict_x = int(input('What value would you like to make a prediction at?: '))
predict_y = (m * predict_x) + b


############ OUTPUT ###################

# Slope
print('Slope from regression line is: {}'.format(m))
print('Slope from sklearn model is: {}'.format(linear_regressor.coef_))
print()

# y intercept
print('The y intercept from regression line is: {}'.format(b))
print('The y intercept from skelarn model is:  {}'.format(linear_regressor.intercept_))
print()

# Prediction point
print('The y value of the prediction point from regression line is: {}'.format(predict_y))
print('The y value of the prediction point from sklearn model is: {}'.format(linear_regressor.predict(predict_x)))
print()

# R squared value
print('The r squared value from regression line is:   {}'.format(r_squared))
print('The r squared value from the sklearn model is: {}'.format(linear_regressor.score(X, Y, sample_weight=None)))
print()


############# PLOT ######################

# Plot data points
plt.scatter(xs,ys)

#plot calculated linear regrsession line
plt.plot(xs,regression_line, color='black', linewidth=5.0)

# Plot Linear Regression Model from sklearn
plt.plot(xs, Y_pred, color='orange', linewidth=2.0)

# Plot prediction points
plt.scatter(predict_x,predict_y, s=200, color='g')
plt.scatter(predict_x,linear_regressor.predict(predict_x), s=100, color='r')

model_patch = mpatches.Patch(color='black',label='Model')
sklearn_patch = mpatches.Patch(color='orange',label='sklearn')
plt.legend(handles=[model_patch, sklearn_patch])

plt.show()
