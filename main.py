from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import random

style.use('fivethirtyeight')

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

data_amount = int(input('How many data points do you want to create? '))
noise = int(input('How much randomness do you want in the data? (insert a positive number): '))
print('If you want the data to be positively correlated, type "pos".')
print('If you want the data to be negatively correlated, type "neg".')
print('If you do not want the data correlated, type anything else. ')
cor = input()

if cor == 'pos' or cor == 'neg':
	steepness = int(input('How steep do you want the slope to be?: '))
else: steepness = 0

xs, ys = create_dataset(data_amount, noise, steepness, correlation=cor) #Creates data with pos correlation

#Perform calculations
m,b = best_fit_slope_and_intercept(xs,ys)
regression_line = [(m*x) + b for x in xs]
r_squared = coefficient_of_determination(ys, regression_line)

#Makes a prediction
predict_x = int(input('What value would you like to make a prediction at?: '))
predict_y = (m * predict_x) + b

#Output
print('The r squared value is: {}'.format(r_squared))
print('The slope is: {}'.format(m))
print('The y intercept is: {}'.format(b))
print('The y value of the prediction value is {}'.format(predict_y))

plt.scatter(xs,ys)
plt.scatter(predict_x,predict_y, s=100, color='g')
plt.plot(xs,regression_line)
plt.show()
