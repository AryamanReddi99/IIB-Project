from random import seed
from random import randint
from numpy import array
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from math import sqrt
from sklearn.metrics import mean_squared_error

# generate examples of random integers and their sum
def random_sum_pairs(n_examples, n_numbers, largest):
	X, y = list(), list()
	for i in range(n_examples):
		in_pattern = [randint(1,largest) for _ in range(n_numbers)]
		out_pattern = sum(in_pattern)
		X.append(in_pattern)
		y.append(out_pattern)
	# format as NumPy arrays
	X,y = array(X), array(y)
	# normalize
	X = X.astype('float') / float(largest * n_numbers)
	y = y.astype('float') / float(largest * n_numbers)
	return X, y

# invert normalization
def invert(value, n_numbers, largest):
	return round(value * float(largest * n_numbers))

# generate training data
seed(1)
n_examples = 100
n_numbers = 3
largest = 100
# define LSTM configuration
n_batch = 2
n_epoch = 100
# create LSTM
model = Sequential()
model.add(Dense(4, input_dim=n_numbers))
model.add(Dense(2))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
# train LSTM
for _ in range(n_epoch):
	X, y = random_sum_pairs(n_examples, n_numbers, largest)
	model.fit(X, y, epochs=1, batch_size=n_batch, verbose=2)
# evaluate on some new patterns
X, y = random_sum_pairs(n_examples, n_numbers, largest)
result = model.predict(X, batch_size=n_batch, verbose=0)
# calculate error
expected = [invert(x, n_numbers, largest) for x in y]
predicted = [invert(x, n_numbers, largest) for x in result[:,0]]
rmse = sqrt(mean_squared_error(expected, predicted))
print('RMSE: %f' % rmse)
# show some examples
for i in range(20):
	error = expected[i] - predicted[i]
	print('Expected=%d, Predicted=%d (err=%d)' % (expected[i], predicted[i], error))
print("Finished")