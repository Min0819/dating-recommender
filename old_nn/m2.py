'''
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
===== NN classifier =====

Predict the probability of a good match using(gender, OCEAN) pairs

Usage: python3.6 m2.py
Python 3.7 is not supported as Tensorflow supports up to Python 3.6
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
'''

# ------ Import Packages ------
from keras import metrics
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE

'''
Count occurrences of binary class
@return counter = {class: count}
'''
def binary_counter(arr):
	bc = [0,0]
	for a in arr:
		bc[int(a)] += 1
	counter = {0 : bc[0], 1: bc[1]}
	return counter

# ------ Load data from file -----
data = pd.read_csv('../data/OCEAN.csv')
print (data.keys())

# input: (gender, raw OCEAN) in pairs (6 * 2 = 12 columns in total)
x = data[['gender', 'gender2', 'O1', 'C1', 'E1', 'A1', 'N1', 'O2', 'C2', 'E2', 'A2', 'N2']]

x = data[['gender', 'gender2', 'date', 'go_out', 'sports',
       'tvsports', 'exercise', 'dining', 'museums', 'art', 'hiking', 'gaming',
       'clubbing', 'reading', 'tv', 'theater', 'movies', 'concerts', 'music',
       'shopping', 'yoga', 'O1', 'C1', 'E1', 'A1', 'N1', 'date2', 'go_out2', 'sports2', 'tvsports2', 'exercise2',
       'dining2', 'museums2', 'art2', 'hiking2', 'gaming2', 'clubbing2',
       'reading2', 'tv2', 'theater2', 'movies2', 'concerts2', 'music2',
       'shopping2', 'yoga2', 'O2', 'C2', 'E2', 'A2', 'N2']]

# label: match value in binary
y = data['match']

# convert pandas dataframe to numpy matrix
x = x.values.astype('float32')
y = y.values.astype('float32')

# ======================== SMOTE Oversampling ========================
print("[INFO] SMOTE Oversampling")
print("Original Dataset: ", binary_counter(y))	# count of +ve and -ve labels
sm = SMOTE(random_state = 209)
x, y = sm.fit_sample(x, y)
print("SMOTE Resampled Dataset: ", binary_counter(y))

# ======================== SMOTE Oversampling ========================
print("[INFO] SMOTE Oversampling")
print("Original Dataset: ", binary_counter(y))	# count of +ve and -ve labels
sm = SMOTE(random_state = 209)
x, y = sm.fit_sample(x, y)
print("SMOTE Resampled Dataset: ", binary_counter(y)) 

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)

norm = True

# ------ Normalize data ------
if norm:
	x_train = normalize(x_train)
	x_test = normalize(x_test)

train = x_train
target = y_train

print(train)
print(target)
print(type(train[0]))

# check data
hasnan = np.all(np.isnan(train))
print("Have nan in x? %r" %hasnan)
hasnan = np.all(np.isnan(target))
print("Have nan in y? %r" %hasnan)


# ------ Prepare Train/Test/Validation set ------
print('[INFO] Training size: %d' %train.shape[0]) 			# 8305 rows
print('[INFO] Input vector dimension: %d' %train.shape[1])	# 12 columns

# ------ NN parameter ------
epochs = 300
batch_size = 8
validation_split = 0.2

# ------ Build NN Model ------
model = Sequential()

# Input Layer
model.add(Dense(128, kernel_initializer='normal',input_dim = train.shape[1], activation='relu'))
model.add(Dropout(0.5, noise_shape=None, seed=None))
# Hidden Layers
model.add(Dense(128, kernel_initializer='normal',activation='relu'))
model.add(Dropout(0.3, noise_shape=None, seed=None))

# Output Layer
model.add(Dense(1, kernel_initializer='normal',activation='sigmoid')) # 1-dimension output: match (probability)
#model.add(Dense(1, kernel_initializer='normal',activation='softmax'))

# Compile the network
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_absolute_error','binary_accuracy'])
#model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['mean_absolute_error', 'binary_accuracy'])
model.summary()

# ------ Checkpoint Call Back ------
checkpoint_name = 'M2-OCEAN-Weights-{epoch:03d}--{val_loss:.5f}.hdf5' 
checkpoint = ModelCheckpoint(checkpoint_name, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
callbacks_list = [checkpoint]

# ------ Train NN Model ------
print("[INFO] Training Model")
model.fit(train, target, epochs=epochs, batch_size=batch_size, validation_split=validation_split, callbacks=callbacks_list)
print("[INFO] Training Finishes")

'''
# ----- Loss Weight File of Best NN Model -----
wights_file = 'Weights-163--6.45449.hdf5' # choose the best checkpoint
model.load_weights(wights_file) # load it
model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])
'''

# ----- Save output -----
def save_prediction(prediction, iid, pid, name):
	pred = pd.DataFrame({'iid':iid, 'pid':pid, 'prediction':prediction})
	pred.to_csv('{}.csv'.format(name), index=False)
	print('Prediction file generated')


# ----- Evaluate NN Model -----
def evaluate_model(model, x_test , y_test, batch_size):

	print("Evaluation Result:")

	#print("[Metrics] MSE, MSE, Accuracy")
	scores = model.evaluate(x_test, y_test, batch_size = batch_size)
	print("\n%s: %.5f%%" % (model.metrics_names[1], scores[1]*100))
	print("\n%s: %.5f%%" % (model.metrics_names[2], scores[2]))

	'''
	# F1 score (Harmonic mean of precision and recall)
	# ROC
	y_pred = model.predict(x_test, batch_size = batch_size)

	f1 = f1_score(y_test, y_pred)
	roc = roc_auc_score(y_test, y_pred)
	kappa = cohen_kappa_score(y_test, y_pred)
	print ("F1_Score = %.5f%%  ROC_AUC = %.5f%%  Cohen_Kappa = %.5f%%" %(f1, roc, kappa))
	'''

print("[INFO] Evaluating Model")
evaluate_model(model, x_test, y_test, batch_size)

# ------ Count class distribution -------
count = [0,0]
for i in y_test:
	count[int(i)] += 1
print("0 / total = %f" %(count[0]/(count[0]+count[1])))


'''
print(x_test.shape)
y_pred = model.predict(x_test)
prob = model.predict_proba(x_test)

acc = metrics.binary_accuracy(y_test, y_pred)
print("Accuracy")
print(acc)

#pred = [int(round(x)) for x in predictions[:,0]]

#save_prediciton(predictions[:,0], iid, pid, 'OCEAN_NN') # unrounded value (year)
'''


