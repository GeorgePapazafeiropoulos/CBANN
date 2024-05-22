# System configuration:
# Install Python & pip
# cd <path>
# python -m pip install numpy
import numpy as np
# python -m pip install scikit-learn
from sklearn_extra.cluster import KMedoids
# python -m pip install tensorflow
import tensorflow as tf
from tensorflow import keras
# python -m pip install keras-models 
from keras.models import Sequential
from keras.layers import InputLayer, Dense
from keras.optimizers import Adam


# Training data
k = 5
x = np.linspace(0, k * 2 * np.pi, 500).reshape(-1, 1)
y = (np.sin(x) + 1)/2
# Query point
xq = 2
# Algorithmic settings
# (1) Related to training data:
# Random perturbation tolerance
pTol = 0
# Scaling factor
scF = 1000
# Validation ratio
tr=0.1
# (2) Related to CBANN:
# Number of additional clustering features
nCl = [2,4,6,8]
# Hidden layer size
H = 2 ** 8
# Maximum number of epochs
max_epochs = 2000

# Step 1.1: Random perturbation of features
nVar = x.shape[1]
C = x * (1 + pTol * np.random.rand(*x.shape))
C0 = C
C1 = np.hstack((C, scF * y))

# Step 1.2: Cluster Boosting (CB)
cInd = []
for i in range(len(nCl)):
    num_clusters = nCl[i]
    kmedoids = KMedoids(n_clusters=num_clusters, random_state=0)
    cInd_i = kmedoids.fit_predict(C1)
    cInd.append(cInd_i)
    C = np.hstack((C, cInd_i.reshape(-1, 1)))
C = np.hstack((C, scF * y))

# Step 2.1: Divide dataset into training and validation
n = C.shape[0]
nV = round(tr * n)
indV = np.random.choice(n, nV, replace=False)
indT = np.setdiff1d(np.arange(n), indV)
dataT = C[indT, :]
dataV = C[indV, :]
# Extract features and target variable
X_T = dataT[:, :-1] # training features
y_T = dataT[:, -1] # training labels
X_V = dataV[:, :-1] # validation features
y_V = dataV[:, -1] # validation labels

# Step 2.2: Train the neural network
model = Sequential([
    # Input layer
    InputLayer(input_shape=(X_T.shape[1],)),
    # Hidden layer with tanh activation
    Dense(H, activation='sigmoid'), 
    # Output layer with linear activation
    Dense(1, activation='linear') 
])
adam = Adam(learning_rate=0.01)
model.compile(optimizer=adam, loss='mean_squared_error', metrics=['mse'])
# Print model summary
model.summary()
# Train the ANN
history = model.fit(X_T, y_T, epochs=max_epochs, batch_size=128,
    verbose=1, shuffle=True, validation_data=(X_V, y_V))
print("Training complete")

# Step 3: Input vector needed for the ANN for prediction at xq
ind_d = np.zeros(len(nCl), dtype=int)
for i in range(len(nCl)):
    num_clusters = nCl[i]
    min_d = np.zeros(num_clusters)
    for j in range(num_clusters):
        cp = C0[cInd[i] == j, :nVar]
        d = np.sum((cp - np.tile(xq, (len(cp), 1))) ** 2, axis=1)
        min_d[j] = np.min(d)
    ind_d[i] = np.argmin(min_d)
# Formatting of input feature for prediction
pFeat = np.hstack((xq, ind_d))
pFeat = tf.reshape(pFeat,shape=(1,len(nCl)+1))

# Step 4: Make prediction on xq
pVal = 1 / scF * model.predict(pFeat)
# Display output
print("Input value: ", pFeat[0].numpy())
print("Prediction: ", pVal[0])
print("True value: ", (np.sin(xq)+1)/2)
