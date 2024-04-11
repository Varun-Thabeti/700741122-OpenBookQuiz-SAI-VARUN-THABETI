import numpy as np
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from tensorflow.keras.optimizers import Adam

# Define a function to create the LSTM model
def create_lstm_model(learning_rate=0.001, units=50, dropout_rate=0.2):
    model = Sequential()
    model.add(LSTM(units=units, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(Dropout(dropout_rate))
    model.add(LSTM(units=units, return_sequences=True))
    model.add(Dropout(dropout_rate))
    model.add(LSTM(units=units))
    model.add(Dropout(dropout_rate))
    model.add(Dense(units=1))
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    return model

# Wrap the Keras model in a scikit-learn estimator
estimator = KerasRegressor(build_fn=create_lstm_model)

# Define the grid of hyperparameters to search
param_grid = {
    'learning_rate': [0.001, 0.01, 0.1],
    'units': [50, 100],
    'dropout_rate': [0.2, 0.3]
}

# Perform grid search with 3-fold cross-validation
grid_search = GridSearchCV(estimator=estimator, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error', verbose=1)
grid_result = grid_search.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.1, verbose=0)

# Print the best hyperparameters and the corresponding mean squared error
print("Best Hyperparameters:", grid_result.best_params_)
print("Best Mean Squared Error:", -grid_result.best_score_)