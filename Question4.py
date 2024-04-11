from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

# Make predictions on the test set
test_predictions = model.predict(X_test)

# Invert predictions and actual values to original scale
test_predictions = scaler.inverse_transform(test_predictions)
y_test = scaler.inverse_transform([y_test])

# Calculate evaluation metrics
mae = mean_absolute_error(y_test[0], test_predictions[:, 0])
rmse = np.sqrt(mean_squared_error(y_test[0], test_predictions[:, 0]))
print('Mean Absolute Error (MAE):', mae)
print('Root Mean Squared Error (RMSE):', rmse)

# Visualize predictions vs actual values
plt.figure(figsize=(10, 6))
plt.plot(y_test[0], label='Actual')
plt.plot(test_predictions[:, 0], label='Predicted')
plt.title('Test Set Predictions vs Actual Values')
plt.xlabel('Time Steps')
plt.ylabel('Temperature')
plt.legend()
plt.show()