import csv
import numpy as np

# Your trained model parameters (replace with actual values after training)

# 0.006158791777281181, 0.4678280082082864, 0.16488740947822755
a0 = 0.006158791777281181
a1 = 0.4678280082082864
a2 = 0.16488740947822755


def predict_distance(x1, x2, a0, a1, a2):
    """Predict distance using the trained model"""
    y = a0 + a1 * x1 + a2 * x2
    if y > 0:
        return 1 / y
    else:
        return float('inf')  # Invalid prediction


def load_test_data(file_path):
    """Load test dataset"""
    x1_values = []
    x2_values = []
    true_distances = []

    with open(file_path, "r") as file:
        reader = csv.DictReader(file)
        for row in reader:
            x1_values.append(float(row["x1"]))
            x2_values.append(float(row["x2"]))
            true_distances.append(float(row["distance"]))

    return x1_values, x2_values, true_distances


def calculate_mae(predicted, actual):
    """Calculate Mean Absolute Error"""
    return np.mean(np.abs(np.array(predicted) - np.array(actual)))


def calculate_metrics(predicted, actual):
    """Calculate various performance metrics"""
    predicted = np.array(predicted)
    actual = np.array(actual)

    mae = np.mean(np.abs(predicted - actual))
    rmse = np.sqrt(np.mean((predicted - actual) ** 2))
    mape = np.mean(np.abs((predicted - actual) / actual)) * 100
    max_error = np.max(np.abs(predicted - actual))

    return mae, rmse, mape, max_error


# Main benchmarking code
if __name__ == "__main__":
    # Load test data
    test_file = "datasets/test_dataset.csv"  # Update path as needed
    x1_test, x2_test, true_distances = load_test_data(test_file)

    # Make predictions
    predicted_distances = []
    for x1, x2 in zip(x1_test, x2_test):
        pred_dist = predict_distance(x1, x2, a0, a1, a2)
        predicted_distances.append(pred_dist)

    # Calculate metrics
    mae, rmse, mape, max_error = calculate_metrics(predicted_distances, true_distances)

    # Print results
    print("=== MODEL BENCHMARK RESULTS ===")
    print(f"Mean Absolute Error (MAE): {mae:.2f} meters")
    print(f"Root Mean Square Error (RMSE): {rmse:.2f} meters")
    print(f"Mean Absolute Percentage Error (MAPE): {mape:.1f}%")
    print(f"Maximum Error: {max_error:.2f} meters")
    print()

    # Performance evaluation
    if mae < 3:
        print("🎉 EXCELLENT performance (MAE < 3m)")
    elif mae < 5:
        print("✅ GOOD performance (MAE < 5m)")
    elif mae < 10:
        print("⚠️  ACCEPTABLE performance (MAE < 10m)")
    else:
        print("❌ POOR performance (MAE >= 10m)")

    print()
    print("=== DETAILED PREDICTIONS ===")
    print("True Dist | Predicted | Error")
    print("-" * 30)
    for i, (true_dist, pred_dist) in enumerate(zip(true_distances, predicted_distances)):
        error = abs(pred_dist - true_dist)
        print(f"{true_dist:8.1f} | {pred_dist:9.1f} | {error:5.1f}")

    # Identify worst predictions
    errors = np.abs(np.array(predicted_distances) - np.array(true_distances))
    worst_idx = np.argmax(errors)
    print()
    print(f"Worst prediction: Test case {worst_idx + 1}")
    print(f"True: {true_distances[worst_idx]:.1f}m, Predicted: {predicted_distances[worst_idx]:.1f}m")
    print(f"Error: {errors[worst_idx]:.1f}m")