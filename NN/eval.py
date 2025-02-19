import numpy as np
from matplotlib import pyplot as plt
import matplotlib

# Usuwa ostrzeżenie 
matplotlib.use('TkAgg')  

# Wczytaj zapisane dane z treningu
data = np.load("NN/results.npz", allow_pickle=True)
trainingEpoch_loss = data["train_loss"]
validationEpoch_loss = data["val_loss"]
y_test = data["y_test"]
y_test_pred = data["y_test_pred"]

# Get data sizes
epochs = np.size(trainingEpoch_loss, 1)
folds = np.size(trainingEpoch_loss, 0)
x_axis = range(epochs)

# Show values for loss
min_sum = 0
for fold in range(folds):
    min_loss = min(validationEpoch_loss[fold])
    min_loss_index = np.argmin(validationEpoch_loss[fold])
    print(f"Minimal loss for fold {fold}: {round(min_loss, 4)} (Epoch: {min_loss_index})")
    min_sum += min_loss

mean_min_loss = min_sum/folds
print(f"Average minimal loss value: {round(mean_min_loss, 4)}")

# Create a 2-row, 3-column subplot grid
figure, axis = plt.subplots(2, 3, figsize=(12, 8))


# Plot each fold
axis[0,0].plot(x_axis, trainingEpoch_loss[0], label="Train Loss", color="blue")
axis[0,0].plot(x_axis, validationEpoch_loss[0], label="Validation Loss", color="orange")
axis[0,0].set_title("Fold 1")

axis[0,1].plot(x_axis, trainingEpoch_loss[1], label="Train Loss", color="blue")
axis[0,1].plot(x_axis, validationEpoch_loss[1], label="Validation Loss", color="orange")
axis[0,1].set_title("Fold 2")

axis[0,2].plot(x_axis, trainingEpoch_loss[2], label="Train Loss", color="blue")
axis[0,2].plot(x_axis, validationEpoch_loss[2], label="Validation Loss", color="orange")
axis[0,2].set_title("Fold 3")

axis[1,0].plot(x_axis, trainingEpoch_loss[3], label="Train Loss", color="blue")
axis[1,0].plot(x_axis, validationEpoch_loss[3], label="Validation Loss", color="orange")
axis[1,0].set_title("Fold 4")

axis[1,1].plot(x_axis, trainingEpoch_loss[4], label="Train Loss", color="blue")
axis[1,1].plot(x_axis, validationEpoch_loss[4], label="Validation Loss", color="orange")
axis[1,1].set_title("Fold 5")

# Hide the last empty subplot (axis[1,2])
axis[1,2].axis("off")

# Create a shared legend for all subplots
figure.legend(["Train Loss", "Validation Loss"], loc="upper center", ncol=2)

# Adjust layout to make space for the legend
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

