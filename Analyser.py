#Basic setup for model
import torch
import torch.nn as nn
import torch.optim as optim

#For testing purposes, the dataset represents a y = 2x relationship, but can be manually changed by changing the values in the x and y tensors.
X = torch.tensor([
    [1.0],
    [2.0],
    [3.0],
    [4.0],
    [5.0],
    [6.0],
    [7.0],
    [8.0],
    [9.0],
    [10.0]
], dtype=torch.float32)

y = torch.tensor([
    [2.0],
    [4.0],
    [6.0],
    [8.0],
    [10.0],
    [12.0],
    [14.0],
    [16.0],
    [18.0],
    [20.0]
], dtype=torch.float32)

class Analyser(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1,8)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(8,1)
    def forward(self,x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x    


model = Analyser()
lossFunction = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

#initialize variables for tracking training progress and loss trends
epochs = 1000
previous_loss = None
improvement_count = 0
regression_count = 0
loss_history = []

for epoch in range(epochs):
    predictions = model(X)
    loss = lossFunction(predictions, y)
#improvement_count is improvemtnent in loss, regression_count is worsening of loss
    if previous_loss is not None:
        if loss.item() < previous_loss:
            improvement_count += 1
        else:
            regression_count += 1
    #appends current loss to the whole loss history for later analysis of trends and statistics
    loss_history.append(loss.item())
    #Every 100 epochs, print out the current training status, including loss, improvement/regression counts, and predictions for monitoring the training process and identifying trends in the loss values.
    if(epoch + 1) % 100 == 0:
        print(f"\nEpoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")
        print(f"improvement_count: {improvement_count}, regression_count: {regression_count}")
        print(f"Predictions: {predictions.squeeze().tolist()}")
        print(f"Current Loss: {loss.item()}")
        print(f"Current Best Loss: {min(loss_history)}")
        print(f"Current Worst Loss: {max(loss_history)}")

    previous_loss = loss.item()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

#After the training process, major key points in the loss are noted down for analysis
final_loss = loss_history[-1]
best_loss = min(loss_history)
worst_loss = max(loss_history)
initial_loss = loss_history[0]    

model.eval()
test_input = torch.tensor([[13.0]])
with torch.no_grad():
    test_output = model(test_input)

current_change = 0
previous_change = 0
worsening_trend_count = 0

#The loop analyzes the loss history to identify the trend of the loss values, which is needed to see if the trend is worsening, improving, or fluctuating.
#it compares the current chnage in loss with the previous to determine the trend and count the number of times the loss has worsened.
for check in range(2, len(loss_history)):
    current_change = loss_history[check]-loss_history[check-1]
    previous_change = loss_history[check-1]-loss_history[check-2]
    if current_change > 0 and previous_change > 0:
        if(current_change > previous_change):
            worsening_trend_count += 1

#Print out the major key points related to the loss/training process.
print("\n--- Training Statistics ---")
print("=" * 30)
print(f"Total epochs: {epochs}")
print(f"Initial Loss: {initial_loss}")
print(f"Final Loss: {final_loss}")
print(f"Final Best Loss: {best_loss}")
print(f"Final Worst Loss: {worst_loss}")

#Print out the generalization test results.
print("\n--- Generalization Test ---")
print("=" * 30)
print(f"Test input: {test_input.item()}")
print(f"Predicted output: {test_output.item()}")

#Print out the anaylysis of the learning process.
print("\n----Learning process analysis:----")
print("=" * 30)
print(f"Improvement: {improvement_count}")
print(f"Regressions: {regression_count}")
print(f"Worsening trend count: {worsening_trend_count}")

#Print out a small summary of the training process and final results based on the inputs taken from the training process.
print("\n----Summary:----")
print("=" * 30)

if final_loss < 0.01:
    print("Model has converged to a good solution."
          " Consider stopping training or reducing learning rate."
          " Final Loss: {:.4f} ".format(final_loss))
else:
    print("Model has not converged yet. Consider increasing epochs or adjusting learning rate."
          " Final Loss: {:.4f}".format(final_loss))

if(worsening_trend_count > 5):
    print("Training is showing repeated worsening trends. Consider monitoring the training process or investigating hyperparameters.")

if(improvement_count > regression_count):
    print("Model is improving overall. Consider continuing training or fine-tuning hyperparameters.")
elif(regression_count > improvement_count):
    print("Model is worsening overall. Consider adjusting learning rate or model architecture.")
else:
    print("Model is showing mixed results. Consider analyzing loss trends and adjusting training strategy accordingly.")

if(regression_count<4):
    print("Model is showing consistent improvement. Consider increasing epochs or reducing learning rate for finer convergence.")
elif(regression_count>=4 and regression_count<10):
    print("Model is showing some fluctuations. Consider monitoring training closely and adjusting hyperparameters as needed.")
else:
    print("Model is showing significant fluctuations. Consider implementing early stopping or adjusting model architecture to stabilize training.")

print("\nOverall Assessment:")
if (
    final_loss < 0.01
    and improvement_count > regression_count
    and worsening_trend_count <= 5
):
    print("Training completed successfully with stable convergence.")
else:
    print("Training completed, but additional tuning may improve performance.")