import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os

# Define the device based on CUDA availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define an enhanced GRU model with more layers and dropout
class EnhancedGRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=3, dropout=0.3):
        super(EnhancedGRUModel, self).__init__()
        
        # Add multiple GRU layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout)
        
        # Fully connected layers with increased hidden size
        self.fc1 = nn.Linear(hidden_size, hidden_size * 2)  # Increase the size of the first FC layer
        self.fc2 = nn.Linear(hidden_size * 2, hidden_size)  # Keep the second layer the same size
        self.fc3 = nn.Linear(hidden_size, output_size)  # Final output layer
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)  # Dropout layer
    
    def forward(self, x):
        out, _ = self.gru(x)
        out = out[:, -1, :]  # Take the output from the last time step
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)  # Apply dropout
        out = self.fc2(out)
        out = self.relu(out)
        out = self.dropout(out)  # Apply dropout again
        return self.fc3(out)

# Load the previously extracted features
features_path = '/home/kothari.je/videos/features/labels_70_Each.pt'
loaded_data = torch.load(features_path)

# Extract features and labels
all_features = loaded_data['features']
all_labels = loaded_data['labels']
label_to_index = loaded_data['label_to_index']
unique_labels = loaded_data['unique_labels']

# Print some information about the dataset
print("Loaded Features Shape:", all_features.shape)
print("Loaded Labels Shape:", all_labels.shape)
print("Unique Labels:", unique_labels)
print("Label to Index Mapping:", label_to_index)

# Hyperparameters
input_size = all_features.shape[2]  # Feature vector size
hidden_size = 2048  # Increased hidden size for better representation
output_size = len(unique_labels)  # Set output size based on the number of unique labels
num_epochs = 30  # Increased epochs for potentially better training
batch_size = 16  # Adjusted batch size
learning_rate = 0.0001

# Prepare the dataset and dataloader
dataset = TensorDataset(all_features, all_labels)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Instantiate the model, loss function, and optimizer
model = EnhancedGRUModel(input_size, hidden_size, output_size).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop with validation trackingc
best_loss = float('inf')
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    
    for features_batch, labels_batch in dataloader:
        # Move data to GPU
        features_batch, labels_batch = features_batch.to(device), labels_batch.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(features_batch)
        
        # Calculate loss
        loss = criterion(outputs, labels_batch)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    # Calculate average loss for the epoch
    avg_loss = total_loss / len(dataloader)
    print(f'Epoch [{epoch + 1}/{num_epochs}], Average Loss: {avg_loss:.4f}')
    
    # Save the best model
    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': best_loss,
            'label_to_index': label_to_index,
            'unique_labels': unique_labels,
            'input_size': input_size,  # Add this
            'output_size': output_size  # Add this
        }, 'videos/model/best_enhanced_gru_model_70_each.pth')
        print(f"Best model saved with loss: {best_loss:.4f}")

# Final model save
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': best_loss,
    'label_to_index': label_to_index,
    'unique_labels': unique_labels,
    'input_size': input_size,  # Add this
    'output_size': output_size  # Add this
}, 'videos/model/best_enhanced_gru_model_70_each.pth')

print("Training completed. Models saved.")