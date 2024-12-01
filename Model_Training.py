import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os
 
# Define the device based on CUDA availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
 
# Attention Mechanism
class Attention(nn.Module):
    def __init__(self, input_dim):
        super(Attention, self).__init__()
        self.attn = nn.Linear(input_dim, 1, bias=False)  # Linear layer to compute attention scores
    def forward(self, x):
        # Calculate attention scores
        attn_weights = self.attn(x)
        attn_weights = torch.softmax(attn_weights, dim=1)  # Softmax to normalize
        weighted_input = x * attn_weights  # Apply attention weights to the input
        return weighted_input, attn_weights
 
# Enhanced GRU Model with Attention
class EnhancedGRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=3, dropout=0.3):
        super(EnhancedGRUModel, self).__init__()
        # GRU layer with multiple layers and dropout
        self.gru = nn.GRU(input_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout)
        # Attention mechanism
        self.attn = Attention(hidden_size)
        # Fully connected layers with increased hidden size
        self.fc1 = nn.Linear(hidden_size, hidden_size * 2)  # Increase the size of the first FC layer
        self.fc2 = nn.Linear(hidden_size * 2, hidden_size)  # Keep the second layer the same size
        self.fc3 = nn.Linear(hidden_size, output_size)  # Final output layer
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)  # Dropout layer
    def forward(self, x):
        # GRU output
        out, _ = self.gru(x)
        # Apply Attention
        weighted_out, attn_weights = self.attn(out)
        # Take the output from the last time step (sum over the weighted output)
        out = weighted_out.sum(dim=1)
        # Fully connected layers with ReLU activations and Dropout
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.dropout(out)
        # Output layer
        return self.fc3(out)
 
# Load the previously extracted features
features_path = '/home/kothari.je/videos/features/labels_10_sec_each.pt'
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
input_size = all_features.shape[2]  # Feature vector size (e.g., from extracted features)
hidden_size = 2048  # Increased hidden size for better representation
output_size = len(unique_labels)  # Set output size based on the number of unique labels
num_epochs = 50  # Increased epochs for potentially better training
batch_size = 16  # Adjusted batch size
learning_rate = 0.0001
 
# Prepare the dataset and dataloader
dataset = TensorDataset(all_features, all_labels)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
 
# Instantiate the model, loss function, and optimizer
model = EnhancedGRUModel(input_size, hidden_size, output_size).to(device)
criterion = nn.CrossEntropyLoss()  # Suitable for multi-class classification
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
 
# Training loop with validation tracking
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
    # Save the best model based on loss
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
        }, '/home/kothari.je/videos/model/final_enhanced_gru_model.pth')
        print(f"Best model saved with loss: {best_loss:.4f}")
 
# Final model save after training completes
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': best_loss,
    'label_to_index': label_to_index,
    'unique_labels': unique_labels,
    'input_size': input_size,  # Add this
    'output_size': output_size  # Add this
}, '/home/kothari.je/videos/model/final_enhanced_gru_model.pth')
 
print("Training completed. Models saved.")