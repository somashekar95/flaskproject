from transformers import GPT2Tokenizer, GPT2LMHeadModel
from torch.utils.data import Dataset, DataLoader
import torch
from flask import Flask

app = Flask(__name__)

# Define the dataset class
class ConversationDataset(Dataset):
    def __init__(self, file_path, tokenizer):
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        # Tokenize the text
        token_ids = tokenizer.encode(text)
        self.tokens = torch.tensor(token_ids)
        
    def __len__(self):
        return len(self.tokens) - 1
        
    def __getitem__(self, index):
        x = self.tokens[index]
        y = self.tokens[index+1]
        
        # Concatenate the input and target sequences along the sequence dimension
        sequence = torch.cat((x.unsqueeze(0), y.unsqueeze(0)), dim=0)
        
        return sequence

# Define the training function
def train(model, tokenizer, dataset, device, batch_size, num_epochs, learning_rate):
    # Set the model to training mode
    model.train()

    # Initialize the optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.CrossEntropyLoss()

    # Initialize the data loader
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0, pin_memory=True, collate_fn=lambda data: torch.nn.utils.rnn.pad_sequence(data, batch_first=True))

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch in data_loader:
            x_batch = batch

            # Move the data to the specified device
            x_batch = x_batch.to(device)

            # Split the input and target sequences
            y_batch = x_batch[:, 1:]
            x_batch = x_batch[:, :-1]

            # Zero the gradients
            optimizer.zero_grad()

            # Compute the logits and hidden state
            logits = model(x_batch)[0]

            # Compute the loss
            loss = loss_fn(logits.view(-1, logits.shape[-1]), y_batch.reshape(-1))

            # Backpropagate the gradients
            loss.backward()

            # Update the parameters
            optimizer.step()

            # Accumulate the loss
            epoch_loss += loss.item()

        # Print the loss for the epoch
        print(f"Epoch {epoch+1}: Loss = {epoch_loss:.4f}")


# Set the parameters
file_path = "conversation_dataset.txt"
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 64
num_epochs = 10
learning_rate = 0.001

# Create the dataset
dataset = ConversationDataset(file_path, tokenizer)

# Move the model to the specified device
model.to(device)

# Train the model
train(model, tokenizer, dataset, device, batch_size, num_epochs, learning_rate)

# Save the trained model
torch.save(model.state_dict(),"trained_model.pth") 