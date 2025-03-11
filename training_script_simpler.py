"""
This is the main script
"""

import torch
from tqdm import tqdm
from torch import nn
from fashion_mnist_model import SimpleCNN
import torch.nn.functional as F
from fashion_mnist_data import FashionMnistDataset, transform_image
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


# Hyperparameters
batch_size = 100
learning_rate = 0.001
num_epochs = 10 # as much iterations as train_set_size / batch_size * epoch


train_dataset = FashionMnistDataset(
    annotations_file="./data/fashion_mnist/fashion_mnist.csv",
    images_folder="./data/fashion_mnist/",
    transform=transform_image,
    train=True,
)


test_dataset = FashionMnistDataset(
    annotations_file="./data/fashion_mnist/fashion_mnist.csv",
    images_folder="./data/fashion_mnist/",
    transform=transform_image,
    train=False,
)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=4,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=4,
)


model = SimpleCNN()

# we define a loss that will be used to train the model
loss_fn = nn.CrossEntropyLoss()

# we define an optimizer to update the parameters
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

summary_writer = SummaryWriter(log_dir="./runs/cnn_mnist")

for epoch in tqdm(range(num_epochs)):
    model = model.train()
    proxy_train_loss = 0

    # we iterate over the train_loader
    for images, labels in tqdm(train_loader):
        # we reset the gradient
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        proxy_train_loss += loss.item()
        
        
        # Backward pass
        loss.backward()
        optimizer.step()

    proxy_train_loss = proxy_train_loss / len(train_loader)

    print(f"Train Epoch {epoch + 1} loss {proxy_train_loss:.4f}")

    # test the model at this epoch
    # we need to disable the gradient calculation for the test
    model = model.eval()
    
    with torch.no_grad():
        eval_loss = 0
        correct = 0

        # we iterate over the test_loader
        for images, labels in tqdm(test_loader):

            logit_predictions = model(images)
            predicted = F.softmax(logit_predictions).argmax(dim=1)
            correct += (predicted == labels).sum().item()
            eval_loss += F.cross_entropy(logit_predictions, labels).item()

        eval_loss = eval_loss / len(test_loader)
        print(f"Eval Loss: {eval_loss:.4f}")
        print(f"Accuracy: {correct / len(test_loader) * 100:.2f}%")
        print(f"correct: {correct}")

    summary_writer.add_scalars("Loss", {"train": proxy_train_loss, "eval": eval_loss}, epoch + 1)
