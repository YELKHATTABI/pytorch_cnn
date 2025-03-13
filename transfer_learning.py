# load pretrained model

# load the model

# load the data

# train the model
import torch
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from animals_data_utils.dataset import AnimalsDataset
from torch.utils.data import DataLoader
from torchvision import transforms

from torch.utils.tensorboard import SummaryWriter

# Initialize tensorboard writer
writer = SummaryWriter("runs/mobilenet_v2_animals")

# Load pretrained MobileNetV2
model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)

# Freeze all layers
for param in model.parameters():
    param.requires_grad = False

# Replace the classifier layer
num_classes = 6
model.classifier = torch.nn.Sequential(torch.nn.Dropout(p=0.2), torch.nn.Linear(model.last_channel, num_classes))

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# get images preprocessing function of mobilenetv2
preprocessing_mobilenet_V2_eval = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

preprocessing_mobilenet_V2_train = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# get the dataset
train_dataset = AnimalsDataset(
    annotations_file="./data/animals/dataset.csv",
    images_folder=".",
    train=True,
    transform=preprocessing_mobilenet_V2_train,
)

test_dataset = AnimalsDataset(
    annotations_file="./data/animals/dataset.csv",
    images_folder=".",
    train=False,
    transform=preprocessing_mobilenet_V2_eval,
)

# get the dataloader
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# get the loss function
criterion = torch.nn.CrossEntropyLoss()

# get the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

# get the number of epochs
num_epochs = 15

# train the model
for epoch in range(num_epochs):
    running_loss = 0.0
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    # Log training loss at end of epoch
    epoch_train_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch+1} loss: {epoch_train_loss}")

    # compute validation loss
    # set model to evaluation mode
    model.eval()
    with torch.no_grad():
        total = 0
        correct = 0
        val_loss = 0.0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            # compute accuracy
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        # Calculate metrics
        epoch_val_loss = val_loss / len(test_loader)
        epoch_accuracy = 100.0 * correct / total

        # Log validation metrics
    writer.add_scalars("Loss", {"train": epoch_train_loss, "eval": epoch_val_loss}, epoch + 1)
    writer.add_scalar("Validation Accuracy", epoch_accuracy, epoch + 1)

    print(f"Epoch {epoch+1} validation loss: {epoch_val_loss}")
    print(f"Epoch {epoch+1} accuracy: {epoch_accuracy}")

    # set model to training mode
    model.train()

# Close tensorboard writer
writer.close()

# save the model
torch.save(model.state_dict(), "mobilenet_v2_animals.pth")
