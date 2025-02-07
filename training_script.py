from mnist_model.model import SimpleCNN, dropoutCNN
from mnist_model.data import MnistDataset
from mnist_model.data import transform_image
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm.auto import tqdm
from datetime import datetime


def evaluate(net, test_dataloader):
    ground_truths = []
    predictions = []
    net = net.eval()
    with torch.no_grad():
        total_correct_classifications = 0
        total_images = 0
        eval_loss = 0
        for x, y in tqdm(test_dataloader):
            x = x.to("cuda")
            y_pred = net(x).to("cpu")
            predicted = y_pred.argmax(dim=1)
            total_images = total_images + y.size(0)
            correct_classifications = (
                (predicted == y).sum().item()
            )  # .item() is necessary to convert the tensor to a number
            total_correct_classifications = total_correct_classifications + correct_classifications
            eval_loss += F.cross_entropy(y_pred, y, reduction="sum").item()
            ground_truths.extend(y.tolist())
            predictions.extend(predicted.tolist())
        accuracy = total_correct_classifications / total_images
        eval_loss = eval_loss / total_images
        print(f"Eval Loss: {eval_loss}")
        print(f"Accuracy: {accuracy:.2f}")

    net = net.train()
    return ground_truths, predictions, accuracy, eval_loss


if __name__ == "__main__":

    # This code section helps us select the device we want to use (CPU or GPU) automatically
    # you can force device to cpu if you don't want to use GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Hyperparameters (you can change them)
    learning_rate = 0.001
    batch_size = 64
    epochs = 20

    # this is purely to create folder names with the time (you can do whatever you want)
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = "fashion_mnist_dropout"
    run_name = f"{run_name}_{current_time}"

    # Datasets and Dataloaders
    train_dataset = MnistDataset(
        annotations_file="./data/fashion_mnist/fashion_mnist.csv",
        images_folder="./data/fashion_mnist/",
        train=True,
        transform=transform_image,
        target_transform=None,
    )
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = MnistDataset(
        annotations_file="./data/fashion_mnist/fashion_mnist.csv",
        images_folder="./data/fashion_mnist/",
        train=False,
        transform=transform_image,
        target_transform=None,
    )
    # for test loaders we don't need to shuffle the data or to have batch size > 1
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    model = dropoutCNN().to("cuda")

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    loss_fn = nn.CrossEntropyLoss()

    summary_writer = SummaryWriter(log_dir=f"./runs/{run_name}")

    for epoch in range(epochs):
        accumulated_loss = 0
        progress_bar = tqdm(train_dataloader)
        for i, (x, y) in enumerate(train_dataloader):
            x = x.to("cuda")
            y = y.to("cuda")
            # Forward pass: Compute predicted y by passing x to the model
            y_pred = model(x)

            # Compute and print loss
            loss = loss_fn(y_pred, y)
            accumulated_loss += loss.item()

            # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            progress_bar.update()
            progress_bar.set_description(f"Loss: {accumulated_loss/(i+1):.4f}")
            progress_bar.refresh()

        print(f"Epoch {epoch + 1}: Train Loss {accumulated_loss/len(train_dataloader):.4f}")
        _, _, eval_acc, eval_loss = evaluate(model, test_dataloader)
        summary_writer.add_scalars(
            "Loss", {"train": accumulated_loss / len(train_dataloader), "eval": eval_loss}, epoch + 1
        )
        summary_writer.add_scalars("Accuracy", {"eval": eval_acc}, epoch + 1)

        # save the model
        torch.save(model.state_dict(), f"./runs/{run_name}/model.pth")
