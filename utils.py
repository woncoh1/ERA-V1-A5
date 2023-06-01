import torch
from torchvision import datasets, transforms
from tqdm.auto import tqdm
import matplotlib.pyplot as plt


# Data
transform = {
    'train': transforms.Compose([
        transforms.RandomApply([transforms.CenterCrop(22)], p=0.1),
        transforms.Resize((28, 28)),
        transforms.RandomRotation((-15., 15.), fill=0),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ]),
    'test': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ]),
}

dataset = {
    'train': datasets.MNIST(
        '../data',
        train=True,
        download=True,
        transform=transform['train'],
    ),
    'test': datasets.MNIST(
        '../data',
        train=False,
        download=False,
        transform=transform['test'],
    ),
}

params_dataloader = {
    'batch_size': 128,
    'shuffle': True,
    'num_workers': 0,
    'pin_memory': True,
}

dataloader = {
    'train': torch.utils.data.DataLoader(
        dataset['train'],
        **params_dataloader
    ),
    'test': torch.utils.data.DataLoader(
        dataset['test'],
        **params_dataloader
    ),
}

def inspect_batch():
    batch_data, batch_label = next(iter(dataloader['train'])) 
    fig = plt.figure()
    for i in range(12):
        plt.subplot(3, 4, i+1)
        plt.tight_layout()
        plt.imshow(batch_data[i].squeeze(0), cmap='gray')
        plt.title(batch_label[i].item())
        plt.xticks([])
        plt.yticks([])


# Train
train_losses = []
test_losses = []
train_acc = []
test_acc = []
test_incorrect_pred = {
    'images': [],
    'ground_truths': [],
    'predicted_vals': [],
}

def GetCorrectPredCount(pPrediction, pLabels):
    return pPrediction.argmax(dim=1).eq(pLabels).sum().item()

def train(model, device, train_loader, optimizer, criterion):
    model.train()
    pbar = tqdm(train_loader)

    train_loss = 0
    correct = 0
    processed = 0

    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        # Predict
        pred = model(data)

        # Calculate loss
        loss = criterion(pred, target)

        # Backpropagation
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        correct += GetCorrectPredCount(pred, target)
        processed += len(data)

        pbar.set_description(desc=(
            "Train: "
            f"Loss={loss.item():0.4f} "
            f"Batch_id={batch_idx} "
            f"Accuracy={100*correct/processed:0.2f}"
        ))

    train_acc.append(100*correct/processed)
    train_losses.append(train_loss/len(train_loader))

def test(model, device, test_loader, criterion):
    model.eval()

    test_loss = 0
    correct = 0

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            
            output = model(data)
            
            test_loss += criterion(output, target, reduction='sum').item()
            correct += GetCorrectPredCount(output, target)

    test_acc.append(100. * correct / len(test_loader.dataset))
    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)

    print("Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n".format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def plot_results():
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    plt.setp(axs, xticks=range(0, 21, 5))
    axs[0, 0].plot(train_losses)
    axs[0, 0].set_title("Training Loss")
    axs[0, 0].axhline(y=0, color='r', linestyle='dotted')
    axs[1, 0].plot(train_acc)
    axs[1, 0].set_title("Training Accuracy")
    axs[1, 0].axhline(y=100, color='r', linestyle='dotted')
    axs[0, 1].plot(test_losses)
    axs[0, 1].set_title("Test Loss")
    axs[0, 1].axhline(y=0, color='r', linestyle='dotted')
    axs[1, 1].plot(test_acc)
    axs[1, 1].set_title("Test Accuracy")
    axs[1, 1].axhline(y=100, color='r', linestyle='dotted')
    plt.show()