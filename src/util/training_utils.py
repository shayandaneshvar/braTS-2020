import torch
from tqdm import tqdm


# must load the model to the device and also prep the loss function with model parameters before using this function
# also do not forget to initialize the optimizer with the desired learning rate
def train(model, epochs=1, training_loader=None, loss_fn=None, device=None,
          optimizer: torch.optim.Optimizer = None):
    for epoch in range(epochs):
        tq_dl = tqdm(training_loader)
        for idx, (image, mask) in enumerate(tq_dl):
            image, mask = image.to(device), mask.to(device)
            # forward pass
            out = model(image)
            loss = loss_fn(out, mask)
            # backward pass
            optimizer.zero_grad()
            loss.backward()

            # optimize
            optimizer.step()

            tq_dl.set_description(f"At epoch [{epoch + 1}/{epochs}]")
            tq_dl.set_postfix(loss=loss.item())  # acc, ...


# do not give in the format - the format will be .pt
def save(model, path):
    torch.save(model.state_dict(), f"{path}.pt")


# do not give in the format - the format will be .pt
def load(model, path, eval=True):
    model.load_state_dict(torch.load(f"{path}.pt"))
    if eval:
        model.eval()


def check_accuracy(data_loader, model, device="cuda"):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()

    with torch.no_grad():
        for x, y in data_loader:
            x = x.to(device)
            y = y.to(device) #.unsqueeze(1)
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2 * (preds * y).sum()) / (
                    (preds + y).sum() + 1e-8
            )

    print(
        f"Results: {num_correct}/{num_pixels} with accuracy {num_correct / num_pixels * 100:.4f}"
    )
    print(f"Dice score: {dice_score / len(data_loader)}")
    model.train()
