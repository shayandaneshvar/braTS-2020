import torch
from tqdm import tqdm


# must load the model to the device and also prep the loss function with model parameters before using this function
# also do not forget to initialize the optimizer with the desired learning rate
def train(model, epochs=1, training_loader=None, loss_fn=None, device=None,
          optimizer: torch.optim.Optimizer = None, from_epoch=0):  # from_epoch is for the resume mode
    for epoch in range(from_epoch, epochs):
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
            y = y.to(device)  # .unsqueeze(1)
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


# our labels
# 1: Necrotic and Non-enhancing tumor core (NCR/NET)
# 2: Peritumoral Edema (ED)
# 3: GD-enhancing tumor (ET)
# common stuff in other papers:
# TC -> 1 + 3
# ET -> 3
# WT -> TC + ED: all
def check_accuracy_v2(data_loader, model, device="cuda"):
    num_correct = {'TC': 0, 'ET': 0, 'WT': 0}
    num_pixels = {'TC': 0, 'ET': 0, 'WT': 0}
    dice_score = {'TC': 0, 'ET': 0, 'WT': 0}
    model.eval()

    with torch.no_grad():
        for x, y in data_loader:
            x = x.to(device)
            y = y.to(device)  # .unsqueeze(1)
            preds = torch.sigmoid(model(x))
            preds = (preds >= 0.5).float()
            TC_pred = (preds[:, 0] + preds[:, 2] >= 1).float()
            TC_real = (y[:, 0] + y[:, 2] >= 1).float()
            ET_pred = preds[:, 2]
            ET_real = y[:, 2]
            WT_pred = (preds[:, 0] + preds[:, 1] + preds[:, 2] >= 1).float()
            WT_real = (y[:, 0] + y[:, 1] + y[:, 2] >= 1).float()

            num_correct['TC'] += (TC_pred == TC_real).sum()
            num_pixels['TC'] += torch.numel(TC_pred)
            dice_score['TC'] += (2 * (TC_pred * TC_real).sum()) / (
                    (TC_pred + TC_real).sum() + 1e-8)

            num_correct['ET'] += (ET_pred == ET_real).sum()
            num_pixels['ET'] += torch.numel(ET_pred)
            dice_score['ET'] += (2 * (ET_pred * ET_real).sum()) / (
                    (ET_pred + ET_real).sum() + 1e-8)

            num_correct['WT'] += (WT_pred == WT_real).sum()
            num_pixels['WT'] += torch.numel(WT_pred)
            dice_score['WT'] += (2 * (WT_pred * WT_real).sum()) / (
                    (WT_pred + WT_real).sum() + 1e-8)

    # print(
    #     f"Results (TC,ET,WT): ({num_correct['TC']}/{num_pixels['TC']}) with accuracy {num_correct / num_pixels * 100:.4f}"
    # )

    print(
        f" Accuracy (TC,ET,WT): \n --> {num_correct['TC'] / num_pixels['TC'] * 100:.4f} , {num_correct['ET'] / num_pixels['ET'] * 100:.4f}, {num_correct['WT'] / num_pixels['WT'] * 100:.4f}")
    print(
        f"Dice Score (TC,ET,WT): \n {dice_score['TC'] / len(data_loader)} , {dice_score['ET'] / len(data_loader)}, {dice_score['WT'] / len(data_loader)}")
    model.train()


def check_accuracy_v2(data_loader, model, device="cuda"):
    num_correct = {'TC': 0, 'ET': 0, 'WT': 0}
    num_pixels = {'TC': 0, 'ET': 0, 'WT': 0}
    dice_score = {'TC': 0, 'ET': 0, 'WT': 0}
    model.eval()

    with torch.no_grad():
        for x, y in data_loader:
            x = x.to(device)
            y = y.to(device)  # .unsqueeze(1)
            preds = torch.sigmoid(model(x))
            preds = (preds >= 0.5).float()
            TC_pred = (preds[:, 0] + preds[:, 2] >= 1).float()
            TC_real = (y[:, 0] + y[:, 2] >= 1).float()
            ET_pred = preds[:, 2]
            ET_real = y[:, 2]
            WT_pred = (preds[:, 0] + preds[:, 1] + preds[:, 2] >= 1).float()
            WT_real = (y[:, 0] + y[:, 1] + y[:, 2] >= 1).float()

            num_correct['TC'] += (TC_pred == TC_real).sum()
            num_pixels['TC'] += torch.numel(TC_pred)
            dice_score['TC'] += (2 * (TC_pred * TC_real).sum()) / (
                    (TC_pred + TC_real).sum() + 1e-8)

            num_correct['ET'] += (ET_pred == ET_real).sum()
            num_pixels['ET'] += torch.numel(ET_pred)
            dice_score['ET'] += (2 * (ET_pred * ET_real).sum()) / (
                    (ET_pred + ET_real).sum() + 1e-8)

            num_correct['WT'] += (WT_pred == WT_real).sum()
            num_pixels['WT'] += torch.numel(WT_pred)
            dice_score['WT'] += (2 * (WT_pred * WT_real).sum()) / (
                    (WT_pred + WT_real).sum() + 1e-8)

    # print(
    #     f"Results (TC,ET,WT): ({num_correct['TC']}/{num_pixels['TC']}) with accuracy {num_correct / num_pixels * 100:.4f}"
    # )

    print(
        f" Accuracy (TC,ET,WT): \n --> {num_correct['TC'] / num_pixels['TC'] * 100:.4f} , {num_correct['ET'] / num_pixels['ET'] * 100:.4f}, {num_correct['WT'] / num_pixels['WT'] * 100:.4f}")
    print(
        f"Dice Score (TC,ET,WT): \n {dice_score['TC'] / len(data_loader)} , {dice_score['ET'] / len(data_loader)}, {dice_score['WT'] / len(data_loader)}")
    model.train()


def check_accuracy_v3(data_loader, model, device="cuda"):  # requires full masks
    num_correct = {'TC': 0, 'ET': 0, 'WT': 0}
    num_pixels = {'TC': 0, 'ET': 0, 'WT': 0}
    dice_score = {'TC': 0, 'ET': 0, 'WT': 0}
    model.eval()

    with torch.no_grad():
        for x, y1 in data_loader:
            y = y1[:, :, ::2, ::2, ::2]  # get the halved mask

            x = x.to(device)
            y = y.to(device)  # .unsqueeze(1)
            y, y1 = y1, y
            original_preds = torch.sigmoid(model(x))

            preds = simple_trilinear_interpolation(original_preds)  # here or next?

            preds = (preds >= 0.5).float()
            TC_pred = (preds[:, 0] + preds[:, 2] >= 1).float()
            TC_real = (y[:, 0] + y[:, 2] >= 1).float()
            ET_pred = preds[:, 2]
            ET_real = y[:, 2]
            WT_pred = (preds[:, 0] + preds[:, 1] + preds[:, 2] >= 1).float()
            WT_real = (y[:, 0] + y[:, 1] + y[:, 2] >= 1).float()

            num_correct['TC'] += (TC_pred == TC_real).sum()
            num_pixels['TC'] += torch.numel(TC_pred)
            dice_score['TC'] += (2 * (TC_pred * TC_real).sum()) / (
                    (TC_pred + TC_real).sum() + 1e-8)

            num_correct['ET'] += (ET_pred == ET_real).sum()
            num_pixels['ET'] += torch.numel(ET_pred)
            dice_score['ET'] += (2 * (ET_pred * ET_real).sum()) / (
                    (ET_pred + ET_real).sum() + 1e-8)

            num_correct['WT'] += (WT_pred == WT_real).sum()
            num_pixels['WT'] += torch.numel(WT_pred)
            dice_score['WT'] += (2 * (WT_pred * WT_real).sum()) / (
                    (WT_pred + WT_real).sum() + 1e-8)

    # print(
    #     f"Results (TC,ET,WT): ({num_correct['TC']}/{num_pixels['TC']}) with accuracy {num_correct / num_pixels * 100:.4f}"
    # )

    print(
        f" Accuracy (TC,ET,WT): \n --> {num_correct['TC'] / num_pixels['TC'] * 100:.4f} , {num_correct['ET'] / num_pixels['ET'] * 100:.4f}, {num_correct['WT'] / num_pixels['WT'] * 100:.4f}")
    print(
        f"Dice Score (TC,ET,WT): \n {dice_score['TC'] / len(data_loader)} , {dice_score['ET'] / len(data_loader)}, {dice_score['WT'] / len(data_loader)}")
    model.train()


def simple_trilinear_interpolation(inputs):
    # very simple function to rescale the 3d image to double the size using trilinear interpolation basics
    assert inputs.shape[2:] == (64, 64, 64), "wrong input shape"
    result = torch.zeros((inputs.shape[0], inputs.shape[1], 128, 128, 128)).float()

    upsampler = torch.nn.Upsample(size=(127, 127, 127), mode="trilinear", align_corners=True)
    result[:, :, :-1, :-1, :-1] = upsampler(inputs)
    return result
