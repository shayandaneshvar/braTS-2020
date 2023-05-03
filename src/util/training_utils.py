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


def get_all_metrics(data_loader, model, device="cuda"):  # requires full masks
    num_correct = {'TC': 0, 'ET': 0, 'WT': 0}
    num_true_predicts = {'TC': 0, 'ET': 0, 'WT': 0}
    num_true_labels = {'TC': 0, 'ET': 0, 'WT': 0}
    num_pixels = {'TC': 0, 'ET': 0, 'WT': 0}
    dice_score = {'TC': 0, 'ET': 0, 'WT': 0}
    iou_score = {'TC': 0, 'ET': 0, 'WT': 0}

    EPS = 1e-16
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
            TC_pred = (preds[:, 0] + preds[:, 2] > 0.9).float()
            TC_real = (y[:, 0] + y[:, 2] > 0.9).float()
            ET_pred = preds[:, 2]
            ET_real = y[:, 2]
            WT_pred = (preds[:, 0] + preds[:, 1] + preds[:, 2] > 0.9).float()
            WT_real = (y[:, 0] + y[:, 1] + y[:, 2] > 0.9).float()

            num_correct['TC'] += (TC_pred == TC_real).sum()
            num_pixels['TC'] += torch.numel(TC_pred)
            num_true_predicts['TC'] += (TC_pred > .9).sum()  # those that are 1
            num_true_labels['TC'] += (TC_real > .9).sum()
            dice_score['TC'] += (2 * (TC_pred * TC_real).sum()) / ((TC_pred + TC_real).sum() + EPS)
            iou_score['TC'] += (TC_pred * TC_real).sum() / ((TC_pred + TC_real).sum() - (TC_pred * TC_real).sum() + EPS)

            num_correct['ET'] += (ET_pred == ET_real).sum()
            num_pixels['ET'] += torch.numel(ET_pred)
            num_true_predicts['ET'] += (ET_pred > .9).sum()
            num_true_labels['ET'] += (ET_real > .9).sum()
            dice_score['ET'] += (2 * (ET_pred * ET_real).sum()) / ((ET_pred + ET_real).sum() + EPS)
            iou_score['ET'] += (ET_pred * ET_real).sum() / ((ET_pred + ET_real).sum() - (ET_pred * ET_real).sum() + EPS)

            num_correct['WT'] += (WT_pred == WT_real).sum()
            num_pixels['WT'] += torch.numel(WT_pred)
            num_true_predicts['WT'] += (WT_pred > .9).sum()
            num_true_labels['WT'] += (WT_real > .9).sum()
            dice_score['WT'] += (2 * (WT_pred * WT_real).sum()) / ((WT_pred + WT_real).sum() + EPS)
            iou_score['WT'] += (WT_pred * WT_real).sum() / ((WT_pred + WT_real).sum() - (WT_pred * WT_real).sum() + EPS)

    # precision: correct true predicts / all true predicts -> TP / (TP + FP)

    precision = {'TC': num_correct['TC'] / num_true_predicts['TC'] * 100,
                 'ET': num_correct['ET'] / num_true_predicts['ET'] * 100,
                 'WT': num_correct['WT'] / num_true_predicts['WT'] * 100}
    # recall: correct true predicts / all true labels -> TP / (TP + FN)

    recall = {'TC': num_correct['TC'] / num_true_labels['TC'] * 100,
              'ET': num_correct['ET'] / num_true_labels['ET'] * 100,
              'WT': num_correct['WT'] / num_true_labels['WT'] * 100}

    # f1 is actually the same as Dice Coefficient, or is it?
    f1 = {'TC': 2 * precision['TC'] * recall['TC'] / (precision['TC'] + recall['TC']),
          'ET': 2 * precision['ET'] * recall['ET'] / (precision['ET'] + recall['ET']),
          'WT': 2 * precision['WT'] * recall['WT'] / (precision['WT'] + recall['WT'])}

    print(
        f" Accuracy (TC,ET,WT): \n {num_correct['TC'] / num_pixels['TC'] * 100:.4f} , {num_correct['ET'] / num_pixels['ET'] * 100:.4f}, {num_correct['WT'] / num_pixels['WT'] * 100:.4f}")
    print(
        f"Dice Score (TC,ET,WT): \n {dice_score['TC'] / len(data_loader)} , {dice_score['ET'] / len(data_loader)}, {dice_score['WT'] / len(data_loader)}")
    print(
        f"IoU Score (TC,ET,WT): \n {iou_score['TC'] / len(data_loader)} , {iou_score['ET'] / len(data_loader)}, {iou_score['WT'] / len(data_loader)}")

    print(
        f" Precision (TC,ET,WT): \n --> {precision['TC'] / len(data_loader):.4f} , {precision['ET'] / len(data_loader):.4f}, {precision['WT'] / len(data_loader):.4f}")
    print(
        f" Recall (TC,ET,WT): \n --> {recall['TC'] / len(data_loader):.4f} , {recall['ET'] / len(data_loader):.4f}, {recall['WT'] / len(data_loader):.4f}")
    print(
        f" F1-score (TC,ET,WT): \n --> {f1['TC'] / len(data_loader):.4f} , {f1['ET'] / len(data_loader):.4f}, {f1['WT'] / len(data_loader):.4f}")

    model.train()

    return None

# This one gives true average on all, and is final
def get_all_metrics_2(data_loader, model, device="cuda"):  # requires full masks
    num_correct = {'TC': 0, 'ET': 0, 'WT': 0}
    num_true_predicts = {'TC': 0, 'ET': 0, 'WT': 0}
    num_true_labels = {'TC': 0, 'ET': 0, 'WT': 0}
    num_pixels = {'TC': 0, 'ET': 0, 'WT': 0}
    dice_score = {'TC': 0, 'ET': 0, 'WT': 0}
    iou_score = {'TC': 0, 'ET': 0, 'WT': 0}

    EPS = 1e-16
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
            for i in range(preds.shape[0]):
                TC_pred = (preds[i, 0] + preds[i, 2] > 0.9).float()
                TC_real = (y[i, 0] + y[i, 2] > 0.9).float()
                ET_pred = preds[i, 2]
                ET_real = y[i, 2]
                WT_pred = (preds[i, 0] + preds[i, 1] + preds[i, 2] > 0.9).float()
                WT_real = (y[i, 0] + y[i, 1] + y[i, 2] > 0.9).float()

                num_correct['TC'] += (TC_pred == TC_real).sum()
                num_pixels['TC'] += torch.numel(TC_pred)
                num_true_predicts['TC'] += (TC_pred > .9).sum()  # those that are 1
                num_true_labels['TC'] += (TC_real > .9).sum()
                dice_score['TC'] += (2 * (TC_pred * TC_real).sum()) / ((TC_pred + TC_real).sum() + EPS)
                iou_score['TC'] += (TC_pred * TC_real).sum() / (
                            (TC_pred + TC_real).sum() - (TC_pred * TC_real).sum() + EPS)

                num_correct['ET'] += (ET_pred == ET_real).sum()
                num_pixels['ET'] += torch.numel(ET_pred)
                num_true_predicts['ET'] += (ET_pred > .9).sum()
                num_true_labels['ET'] += (ET_real > .9).sum()
                dice_score['ET'] += (2 * (ET_pred * ET_real).sum()) / ((ET_pred + ET_real).sum() + EPS)
                iou_score['ET'] += (ET_pred * ET_real).sum() / (
                            (ET_pred + ET_real).sum() - (ET_pred * ET_real).sum() + EPS)

                num_correct['WT'] += (WT_pred == WT_real).sum()
                num_pixels['WT'] += torch.numel(WT_pred)
                num_true_predicts['WT'] += (WT_pred > .9).sum()
                num_true_labels['WT'] += (WT_real > .9).sum()
                dice_score['WT'] += (2 * (WT_pred * WT_real).sum()) / ((WT_pred + WT_real).sum() + EPS)
                iou_score['WT'] += (WT_pred * WT_real).sum() / (
                            (WT_pred + WT_real).sum() - (WT_pred * WT_real).sum() + EPS)

    # precision: correct true predicts / all true predicts -> TP / (TP + FP)

    precision = {'TC': num_correct['TC'] / num_true_predicts['TC'] * 100,
                 'ET': num_correct['ET'] / num_true_predicts['ET'] * 100,
                 'WT': num_correct['WT'] / num_true_predicts['WT'] * 100}
    # recall: correct true predicts / all true labels -> TP / (TP + FN)

    recall = {'TC': num_correct['TC'] / num_true_labels['TC'] * 100,
              'ET': num_correct['ET'] / num_true_labels['ET'] * 100,
              'WT': num_correct['WT'] / num_true_labels['WT'] * 100}

    # f1 is actually the same as Dice Coefficient, or is it?
    f1 = {'TC': 2 * precision['TC'] * recall['TC'] / (precision['TC'] + recall['TC']),
          'ET': 2 * precision['ET'] * recall['ET'] / (precision['ET'] + recall['ET']),
          'WT': 2 * precision['WT'] * recall['WT'] / (precision['WT'] + recall['WT'])}

    items_size = preds.shape[0] * len(data_loader)

    print(
        f" Accuracy (TC,ET,WT): \n {num_correct['TC'] / num_pixels['TC'] * 100:.4f} , {num_correct['ET'] / num_pixels['ET'] * 100:.4f}, {num_correct['WT'] / num_pixels['WT'] * 100:.4f}")
    print(
        f"Dice Score (TC,ET,WT): \n {dice_score['TC'] / items_size} , {dice_score['ET'] / items_size}, {dice_score['WT'] / items_size}")
    print(
        f"IoU Score (TC,ET,WT): \n {iou_score['TC'] / items_size} , {iou_score['ET'] / items_size}, {iou_score['WT'] / items_size}")

    print(
        f" Precision (TC,ET,WT): \n --> {precision['TC'] / items_size:.4f} , {precision['ET'] / items_size:.4f}, {precision['WT'] / items_size:.4f}")
    print(
        f" Recall (TC,ET,WT): \n --> {recall['TC'] / items_size:.4f} , {recall['ET'] / items_size:.4f}, {recall['WT'] / items_size:.4f}")
    print(
        f" F1-score (TC,ET,WT): \n --> {f1['TC'] / items_size:.4f} , {f1['ET'] / items_size:.4f}, {f1['WT'] / items_size:.4f}")

    model.train()

    return None


def simple_trilinear_interpolation(inputs):
    # very simple function to rescale the 3d image to double the size using trilinear interpolation basics
    assert inputs.shape[2:] == (64, 64, 64), "wrong input shape"
    result = torch.zeros((inputs.shape[0], inputs.shape[1], 128, 128, 128)).float()

    upsampler = torch.nn.Upsample(size=(127, 127, 127), mode="trilinear", align_corners=True)
    result[:, :, :-1, :-1, :-1] = upsampler(inputs)
    return result
