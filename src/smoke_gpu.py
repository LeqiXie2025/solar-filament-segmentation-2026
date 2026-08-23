import torch

import config
from src.losses import DiceBCELoss
from src.model import UNet


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")
    if device.type == "cuda":
        print(f"gpu={torch.cuda.get_device_name(0)}")

    model = UNet(in_channels=config.IN_CHANNELS, base_channels=16).to(device)
    images = torch.rand(1, config.IN_CHANNELS, 256, 256, device=device)
    masks = (torch.rand(1, 1, 256, 256, device=device) > 0.97).float()
    criterion = DiceBCELoss()

    logits = model(images)
    loss = criterion(logits, masks)
    loss.backward()
    print(f"logits_shape={tuple(logits.shape)} loss={float(loss.detach().cpu()):.5f}")


if __name__ == "__main__":
    main()
