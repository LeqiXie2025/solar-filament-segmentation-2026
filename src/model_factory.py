import config
from src.model import UNet


def create_model(
    arch: str,
    in_channels: int = config.IN_CHANNELS,
    base_channels: int = config.BASE_CHANNELS,
    encoder_name: str = "resnet34",
    encoder_weights: str | None = None,
):
    if arch == "custom_unet":
        return UNet(in_channels=in_channels, base_channels=base_channels)

    if arch == "smp_unet":
        import segmentation_models_pytorch as smp

        return smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=1,
            activation=None,
        )

    if arch == "smp_unetplusplus":
        import segmentation_models_pytorch as smp

        return smp.UnetPlusPlus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=1,
            activation=None,
        )

    if arch == "smp_deeplabv3plus":
        import segmentation_models_pytorch as smp

        return smp.DeepLabV3Plus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=1,
            activation=None,
        )

    raise ValueError(f"Unknown model architecture: {arch}")


def create_model_from_checkpoint(checkpoint: dict):
    return create_model(
        arch=checkpoint.get("arch", "custom_unet"),
        in_channels=int(checkpoint.get("in_channels", config.IN_CHANNELS)),
        base_channels=int(checkpoint.get("base_channels", config.BASE_CHANNELS)),
        encoder_name=checkpoint.get("encoder_name", "resnet34"),
        encoder_weights=None,
    )
