from src.data.datasets import build_dataloaders


def test_build_dataloaders_mnist() -> None:
    config = {
        "dataset": {
            "name": "mnist",
            "image_size": [28, 28],
            "channels": 1,
            "num_classes": 10,
            "condition_mode": "one_hot",
        },
        "training": {
            "batch_size": 16,
        },
    }
    train_loader, val_loader, test_loader, dataset_info = build_dataloaders(config)
    batch = next(iter(train_loader))
    x, y = batch
    assert x.shape[1:] == (1, 28, 28)
    assert y.shape[0] == x.shape[0]
    assert len(dataset_info.class_names) == 10
    assert dataset_info.channels == 1
