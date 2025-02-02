"""Fitting script."""

import importlib

_CONFIG_PATH = 'configs.images'


def main():
    """Run fitting."""
    config = importlib.import_module(_CONFIG_PATH)
    trainer = config.get_trainer()
    trainer()
    

if __name__ == "__main__":
    main()