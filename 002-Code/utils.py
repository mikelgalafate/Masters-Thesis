import argparse
import importlib
from tensorflow import keras


def load_model(args: argparse.Namespace, input_shape: tuple) -> keras.models.Model:
    if "architecture" in args and args.architecture is not None:
        module = importlib.import_module(f"networks.models.{args.architecture}")
        architecture = getattr(module, args.architecture)
        model = architecture(input_shape=input_shape).model
        model.compile()
    else:
        model = keras.models.load_model(f"{args.model}")

    return model


if __name__ == "__main__":
    pass
