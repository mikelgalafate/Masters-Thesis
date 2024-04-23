import os.path
import sys
import utils
import argparse
from parser import Parser


class Experiment(object):
    def __init__(self):
        if not len(sys.argv) > 1:
            utils.error("No arguments given. Use --help for usage information.")

        # Parse parameters
        self.parameters = Parser().parse_arguments()
        self.model = None


if __name__ == "__main__":
    # Create experiment object
    experiment = Experiment()

    for arg in vars(experiment.parameters):
        atr = getattr(experiment.parameters, arg)
        print(f"{arg}: {atr}")

    # Load model
    experiment.model = utils.load_model(experiment.parameters, input_shape=(512, 512, 512))

    # Print model summary
    if args.verbose:
        model.summary()


