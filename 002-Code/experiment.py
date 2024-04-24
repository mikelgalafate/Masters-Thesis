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
    if experiment.parameters.verbose:
        experiment.model.summary()

    # Train the model
    if experiment.parameters.train:

        # Split the dataset into k_folds
        if hasattr(experiment.parameters, 'large_dataset') or experiment.parameters.folds == 1:
            dataset_path = utils.split_data(experiment.parameters.input,
                                            input_features=experiment.parameters.input_features,
                                            output_features=experiment.parameters.ground_truth,
                                            n_folds=experiment.parameters.folds)
        else:
            dataset_path = utils.k_fold_split(experiment.parameters.input,
                                              input_features=experiment.parameters.input_features,
                                              output_features=experiment.parameters.ground_truth,
                                              n_folds=experiment.parameters.folds)

        print(f"Dataset created in: {dataset_path}")
