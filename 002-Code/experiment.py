import os.path
import sys
import utils
import argparse
from parser import Parser


class Experiment(object):
    def __init__(self):
        if len(sys.argv) <= 2:
            utils.error('No arguments given. Use --help for usage information.')

        # Parse parameters
        self.parameters = Parser().parse_arguments()
        self.model = None
        self.name = f'{"-".join(self.parameters.input_features)}'
        if 'architecture_name' in self.parameters:
            self.name += f'_to_{"-".join(self.parameters.ground_truth)}'
            self.name += f'_{utils.module_name_to_class_name(self.parameters.architecture_name)}'
        else:
            if self.parameters.model_train == 'train':
                self.name += f'_to_{"-".join(self.parameters.ground_truth)}'
                self.name += f'_pretrained_{os.path.splitext(os.path.basename(self.parameters.model_path))[0]}'
            else:
                pass
        self.parameters.output = os.path.join(self.parameters.output, self.name)


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
        # Variables to make code easier
        is_one_fold = experiment.parameters.folds == 1
        dataset_is_large = experiment.parameters.large_dataset

        # Split the dataset into k_folds
        if dataset_is_large or is_one_fold:
            dataset_path = utils.split_data(experiment.parameters.input,
                                            experiment_name=experiment.name,
                                            input_features=experiment.parameters.input_features,
                                            output_features=experiment.parameters.ground_truth,
                                            n_folds=experiment.parameters.folds)
        else:
            dataset_path = utils.k_fold_split(experiment.parameters.input,
                                              experiment_name=experiment.name,
                                              input_features=experiment.parameters.input_features,
                                              output_features=experiment.parameters.ground_truth,
                                              n_folds=experiment.parameters.folds)

        print(f"Dataset created in: {dataset_path}")
