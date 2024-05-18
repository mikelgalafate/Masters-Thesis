"""Module for command-line argument parsing.

This module provides functionality for parsing command-line arguments using the argparse
library. Defines a custom action class 'CreateTrainArgument' to create a train=True or
train=False argument and another argument at the same time.

The 'Parser' class creates an argument parser with the needed arguments for two different
modes:

- 'architecture' mode: Arguments for training a new model from an architecture definition.
  This mode is not available if no architecture is found inside the networks.models
  package.

- 'model' mode: Arguments for working with a pre-saved model. Differentiates two sub-modes:
    - 'train': This mode is used to Re-train the model from the given weights.
    - 'predict': Use the model to predict.
"""

import glob
import utils
import argparse
from typing import List


class CreateTrainArgument(argparse.Action):
    """Custom argparse.Action that creates a new argument and a 'train' argument.

    This class is an implementation of argparse.Action. This action is used
    to create the argument that calls to the action and a second additional
    argument storing the value of train=True or train=False associated to
    the argument that activated the action.
    """

    def __init__(self, option_strings, dest, train, **kwargs):
        """Initialize the CreateTrainArgument instance.

        Args:
            option_strings (list): A list of option strings.
            dest (str): The name of the attribute to be associated with the parsed value.
            train (str): The predefined training mode value associated with the option.
            **kwargs: Additional keyword arguments.

        """
        super().__init__(option_strings, dest, **kwargs)
        self.train = train

    def __call__(self, parser, namespace, values, option_string=None):
        """Set the value for both the calling and the 'train' arguments.

        Args:
            parser (argparse.ArgumentParser): instance that calls the action.
            namespace (argparse.Namespace): instance to which the argument will be added.
            values (bool): value for the argument that called to the action.
            option_string (str): The option string

        """
        # Set the mode
        setattr(namespace, self.dest, values)
        # Set the private variable with a predefined value
        setattr(namespace, 'train', self.train)


class Parser:
    """A class that creates a new argument parser instance that parses the command line arguments.

    Methods:

    """

    def __init__(self):
        """Initialize the Parser instance."""
        self.parser_arguments = {"description": "Run experiments",
                                 "formatter_class": argparse.ArgumentDefaultsHelpFormatter}
        # Read available architectures to choose among them
        self.available_architectures = [file.rsplit('/')[-1][:-3] for file in glob.glob('networks/models/*.py')
                                        if 'architecture.py' not in file and '__init__' not in file]

    @staticmethod
    def string_to_list(string: str) -> List[str]:
        """
        Converts a string to a list
        Args:
            string: String containing a comma-separated list

        Returns: A list of strings

        """
        return sorted(string.split(','))

    @staticmethod
    def add_in_out_arguments(parser_: argparse.ArgumentParser):
        """Adds input/output arguments to a parser.

        Args:
            parser_ (argparse.ArgumentParser): Parser to add arguments.

        """
        io_group = parser_.add_argument_group('Input/Output arguments')
        io_group.add_argument('-i', '--input', dest='input', required=True,
                              action=CreateTrainArgument, train=False,
                              type=str, help='Path to directory containing input images')
        io_group.add_argument('-o', '--output', dest='output',
                              type=str, help='Path to output directory')
        io_group.add_argument('-if', '--input_features', dest='input_features', required=True,
                              type=Parser.string_to_list,
                              help='Comma separated list of input images')

    @staticmethod
    def add_train_arguments_to_parser(parser_: argparse.ArgumentParser):
        """Adds training arguments to the parser.

        Args:
            parser_ (argparse.ArgumentParser): Parser to add arguments.

        """
        train_group = parser_.add_argument_group('Training arguments')
        for group in parser_._action_groups:
            if group.title == 'Input/Output arguments':
                group.add_argument('-gt', '--ground_truth', required=True,
                                   action=CreateTrainArgument, train=True,
                                   type=Parser.string_to_list,
                                   help='Comma separated list of output images')
        train_group.add_argument("-e", "--epochs", type=int, default=500, help="Number of epochs")
        train_group.add_argument("-b", "--batch_size", type=int, default=16, help="Batch size")
        train_group.add_argument("-f", "--folds", type=int, default=1, help="Number of folds")
        train_group.add_argument("-s", "--early_stopping", action="store_true", help="Add early stopping")
        train_group.add_argument("-p", "--patience",
                                 type=int, default=30,
                                 help="Number of patience epochs for early stopping")
        train_group.add_argument("-l", "--learning_rate", type=float, default=0.001, help="Learning rate")
        train_group.add_argument("--large_dataset", action="store_true",
                                 help="Only use if the dataset is large enough for k_folds with separate test set")
        parser_.add_argument('-v', '--verbose', action='store_true', help='Display model summary')
        parser_.add_argument('--cpu', action='store_true', help='Use CPU only')

    def parse_arguments(self) -> argparse.Namespace:
        """Parse command line arguments.

        Creates the parser with the needed arguments and parses them. The structure of the parser is specified
        in the module documentation.

        Returns: argparse.Namespace with parsed arguments.

        """
        # If no architectures are installed, the epilog variable is set in the parser_arguments dict
        if not self.available_architectures:
            self.parser_arguments["epilog"] = "To enable the \'architecture\' mode, please install a new architecture."

        # Create the Parser with the version argument
        pars = argparse.ArgumentParser(**self.parser_arguments)

        # General optional arguments
        pars.add_argument('--version', action='version', version=f'%(prog)s {utils.__version__}')

        # Mode-specific  arguments

        # Create subparsers for different modes: architecture and model
        subparsers = pars.add_subparsers(title="available commands", help="\'sub-command\' -h for help")

        # Parser to use a pre-saved model with the positional 'model' argument
        model_parser = subparsers.add_parser('model',
                                             aliases=['m'],
                                             help=f'Use a pre-saved model', **self.parser_arguments)
        # Argument for path to the model
        model_parser.add_argument('model_path',
                                  action=CreateTrainArgument, train=False,
                                  type=str,
                                  help=f'Path to saved model. '
                                       f'Accepted file types: {list(utils.ACCEPTED_FILES.values())}')

        # Subparser for training a pre-saved model
        model_subparser = model_parser.add_subparsers(description="Run experiments",
                                                      dest="model_train",
                                                      help="sub-command help")
        train_parser = model_subparser.add_parser('train', help='Train a pre-saved model', **self.parser_arguments)

        # Parser for predicting with a pre-saved model
        predict_parser = model_subparser.add_parser('predict', help='Predict with pre-saved model',
                                                    **self.parser_arguments)

        # I/O arguments to predict with a pre-saved model
        self.add_in_out_arguments(predict_parser)
        predict_parser.add_argument('-v', '--verbose', action='store_true', help='Display model summary')
        predict_parser.add_argument('--cpu', action='store_true', help='Use CPU only')

        # Train pre-saved model arguments
        self.add_in_out_arguments(train_parser)
        self.add_train_arguments_to_parser(train_parser)

        # Parser to use an available architecture with the positional 'architecture' argument
        if self.available_architectures:
            architecture_parser = subparsers.add_parser('architecture',
                                                        aliases=['a'],
                                                        help=f'Train a preinstalled network. Available options: '
                                                             f'{self.available_architectures}',
                                                        **self.parser_arguments)
            # Choose the name of the architecture to be used
            architecture_parser.add_argument('architecture_name', choices=self.available_architectures)
            # I/O and Train arguments are always used in this mode
            self.add_in_out_arguments(architecture_parser)
            self.add_train_arguments_to_parser(architecture_parser)
            # Architecture-specific arguments
            architecture_group = architecture_parser.add_argument_group('Architecture options')
            architecture_group.add_argument('--num_blocks', type=int, default=4, help="Number of blocks")
            architecture_group.add_argument('-g', '--graph', action='store_true', help='Plot model graph')

        # Parse the arguments
        arguments = pars.parse_args()

        # If no output directory is provided, the output will be in the input directory
        if arguments.output is None:
            arguments.output = arguments.input

        return arguments
