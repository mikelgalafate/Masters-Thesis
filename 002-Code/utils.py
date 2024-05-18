import argparse
import importlib
import os
import random
import shutil
from typing import Tuple

from sklearn.model_selection import KFold, ShuffleSplit
from tensorflow import keras
from tqdm import tqdm


ACCEPTED_FILES = {'h5': '(.h5)', 'json': '(.json)', 'pickle': '(*.pkl)'}


def _ignore_other_features(input_features, output_features):
    """Function to ignore patterns for shutil.copytree()

    Creates a new function that returns a list with the elements to be ignored by shutil.copytree()


    Args:
        input_features (list): List of the input features to be used.
        output_features (list): List of the output features to be used.

    Returns:
        function: Function that returns a list with the elements to be ignored by shutil.copytree()

    """
    words_to_accept = [*input_features, *output_features]

    def _ignore_files(path, names):
        """Function to ignore patterns for shutil.copytree()

        Receives the path and the names contained in the path and returns a list with the elements to be ignored by
        shutil.copytree()
        If a name is not contained in neither of the feature lists, it's added to the ignore list.
        Args:
            path (str): Path being analyzed
            names (list): Names contained in the path

        Returns:
            List of the elements to be ignored by shutil.copytree()
        """
        return [name for name in names if not any(word.lower() in name.lower() for word in words_to_accept)]

    return _ignore_files


def split_data(path: str,
               input_features: Tuple[str] = None,
               output_features: Tuple[str] = None,
               experiment_name: str = '',
               n_folds: int = 1,
               train_frac: float = 0.7,
               validation_frac: float = 0.15,
               test_frac: float = 0.15,
               random_state: int = 1) -> str:
    """Split data into n_folds and a test set to perform cross validation.

    Creates a test directory containing the test data with test_frac portion of the dataset.

    - n_folds == 1: the rest of the dataset is divided into train and validation

    - n_folds > 1: inside the n_folds directory, a directory is created for each split where the corresponding fold is
      inside the validation folder and the rest go to the train directory.

    Notes: This function is recommended only for larger datasets using the large_dataset flag or to divide the dataset
    into a single train-validation-test partition. For smaller datasets use the split_small_dataset function.

    Args:
        path (str): Path to the directory containing the data
        input_features (Tuple[str]): Input features to be copied
        output_features (Tuple[str]): Output features to be copied
        experiment_name (str): Name of the experiment for which the dataset will be used
        n_folds (int): Number of folds
        train_frac (float): Fraction of data used for training. If n_folds > 1, value will be ignored.
        validation_frac (float): Fraction of data used for validation. If n_folds > 1, value will be ignored.
        test_frac (float): Fraction of data used for testing
        random_state (int): Random state for shuffling
    Returns:
        output_dir (str): Path to the output directory where the splits will be created
    Examples:
        - If n_folds == 1, the structure of the dataset will be:

        Train

        Validation

        Test

        - For a 5-fold cross-validation, the dataset is divided with the following structure:

        split_1: T T T T S

        split_2: T T T S T

        split_3: T T S T T

        split_4: T S T T T

        split_5: S T T T T

        Test

        With T meaning train V validation
    """
    if not os.path.exists(path):
        error("Path to dataset does not exist")
    if n_folds < 1:
        error("Number of folds must be greater than or equal to 1")
    elif n_folds == 1 and train_frac + validation_frac + test_frac != 1:
        error("Train/Valid/Test partitions must sum to 1")
    if test_frac >= 1 or test_frac <= 0:
        error("Test fraction must be 0 < test_frac < 1")
    if output_features is None:
        output_features = []
    if input_features is None:
        input_features = []

    output_dir = os.path.join(os.path.dirname(path),
                              experiment_name,
                              f'{n_folds}_Folds_test' if n_folds > 1 else 'dataset')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    elif os.listdir(output_dir):
        warning(f"Dataset directory \'{output_dir}\' already exists.## "
                f"While this may not indicate an error, if the dataset folder is not correctly created, "
                f"errors may occur while running.## "
                f"Removing the existing dataset directory before running is recommended.")
        return output_dir

    random.seed(random_state)
    print("\nSplitting data...")
    # Get list of subfolders in the data folder
    subfolders = sorted([directory for directory in os.listdir(path) if os.path.isdir(os.path.join(path, directory))])

    # Select the subjects for the test subset
    test_subjects = random.sample(subfolders, int(len(subfolders) * test_frac))
    ignore_other_features = _ignore_other_features(input_features, output_features)
    print("Copying the test subjects...")
    for subject in tqdm(test_subjects):
        shutil.copytree(os.path.join(path, subject),
                        os.path.join(os.path.join(output_dir, "test"), subject),
                        ignore=ignore_other_features)

    # Take the test subjects out of the subfolders list
    [subfolders.pop(subfolders.index(subject)) for subject in test_subjects]

    if n_folds > 1:
        print(f"{n_folds} folds will be created")
        # Split the data into n splits with train and test sets
        kf = KFold(n_splits=n_folds, random_state=random_state, shuffle=True)
        splits = kf.split(subfolders)
    else:
        # Split the data to train and
        ss = ShuffleSplit(n_splits=1, test_size=validation_frac, random_state=random_state)
        splits = ss.split(subfolders)

    for i, (train_index, test_index) in enumerate(splits):
        fold_directory = os.path.join(output_dir, f"split_{i + 1}" if n_folds > 1 else "")
        if not os.path.exists(fold_directory):
            os.makedirs(fold_directory)
        if n_folds > 1:
            print(f'\nCreating fold {i + 1}/{n_folds}')
        # Copy the Train set for the i_th split
        print('Creating train set...')
        for index in tqdm(train_index):
            shutil.copytree(os.path.join(path, subfolders[index]),
                            os.path.join(fold_directory, "train", subfolders[index]),
                            ignore=ignore_other_features)
        # Copy the Validation set for the i_th split
        print('Creating validation set...')
        for index in tqdm(test_index):
            # Copy data from data_folder to fold_directory
            shutil.copytree(os.path.join(path, subfolders[index]),
                            os.path.join(fold_directory, "validation", subfolders[index]),
                            ignore=ignore_other_features)

    return output_dir


def k_fold_split(path: str,
                 input_features: Tuple[str] = None,
                 output_features: Tuple[str] = None,
                 experiment_name: str = '',
                 n_folds: int = 5,
                 random_state: int = 1) -> str:
    """Split data into n_folds.

    Divides the dataset into n_folds. For each split, one fold is used for test,
    one for validation and the rest for train.
    Args:
        path (str): Path to the directory containing the data
        input_features (Tuple[str]): Input features to be copied
        output_features (Tuple[str]): Output features to be copied
        experiment_name (str): Name of the experiment for which the dataset will be used
        n_folds (int): Number of folds
        random_state (int): Random state for shuffling

    Returns:
        output_dir (str): Path to the output directory where the splits will be created

    Examples:
        For a 5-fold cross-validation, the dataset is divided with the following structure:

        split_1: T T T V S

        split_2: T T V S T

        split_3: T V S T T

        split_4: V S T T T

        split_5: S T T T V

        With T meaning train, V validation and S test

    """
    if not os.path.exists(path):
        error("Path to dataset does not exist")
    if n_folds <= 2:
        error("Number of folds must be greater than 2")
    if output_features is None:
        output_features = []
    if input_features is None:
        input_features = []

    output_dir = os.path.join(os.path.dirname(path), experiment_name, f'{n_folds}_Folds')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    elif os.listdir(output_dir):
        warning(f"Dataset directory \'{output_dir}\' already exists.## "
                f"While this may not indicate an error, if the dataset folder is not correctly created, "
                f"errors may occur while running.## "
                f"Removing the existing dataset directory before running is recommended.")
        return output_dir

    random.seed(random_state)

    # Get list of subfolders in the data folder
    subfolders = sorted([directory for directory in os.listdir(path) if os.path.isdir(os.path.join(path, directory))])

    # Split the data into n splits with train and test sets
    kf = KFold(n_splits=n_folds, random_state=random_state, shuffle=True)
    splits = kf.split(subfolders)

    print("\nSplitting data...")
    folds_list = [test_index for _, test_index in splits]
    ignore_other_features = _ignore_other_features(input_features, output_features)
    for i in range(n_folds):
        fold_directory = os.path.join(output_dir, f"split_{i + 1}")
        if not os.path.exists(fold_directory):
            os.makedirs(fold_directory)

        print(f'\nCreating fold {i + 1}/{n_folds}')
        print('Creating train set...')
        # Create a list with indexes not in fold i and (i - 1) for the training set
        train_folds = [list(item) for idx, item in enumerate(folds_list) if idx != i and idx != (i - 1) % n_folds]
        train_index = sum(train_folds, [])
        # Copy training set
        for index in tqdm(train_index):
            shutil.copytree(os.path.join(path, subfolders[index]),
                            os.path.join(fold_directory, "train", subfolders[index]),
                            ignore=ignore_other_features)
        # Create the validation set with the (i - 1)_th fold
        print('Creating validation set...')
        for index in tqdm(folds_list[i - 1]):
            shutil.copytree(os.path.join(path, subfolders[index]),
                            os.path.join(fold_directory, "validation", subfolders[index]),
                            ignore=ignore_other_features)
        # Create the test set with the i_th fold
        print('Creating test set...')
        for index in tqdm(folds_list[i]):
            shutil.copytree(os.path.join(path, subfolders[index]),
                            os.path.join(fold_directory, "test", subfolders[index]),
                            ignore=ignore_other_features)

    return output_dir


def module_name_to_class_name(string: str) -> str:
    """Changes module name in lower_case_with_underscores format to class name in CapWords format

    Args:
        string: name of the module

    Returns:
        name of the module in CapWords format

    Examples:
        >>> module_name_to_class_name('lowercase_with_underscores_module_with_class')
        'LowercaseWithUnderscoresModuleWithClass'

    """
    # Split the string by underscores
    words = string.split('_')

    # Capitalize each word using title() and join them back
    return ''.join(word.title() for word in words)


def load_model(argv: argparse.Namespace, input_shape: tuple = None, experiment_name: str = None) -> keras.models.Model:
    """Load or create model.

    Loads a saved model or creates an instance of an available model.
    Args:
        argv (argparse.Namespace): Namespace containing either the path to the model or the model name
        input_shape (tuple): Input shape for the model
        experiment_name (str): Name of the experiment to be run

    Returns:
        model (keras.Model): Keras model

    """
    if "architecture_name" in argv:
        number_of_inputs = input_shape[0]
        input_shape = input_shape[2:]
        module = importlib.import_module(f"networks.models.{argv.architecture_name}")
        module_class = getattr(module, module_name_to_class_name(argv.architecture_name))
        architecture = module_class(input_shape=input_shape,
                                    number_of_inputs=number_of_inputs,
                                    num_blocks=argv.num_blocks,
                                    learning_rate=argv.learning_rate,
                                    graph=argv.graph,
                                    experiment_name=experiment_name)
        x = np.random.random(input_shape)
        model = architecture.get_model()
        return model
    elif "model_path" in argv:
        check_path_is_model(argv.model_path)
        model = keras.models.load_model(argv.model_path)
        return model


def check_gpus(set_cpu_only):
    devices = tf.config.list_physical_devices("GPU")
    if devices:
        if set_cpu_only:
            tf.config.set_visible_devices([], "GPU")
        else:
            print(f'Available: {devices}')
    else:
        print('No GPUs available.')
def error(error_name: str):
    """Print an error message and exit

    Args:
        error_name: name of the error

    """
    message = "\nERROR: " + error_name + "\n       Aborting."
    exit(message)


def warning(message: str):
    """Print a warning message.

    Prints a warning message with a specific format.

    Notes:
        To create a new line in the message, use '## ', not '\\\\n'
    Examples:
        >>> warning('This is a warning.')
        <BLANKLINE>
        WARNING: This is a warning.
        <BLANKLINE>
        >>> warning('This is a warning example.## This is the second line of the warning. \\nThis is the same line.')
        <BLANKLINE>
        WARNING: This is a warning example.
                 This is the second line of the warning. This is the same line.
        <BLANKLINE>
    """
    message = message.replace('\n', '')
    warn = '\n         '.join(message.split('## '))
    print('\nWARNING:', warn + '\n')


if __name__ == '__main__':
    pass
    k_fold_split("../001-Data/CERMEP-IDB-MRXFDG_Database/NII_with_FDG_NAC/sourcedata_BC")
    split_data("../001-Data/CERMEP-IDB-MRXFDG_Database/NII_with_FDG_NAC/sourcedata_BC", n_folds=5)
