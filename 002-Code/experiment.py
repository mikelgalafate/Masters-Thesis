import glob
import utils
import argparse


if __name__ == "__main__":

    argparse = argparse.ArgumentParser()
    available_architectures = [file.rsplit('/')[-1][:-3] for file in glob.glob('networks/models/*.py')
                               if 'Model.py'not in file and '__init__' not in file]
    if len(available_architectures) != 0:
        mode_group = argparse.add_mutually_exclusive_group(required=True)
        mode_group.add_argument("-a", "--architecture",
                                type=str, choices=available_architectures,
                                default=None,
                                help=f"Name of the architecture to be used.")
        mode_group.add_argument("-m", "--model",
                                type=str,
                                default=None,
                                help="Path to saved model")
    else:
        argparse.add_argument("-m", "--model",
                              type=str,
                              default=None,
                              help="Path to saved model",
                              required=True)
    argparse.add_argument("-v", "--verbose", action="store_true", help="Verbose network summary")

    args = argparse.parse_args()
    for arg in vars(args):
        atr = getattr(args, arg)
        print(f"{arg}: {atr}")

    # Load model
    model = utils.load_model(args, input_shape=(512, 512, 3))

    # Print model summary
    if args.verbose:
        model.summary()


