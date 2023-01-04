import argparse
import sys
from .commands import *
from ..utils import setup_logger
import torch
import random
import numpy as np
import string


class Formatter(
    argparse.ArgumentDefaultsHelpFormatter,
    argparse.RawDescriptionHelpFormatter,
):
    def __init__(self, prog: str) -> None:
        super().__init__(prog, max_help_position=50, width=None)


class FormatterMetavar(Formatter, argparse.MetavarTypeHelpFormatter):
    pass


def update_defaults(parser, **kwargs):
    parser.set_defaults(**kwargs)


def build_parser_training() -> argparse.ArgumentParser:
    _parser = argparse.ArgumentParser(add_help=False)
    parser = _parser.add_argument_group("model architecture")
    # hash grid encoding
    parser.add_argument(
        "--n-features-per-level",
        default=2,
        type=int,
        help="Length of the feature vector at each level.",
    )
    parser.add_argument(
        "--log2-hashmap-size",
        default=19,
        type=int,
        help="Max log2 size of the hash grid per level.",
    )
    parser.add_argument(
        "--level-scale",
        default=1.3819,
        type=float,
        help="Scaling factor between two levels.",
    )
    parser.add_argument(
        "--coarsest-resolution",
        default=16.0,
        type=float,
        help="Resolution of the coarsest grid in millimeter.",
    )
    parser.add_argument(
        "--finest-resolution",
        default=0.5,
        type=float,
        help="Resolution of the finest grid in millimeter.",
    )
    parser.add_argument(
        "--n-levels-bias",
        default=0,
        type=int,
        help="Number of levels used for bias field estimation.",
    )
    # implicit network
    parser.add_argument(
        "--depth", default=1, type=int, help="Number of hidden layers in MLPs."
    )
    parser.add_argument(
        "--width", default=64, type=int, help="Number of neuron in each hidden layer."
    )
    parser.add_argument(
        "--n-features-z",
        default=15,
        type=int,
        help="Length of the intermediate feature vector z.",
    )
    parser.add_argument(
        "--n-features-slice",
        default=16,
        type=int,
        help="Length of the slice embedding vector e.",
    )
    parser.add_argument(
        "--no-transformation-optimization",
        action="store_true",
        help="Disable optimization for rigid slice transfromation, i.e., the slice transformations are fixed",
    )
    parser.add_argument(
        "--no-slice-scale",
        action="store_true",
        help="Disable adaptive scaling for slices.",
    )
    parser.add_argument(
        "--no-pixel-variance",
        action="store_true",
        help="Disable pixel-level variance.",
    )
    parser.add_argument(
        "--no-slice-variance",
        action="store_true",
        help="Disable slice-level variance.",
    )
    parser.add_argument(
        "--single-precision",
        action="store_true",
        help="use float32 (default: float16)",
    )
    # loss function
    parser = _parser.add_argument_group("loss function")
    parser.add_argument(
        "--weight-transformation",
        default=0.1,
        type=float,
        help="Weight of transformation regularization.",
    )
    parser.add_argument(
        "--weight-bias",
        default=100.0,
        type=float,
        help="Weight of bias field regularization.",
    )
    parser.add_argument(
        "--image-regularization",
        default="edge",
        type=str,
        choices=["TV", "edge", "L2"],
        help="Type of image regularization (TV: total variation, edge: edge-preserving, L2: L2 regularization of image gradient).",
    )
    parser.add_argument(
        "--weight-image",
        default=1.0 * 2,
        type=float,
        help="Weight of image regularization.",
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=0.2,
        help="Parameter to define intensity of an edge in edge-preserving regularization.",
    )
    # training
    parser = _parser.add_argument_group("training")
    parser.add_argument(
        "--learning-rate",
        default=5e-3,
        type=float,
        help="Learning rate of Adam optimizer.",
    )
    parser.add_argument(
        "--gamma",
        default=0.33,
        type=float,
        help="Multiplicative factor of learning rate decay.",
    )
    parser.add_argument(
        "--milestones",
        nargs="+",
        type=float,
        default=[0.5, 0.75, 0.9],
        help="List of milestones of learning rate decay. Must be in (0, 1) and increasing.",
    )
    parser.add_argument(
        "--n-iter", default=6000, type=int, help="Number of iterations for training."
    )
    parser.add_argument(
        "--batch-size", default=1024 * 4, type=int, help="Batch size for training."
    )
    parser.add_argument(
        "--n-samples",
        default=128 * 2,
        type=int,
        help="Number of sample for PSF during training.",
    )
    return _parser


def build_parser_inputs(
    input_stacks=False, input_slices=False, input_model=False
) -> argparse.ArgumentParser:
    _parser = argparse.ArgumentParser(add_help=False)
    parser = _parser.add_argument_group("input")
    # stack input
    if input_stacks:
        parser.add_argument(
            "--input-stacks",
            nargs="+",
            type=str,
            required=input_stacks == "required",
            help="Paths to the input stacks (NIfTI).",
        )
        parser.add_argument(
            "--thicknesses",
            nargs="+",
            type=float,
            help="Slice thickness of each input stack. Use the slice gap in the input stack if not provided.",
        )
        parser.add_argument(
            "--stack-masks", nargs="+", type=str, help="Paths to masks of input stacks."
        )
    # slices input
    if input_slices:
        parser.add_argument(
            "--input-slices",
            type=str,
            required=input_slices == "required",
            help="Folder of the input slices.",
        )
    # input model
    if input_model:
        parser.add_argument(
            "--input-model",
            type=str,
            required=input_model == "required",
            help="Path to the trained NeSVoR model.",
        )
    return _parser


def build_parser_outputs(
    output_volume=False,
    output_slices=False,
    simulate_slices=False,
    output_model=False,
    **kwargs,
) -> argparse.ArgumentParser:
    _parser = argparse.ArgumentParser(add_help=False)
    parser = _parser.add_argument_group("output")
    # output volume
    if output_volume:
        parser.add_argument(
            "--output-volume",
            type=str,
            required=output_volume == "required",
            help="Paths to the reconstructed volume",
        )
        parser.add_argument(
            "--output-resolution",
            default=0.8,
            type=float,
            help="Isotropic resolution of the reconstructed volume",
        )
        parser.add_argument(
            "--output-intensity-mean",
            default=700.0,
            type=float,
            help="mean intensity of the output volume",
        )
        parser.add_argument(
            "--inference-batch-size", type=int, help="batch size for inference"
        )
        parser.add_argument(
            "--n-inference-samples",
            type=int,
            help="number of sample for PSF during inference",
        )
        parser.add_argument(
            "--no-output-psf",
            action="store_true",
            help="Disable psf for generating output volume",
        )
    # output slices
    if output_slices:
        parser.add_argument(
            "--output-slices",
            required=output_slices == "required",
            type=str,
            help="Folder to save the motion corrected slices",
        )
    # simulate slices
    if simulate_slices:
        parser.add_argument(
            "--simulated-slices",
            required=simulate_slices == "required",
            type=str,
            help="Folder to save the simulated slices from the reconstructed volume",
        )
    # output model
    if output_model:
        parser.add_argument(
            "--output-model",
            type=str,
            required=output_model == "required",
            help="Path to save the output model (.pt)",
        )
    parser.add_argument("--mask-threshold", type=float, default=1.0)
    update_defaults(_parser, **kwargs)
    return _parser


def build_parser_svort() -> argparse.ArgumentParser:
    _parser = argparse.ArgumentParser(add_help=False)
    parser = _parser.add_argument_group("registration")
    parser.add_argument(
        "--registration",
        default="svort",
        type=str,
        choices=["svort", "svort-stack", "stack", "none"],
        help="The type of registration method applied before reconstruction. svort: the full SVoRT model, svort-stack: only apply stack transformations of SVoRT, stack: stack-to-stack rigid registration, none: no registration.",
    )
    parser.add_argument(
        "--svort-version",
        default="v1",
        type=str,
        choices=["v1", "v2"],
        help="version of SVoRT",
    )
    return _parser


def build_parser_common() -> argparse.ArgumentParser:
    _parser = argparse.ArgumentParser(add_help=False)
    parser = _parser.add_argument_group("common")
    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        choices=[0, 1, 2],
        help="level of verbosity: (0: warning/error, 1: info, 2: debug)",
    )
    parser.add_argument("--output-log", type=str, help="Path to the output log file")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--debug", action="store_true", help="Debug mode.")
    return _parser


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="nesvor",
        description="NeSVoR: a toolkit for neural slice-to-volume reconstruction",
        epilog="Run 'nesvor COMMAND --help' for more information on a command.\n\n"
        + "To learn more about NeSVoR, check out our repo at "
        + "https://github.com/daviddmc/NeSVoR",
        formatter_class=Formatter,
        add_help=False,
    )
    parser.add_argument("-h", "--help", action="help", help=argparse.SUPPRESS)
    subparsers = parser.add_subparsers(title="commands", metavar=None, dest="command")
    # reconstruct
    parser_common = build_parser_common()
    parser_svort = build_parser_svort()
    parser_reconstruct = subparsers.add_parser(
        "reconstruct",
        help="slice-to-volume reconstruction using NeSVoR",
        description="slice-to-volume reconstruction using NeSVoR",
        parents=[
            build_parser_inputs(input_stacks=True, input_slices=True),
            build_parser_outputs(
                output_volume=True,
                output_slices=True,
                simulate_slices=True,
                output_model=True,
            ),
            parser_svort,
            build_parser_training(),
            parser_common,
        ],
        formatter_class=FormatterMetavar,
        add_help=False,
    )
    parser_reconstruct.add_argument(
        "-h", "--help", action="help", help=argparse.SUPPRESS
    )
    # sample-volume
    parser_sample_volume = subparsers.add_parser(
        "sample-volume",
        help="sample a volume from a trained NeSVoR model",
        description="sample a volume from a trained NeSVoR model",
        parents=[
            build_parser_inputs(input_model="required"),
            build_parser_outputs(
                output_volume="required",
                inference_batch_size=1024 * 4 * 8,
                n_inference_samples=128 * 2 * 2,
            ),
            parser_common,
        ],
        formatter_class=FormatterMetavar,
        add_help=False,
    )
    parser_sample_volume.add_argument(
        "-h", "--help", action="help", help=argparse.SUPPRESS
    )
    # sample-slices
    parser_sample_slices = subparsers.add_parser(
        "sample-slices",
        help="sample slices from a trained NeSVoR model",
        description="sample slices from a trained NeSVoR model",
        parents=[
            build_parser_inputs(input_slices="required", input_model="required"),
            build_parser_outputs(simulate_slices="required"),
            parser_common,
        ],
        formatter_class=FormatterMetavar,
        add_help=False,
    )
    parser_sample_slices.add_argument(
        "-h", "--help", action="help", help=argparse.SUPPRESS
    )
    # register
    parser_register = subparsers.add_parser(
        "register",
        help="slice-to-volume registration using SVoRT",
        description="slice-to-volume registration using SVoRT",
        parents=[
            build_parser_inputs(input_stacks="required"),
            build_parser_outputs(output_slices="required"),
            parser_svort,
            parser_common,
        ],
        formatter_class=FormatterMetavar,
        add_help=False,
    )
    parser_register.add_argument("-h", "--help", action="help", help=argparse.SUPPRESS)
    # parse arguments
    if len(sys.argv) == 1:
        parser.print_help(sys.stdout)
        return
    args = parser.parse_args()
    if len(sys.argv) == 2:
        locals()["parser_" + args.command.replace("-", "_")].print_help(sys.stdout)
        return
    device = torch.device(0)
    args.device = device
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)

    setup_logger(args.output_log, args.verbose)
    command_class = "".join(string.capwords(w) for w in args.command.split("-"))
    globals()[command_class](args).main()


if __name__ == "__main__":
    main()
