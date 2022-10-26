import time
import argparse
import logging
import re
from typing import List, Optional, Tuple
from ..image import Stack, Slice
from ..svort.inference import svort_predict
from ..nesvor.train import train
from ..nesvor.sample import sample_volume, sample_slices
from .io import outputs, inputs
from ..utils import makedirs, log_args


class Command(object):
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.timer: List[Tuple[Optional[str], float]] = []

    def check_args(self) -> None:
        pass

    def get_command(self) -> str:
        return "-".join(
            w.lower() for w in re.findall("[A-Z][^A-Z]*", self.__class__.__name__)
        )

    def new_timer(self, name: Optional[str] = None) -> None:
        t = time.time()
        if len(self.timer) > 1 and self.timer[-1][0] is not None:
            # the previous timer ends
            logging.info(
                "%s finished in %.1f s", self.timer[-1][0], t - self.timer[-1][1]
            )
        if name is None:
            if len(self.timer) == 0:  # begining of command
                pass
            else:  # end of command
                logging.info(
                    "Command 'nesvor %s' finished, overall time: %.1f s",
                    self.get_command(),
                    t - self.timer[0][1],
                )
        else:
            logging.info("%s starts ...", name)
        self.timer.append((name, t))

    def makedirs(self) -> None:
        keys = ["output_slices", "simulated_slices"]
        makedirs([getattr(self.args, k, None) for k in keys])

    def main(self) -> None:
        self.check_args()
        log_args(self.args)
        self.makedirs()
        self.new_timer()
        self.exec()
        self.new_timer()

    def exec(self) -> None:
        raise NotImplementedError("The exec method for Command is not implemented.")


class Reconstruct(Command):
    def check_args(self) -> None:
        assert (
            self.args.input_slices is not None or self.args.input_stacks is not None
        ), "No image data provided! Use --input-slices or --input-stacks to input data."
        if self.args.input_slices is not None:
            # use input slices
            if (
                self.args.stack_masks is not None
                or self.args.input_stacks is not None
                or self.args.thicknesses is not None
            ):
                logging.warning(
                    "Since <input-slices> is provided, <input-stacks>, <stack_masks> and <thicknesses> would be ignored."
                )
                self.args.stack_masks = None
                self.args.input_stacks = None
                self.args.thicknesses = None
        else:
            # use input stacks
            if self.args.stack_masks is not None:
                assert len(self.args.stack_masks) == len(
                    self.args.input_stacks
                ), "The numbers of stack masks and input stacks are different!"
            if self.args.thicknesses is not None:
                assert len(self.args.thicknesses) == len(
                    self.args.input_stacks
                ), "The numbers of thicknesses and input stacks are different!"
        if self.args.output_volume is None and self.args.output_model is None:
            logging.warning("Both <output-volume> and <output-model> are not provided.")
        if not self.args.inference_batch_size:
            self.args.inference_batch_size = 8 * self.args.batch_size
        if not self.args.n_inference_samples:
            self.args.n_inference_samples = 2 * self.args.n_samples

    def exec(self) -> None:
        self.new_timer("Data loading")
        input_dict, args = inputs(self.args)
        if "input_stacks" in input_dict and input_dict["input_stacks"]:
            self.new_timer("Registration")
            slices = register(args, input_dict["input_stacks"])
        elif "input_slices" in input_dict and input_dict["input_slices"]:
            slices = input_dict["input_slices"]
        else:
            raise ValueError("No data found!")
        self.new_timer("Reconsturction")
        model, output_slices, mask = train(slices, args)
        self.new_timer("Results saving")
        output_volume = sample_volume(model, mask, args)
        simulated_slices = sample_slices(model, output_slices, mask, args)
        outputs(
            {
                "output_volume": output_volume,
                "mask": mask,
                "output_model": model,
                "output_slices": output_slices,
                "simulated_slices": simulated_slices,
            },
            args,
        )


class SampleVolume(Command):
    def exec(self) -> None:
        self.new_timer("Data loading")
        input_dict, args = inputs(self.args)
        self.new_timer("Volume sampling")
        v = sample_volume(input_dict["model"], input_dict["mask"], args)
        self.new_timer("Results saving")
        outputs({"output_volume": v}, args)


class SampleSlices(Command):
    def exec(self) -> None:
        self.new_timer("Data loading")
        input_dict, args = inputs(self.args)
        self.new_timer("Slices sampling")
        simulated_slices = sample_slices(
            input_dict["model"], input_dict["input_slices"], input_dict["mask"], args
        )
        self.new_timer("Results saving")
        outputs({"simulated_slices": simulated_slices}, args)


class Register(Command):
    def check_args(self) -> None:
        if self.args.stack_masks is not None:
            assert len(self.args.stack_masks) == len(
                self.args.input_stacks
            ), "The numbers of stack masks and input stacks are different!"
        if self.args.thicknesses is not None:
            assert len(self.args.thicknesses) == len(
                self.args.input_stacks
            ), "The numbers of thicknesses and input stacks are different!"

    def exec(self) -> None:
        self.new_timer("Data loading")
        input_dict, args = inputs(self.args)
        if not ("input_stacks" in input_dict and input_dict["input_stacks"]):
            raise ValueError("No data found!")
        self.new_timer("Registration")
        slices = register(args, input_dict["input_stacks"])
        self.new_timer("Results saving")
        outputs({"output_slices": slices}, args)


def register(args: argparse.Namespace, data: List[Stack]) -> List[Slice]:
    svort = args.registration == "svort" or args.registration == "svort-stack"
    vvr = args.registration != "none"
    force_vvr = args.registration == "svort-stack"
    slices = svort_predict(data, args.device, svort, vvr, force_vvr)
    return slices
