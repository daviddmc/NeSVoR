import torch
from typing import Dict, Tuple, Any
from argparse import Namespace
from ..image import Volume, save_slices, load_slices, load_stack
from ..nesvor.models import INR
from ..utils import merge_args


def inputs(args: Namespace) -> Tuple[Dict, Namespace]:
    input_dict: Dict[str, Any] = dict()
    if getattr(args, "input_stacks", None) is not None:
        input_dict["input_stacks"] = []
        for i, f in enumerate(args.input_stacks):
            stack = load_stack(
                f,
                args.stack_masks[i] if args.stack_masks is not None else None,
                device=args.device,
            )
            if args.thicknesses is not None:
                stack.thickness = args.thicknesses[i]
            input_dict["input_stacks"].append(stack)
    if getattr(args, "input_slices", None) is not None:
        input_dict["input_slices"] = load_slices(args.input_slices, args.device)
    if getattr(args, "input_model", None) is not None:
        cp = torch.load(args.input_model, map_location=args.device)
        input_dict["model"] = INR(cp["model"]["bounding_box"], cp["args"])
        input_dict["model"].load_state_dict(cp["model"])
        input_dict["mask"] = cp["mask"]
        args = merge_args(cp["args"], args)
    return input_dict, args


def outputs(data: Dict, args: Namespace) -> None:
    if getattr(args, "output_volume", None) and "output_volume" in data:
        if args.output_intensity_mean:
            data["output_volume"].rescale(args.output_intensity_mean)
        data["output_volume"].save(args.output_volume)
    if getattr(args, "output_model", None) and "output_model" in data:
        torch.save(
            {
                "model": data["output_model"].state_dict(),
                "mask": data["mask"],
                "args": args,
            },
            args.output_model,
        )
    if getattr(args, "output_slices", None) and "output_slices" in data:
        save_slices(args.output_slices, data["output_slices"])
    if getattr(args, "simulated_slices", None) and "simulated_slices" in data:
        save_slices(args.simulated_slices, data["simulated_slices"])


def load_model(args: Namespace) -> Tuple[INR, Volume, Namespace]:
    cp = torch.load(args.input_model, map_location=args.device)
    inr = INR(cp["model"]["bounding_box"], cp["args"])
    inr.load_state_dict(cp["model"])
    mask = cp["mask"]
    args = merge_args(cp["args"], args)
    return inr, mask, args
