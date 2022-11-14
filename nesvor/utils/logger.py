from typing import Callable, List, Any, Optional
import torch
from argparse import Namespace
import logging
import traceback
import sys


class LazyLog(object):
    def __init__(self, func: Callable[..., Any], *args, **kwargs) -> None:
        self.func = func
        self.args = args
        self.kwargs = kwargs

    def __str__(self) -> str:
        return self.func(*self.args, **self.kwargs)


class TrainLogger(object):
    def __init__(self, *args: str) -> None:
        self.headers = args
        self.template = "%12s " * len(self.headers)
        logging.info(LazyLog(self._log, self.template, args))

    def log(self, *args):
        assert len(args) == len(
            self.headers
        ), "The length of inputs differ from the length of header!"
        logging.info(LazyLog(self._log, self.template, args))

    def _log(self, template, args):
        args = list(args)
        for i in range(len(args)):
            if isinstance(args[i], float):
                args[i] = "%.3e" % args[i]
        return template % tuple(args)


def _log_params(model: torch.nn.Module) -> str:
    name_len = max(len(name) for name, _ in model.named_parameters()) + 1
    shape_len = 20
    n_param_len = 20
    sep_len = name_len + shape_len + n_param_len + 3
    sep = "-" * sep_len
    template = f"%{name_len}s %{shape_len}s %{n_param_len}s\n"
    args: List = ["Name", "Shape", "# Param"]
    for name, param in model.named_parameters():
        args.extend([name, list(param.shape), param.numel()])
    template = "trainable parameters in %s\n%s\n" + template * (len(args) // 3) + "%s"
    return template % (model.__class__.__name__, sep, *args, sep)


def log_params(model: torch.nn.Module) -> LazyLog:
    return LazyLog(_log_params, model)


def log_args(args: Namespace):
    d = vars(args)
    logging.debug(
        "input arguments\n"
        + "----------------------------------------\n"
        + "%s: %s\n" * len(d)
        + "----------------------------------------",
        *sum(d.items(), ()),
    )


def setup_logger(filename: Optional[str], verbose: int) -> None:
    if verbose == 0:
        level = logging.WARNING
    elif verbose == 1:
        level = logging.INFO
    elif verbose == 2:
        level = logging.DEBUG
    else:
        level = logging.NOTSET

    handlers: List[Any] = []
    log_formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    if filename:
        file_handler = logging.FileHandler(filename, mode="w")
        file_handler.setFormatter(log_formatter)
        handlers.append(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_formatter)
    handlers.append(console_handler)

    logging.basicConfig(
        handlers=handlers,
        level=level,
    )

    def log_except_hook(*exc_info):
        text = "".join(traceback.format_exception(*exc_info))
        logging.error("Unhandled exception:\n%s", text)

    sys.excepthook = log_except_hook
