from functools import cache
from pathlib import Path
from typing import (
    Any,
    Callable,
    Generic,
    Iterable,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    cast,
)

import torch
from jaxtyping import Bool, Float, Integer
from torch import Tensor
from transformer_lens import HookedTransformer

S = TypeVar("S")


@cache
def get_model(model_path: str | Path, device: str = "cpu") -> HookedTransformer:
    cached_data = torch.load(model_path, map_location=device)
    cached_data["model_config"].device = device
    model = HookedTransformer(cached_data["model_config"])
    model.load_state_dict(cached_data["model"])
    model.to(device, print_details=False)
    return model


def extract_samples(
    samples: (
        int
        | Sequence[int]
        | list[
            Sequence[int]
            | Integer[Tensor, "n_ctx"]  # noqa: F821
            | Integer[Tensor, "batch n_ctx"]  # noqa: F722
        ]
        | Integer[Tensor, "batch n_ctx"]  # noqa: F722
    ),
) -> Iterable[list[int]]:  # noqa: F821
    if isinstance(samples, int):
        yield [samples]
    elif isinstance(samples, Tensor):
        if samples.squeeze().ndim == 0:
            yield [int(samples.item())]
        elif samples.ndim == 1:
            yield list(map(int, samples.tolist()))
        elif samples.ndim == 2:
            for sample in samples:
                yield list(map(int, sample.tolist()))
        elif samples.ndim > 2:
            for sample in samples:
                yield from extract_samples(sample)
    else:
        if all(isinstance(sample, int) for sample in samples):
            yield [cast(int, sample) for sample in samples]
        elif all(isinstance(sample, Tensor) and sample.ndim == 0 for sample in samples):
            yield [int(sample.item()) for sample in cast(list[Tensor], samples)]
        else:
            for sample in samples:
                yield from extract_samples(sample)


class ModelInterventionFunctions(Generic[S]):
    def __init__(
        self,
        model_path: str | Path,
        label_fn: Optional[
            Callable[
                [Integer[Tensor, "batch n_ctx"]],  # noqa: F722
                S,
            ]
        ] = None,
        predict_fn: Optional[
            Callable[
                [Float[Tensor, "batch n_ctx d_vocab_out"]],  # noqa: F722
                S,
            ]
        ] = None,
        acc_fn: Optional[Callable[[S, S], Bool[Tensor, "batch"]]] = (  # noqa: F821
            lambda x, y: x == y
        ),
        *,
        # example_log: Optional[list[Integer[Tensor, "batch n_ctx"]]] = None,
        # example_results_log: Optional[list[Tuple[int, int]]] = None,
        device: str | torch.device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu"
        ),
    ):
        self.label_fn = label_fn
        self.predict_fn = predict_fn
        self.acc_fn = acc_fn
        self.device = device
        self.model = get_model(model_path, device)

    @torch.no_grad()
    def prepare_data(
        self,
        samples: (
            Sequence[
                Sequence[int]
                | Integer[Tensor, "n_ctx"]  # noqa: F821
                | Integer[Tensor, "batch n_ctx"]  # noqa: F722
            ]
            | Integer[Tensor, "batch n_ctx"]  # noqa: F722
        ),
    ) -> Integer[Tensor, "batch n_ctx"]:  # noqa: F722
        return torch.tensor(
            list(map(list, sorted(set(map(tuple, extract_samples(samples)))))),
            dtype=torch.long,
            device=self.device,
        )

    @torch.no_grad()
    def label_data(
        self,
        data: Integer[Tensor, "batch n_ctx"],  # noqa: F722
    ) -> S:
        return self.label_fn(data) if self.label_fn is not None else data

    @torch.no_grad()
    def predict_data(
        self,
        data: Integer[Tensor, "batch n_ctx"],  # noqa: F722
    ) -> S:
        ys = self.model(data)
        return self.predict_fn(ys) if self.predict_fn is not None else ys

    @torch.no_grad()
    def compute_acc_and_success_failure(
        self, data: Integer[Tensor, "batch n_ctx"], lbls: S, ys: S  # noqa: F722
    ) -> Tuple[
        int,
        Tuple[
            Optional[set[Tuple[tuple[int, ...], Any, Any]]],
            Optional[set[Tuple[tuple[int, ...], Any, Any]]],
        ],
    ]:
        acc = self.acc_fn(lbls, ys)
        success = None
        failure = None
        if (
            isinstance(lbls, Tensor)
            and isinstance(ys, Tensor)
            and lbls.shape == ys.shape
            and lbls.shape[0] == data.shape[0]
            and isinstance(acc, Tensor)
            and acc.ndim == 1
            and acc.shape[0] == data.shape[0]
        ):
            failure = set(
                (tuple(map(int, sample)), lbl, y)
                for sample, lbl, y in zip(
                    data[~acc].tolist(), lbls[~acc].tolist(), ys[~acc].tolist()
                )
            )
            success = set(
                (tuple(map(int, sample)), lbl, y)
                for sample, lbl, y in zip(
                    data[acc].tolist(), lbls[acc].tolist(), ys[acc].tolist()
                )
            )
        return int(acc.sum().item()), (success, failure)

    @torch.no_grad()
    def run_model_acc(
        self,
        samples: list[list[int]] | Integer[Tensor, "batch n_ctx"],  # noqa: F722
    ) -> Tuple[
        int,
        Tuple[
            Optional[set[Tuple[tuple[int, ...], Any, Any]]],
            Optional[set[Tuple[tuple[int, ...], Any, Any]]],
        ],
    ]:
        data = self.prepare_data(samples)
        lbls = self.label_data(data)
        ys = self.predict_data(data)
        return self.compute_acc_and_success_failure(data, lbls, ys)
