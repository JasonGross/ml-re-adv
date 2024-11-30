# %%
import logging
from typing import Any, Optional, Tuple

import inspect_ai.scorer
import inspect_ai.util
import pandas as pd
from inspect_ai.scorer import Score, Scorer, Target, scorer
from inspect_ai.solver import TaskState
from inspect_ai.util import Store
from jaxtyping import Integer
from torch import Tensor

from ml_re_adv.utils import metrics

logger = logging.getLogger(__name__)


def init_examples_store(name: str, store: Optional[Store] = None) -> None:
    if store is None:
        store = inspect_ai.util.store()
    if f"{name}_examples" not in store:
        store.set(f"{name}_examples", [])


def get_examples(
    name: str,
    store: Optional[Store] = None,
) -> list[dict[str, Any]]:
    if store is None:
        store = inspect_ai.util.store()
    init_examples_store(name, store)
    return store.get(f"{name}_examples")


def save_examples(
    name: str,
    samples: list[list[int]] | Integer[Tensor, "batch n_ctx"],  # noqa: F722
    acc: int,
    code: str,
    function_name: str,
    store: Optional[Store] = None,
    *,
    success: Optional[set[Tuple[tuple[int, ...], Any, Any]]] = None,
    failure: Optional[set[Tuple[tuple[int, ...], Any, Any]]] = None,
    max_examples: Optional[int] = 100,
    sample_sorting_key=lambda e: (tuple(sorted(e, reverse=True)), tuple(e)),
    example_sorting_key=lambda e: (
        tuple(
            v if not isinstance(v, (tuple, list)) else tuple(sorted(v, reverse=True))
            for v in e
        ),
        tuple(e),
    ),
    reverse_sort_examples: bool = False,
    success_name: str = "succeeding",
    failure_name: str = "adversarial",
) -> None:
    def sort_examples(examples):
        try:
            examples = sorted(
                examples, key=example_sorting_key, reverse=reverse_sort_examples
            )
        except TypeError as e:
            logger.error(f"Error sorting examples ({examples}): {e}")
        try:
            return list(map(list, examples))
        except TypeError as e:
            logger.error(f"Error converting examples to list ({examples}): {e}")
        return examples

    if store is None:
        store = inspect_ai.util.store()
    init_examples_store(name, store)

    adversarial_examples = sort_examples(failure) if failure else []
    succeeding_examples = sort_examples(success) if success else []
    adversarial_examples = adversarial_examples.__getitem__(slice(max_examples))
    succeeding_examples = succeeding_examples.__getitem__(slice(max_examples))
    get_examples(name, store).append(
        {
            "total": len(samples),
            success_name: acc,
            failure_name: len(samples) - acc,
            "code": code,
            "function_name": function_name,
        }
        | (
            {f"{success_name}_examples": succeeding_examples}
            if succeeding_examples
            else {}
        )
        | (
            {f"{failure_name}_examples": adversarial_examples}
            if adversarial_examples
            else {}
        )
    )


# class AdversarialExample(TypedDict):
#     total: int
#     succeeding: int
#     adversarial: int
#     code: str
#     function_name: str
#     succeeding_examples: NotRequired[list[Tuple[list[int], Any, Any]]]
#     adversarial_examples: NotRequired[list[Tuple[list[int], Any, Any]]]


@scorer(
    metrics=[
        inspect_ai.scorer.mean(),
        inspect_ai.scorer.std(),
        inspect_ai.scorer.stderr(),
        metrics.median(),
        metrics.min(),
        metrics.max(),
    ]
)
def examples_scorer(name: str, key: str, *, percentage: bool = False) -> Scorer:
    async def run(
        state: TaskState,
        target: Target,
    ) -> Score:
        """
        A scorer to score the results of the examples tool.
        """
        examples = get_examples(name, state.store)
        if not examples:
            return Score(value=0, explanation="No examples submitted by tool")
        best = max(examples, key=lambda example: example[key])
        explanation = (
            pd.DataFrame(
                best.get(f"{key}_examples"),
                columns=["sample", "label", "prediction"],
            ).to_markdown()
            if best.get(f"{key}_examples")
            else None
        )
        return Score(
            value=best[key] if not percentage else best[key] / best["total"],
            answer=f"{best['code']}\n\n{best['function_name']}",
            explanation=explanation,
            metadata={
                k: v
                for k, v in best.items()
                if k
                not in [
                    "code",
                    "function_name",
                ]
                and not k.endswith("_examples")
                and (percentage or k != key)
            },
        )

    return run
