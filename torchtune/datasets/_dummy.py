# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

import datasets
import numpy as np
from torch.utils.data import Dataset
from torchtune.config._utils import _get_instruct_template
from torchtune.data import (
    CROSS_ENTROPY_IGNORE_IDX,
    InstructTemplate,
    Message,
    validate_messages,
)
from torchtune.modules.tokenizers import Tokenizer

class DummyDataset(Dataset):
    """
    Class that represents a deterministic dummy dataset for the purpose of benchmarking.
    Every sample looks like: "instruction", "input", and a really long, to-be-truncated "outputoutput...output"

    The general flow from loading a sample to tokenized prompt is:
    load sample -> apply transform -> format into template -> tokenize

    Masking of the prompt during training is controlled by the `train_on_input` flag, which is
    set to `False` by default.
    - If `train_on_input` is True, the prompt is used during training and
    contributes to the loss.
    - If `train_on_input` is False, the prompt is masked out (tokens replaced with -100)

    Args:
        tokenizer (Tokenizer): Tokenizer used to encode data. Tokenize must implement an `encode` and `decode` method.
        template (InstructTemplate): template used to format the prompt. If the placeholder variable
            names in the template do not match the column/key names in the dataset, use `column_map` to map them.
        transform (Optional[Callable]): transform to apply to the sample before formatting to the template.
            Default is None.
        column_map (Optional[Dict[str, str]]): a mapping from the expected placeholder names in the template
            to the column/key names in the sample. If None, assume these are identical.
        train_on_input (bool): Whether the model is trained on the prompt or not. Default is False.
        max_seq_len (Optional[int]): Maximum number of tokens in the returned input and label token id lists.
            Default is None, disabling truncation. We recommend setting this to the highest you can fit in memory
            and is supported by the model. For example, llama2-7B supports up to 4096 for sequence length.
        **load_dataset_kwargs (Dict[str, Any]): additional keyword arguments to pass to `load_dataset`.
    """

    def __init__(
        self,
        tokenizer: Tokenizer,
        template: InstructTemplate,
        num_samples: int = 128,
        transform: Optional[Callable] = None,
        column_map: Optional[Dict[str, str]] = None,
        train_on_input: bool = False,
        max_seq_len: Optional[int] = 512,
    ) -> None:
        self._tokenizer = tokenizer
        self._data = datasets.Dataset.from_dict({
            'instruction': ["instruction"]*num_samples,
            'input': ["input"]*num_samples,
            'output': ["output"*max_seq_len*2]*num_samples}  # will get truncated
        )
        self.template = template
        self._transform = transform
        self._column_map = column_map
        self.train_on_input = train_on_input
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self._data)

    def __getitem__(self, index: int) -> Tuple[List[int], List[int]]:
        sample = self._data[index]
        return self._prepare_sample(sample)

    def _prepare_sample(self, sample: Mapping[str, Any]) -> Tuple[List[int], List[int]]:
        transformed_sample = self._transform(sample) if self._transform else sample

        prompt = self.template.format(transformed_sample, self._column_map)
        key_output = (
            self._column_map["output"]
            if self._column_map and "output" in self._column_map
            else "output"
        )
        messages = [
            Message(role="user", content=prompt, masked=(not self.train_on_input)),
            Message(role="assistant", content=transformed_sample[key_output]),
        ]

        validate_messages(messages)

        tokens, mask = self._tokenizer.tokenize_messages(
            messages, max_seq_len=self.max_seq_len
        )

        # Wherever mask == True, set to CROSS_ENTROPY_IGNORE_IDX. Otherwise keep as tokens
        labels = list(np.where(mask, CROSS_ENTROPY_IGNORE_IDX, tokens))
        assert len(tokens) == len(labels)

        return tokens, labels

# The same dummy dataset in answer.ai train.py for accurate comparisons
def dummy_dataset(
    *,
    tokenizer: Tokenizer,
    num_samples: int = 128,
    transform: Optional[Callable] = None,
    column_map: Optional[Dict[str, str]] = None,
    train_on_input: bool = False,
    max_seq_len: Optional[int] = 512,
):
    return DummyDataset(
        tokenizer,
        _get_instruct_template("AlpacaInstructTemplate"),
        num_samples=num_samples,
        transform=transform,
        column_map=column_map,
        train_on_input=train_on_input,
        max_seq_len=max_seq_len
    )