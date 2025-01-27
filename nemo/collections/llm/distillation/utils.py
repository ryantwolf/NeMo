# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from contextlib import contextmanager
from types import MethodType
from typing import TYPE_CHECKING, Any, Dict

import modelopt.torch.distill as mtd
import modelopt.torch.opt as mto
import torch
from megatron.core import parallel_state
from torch import Tensor

from nemo import lightning as nl
from nemo.collections import llm
from nemo.utils import logging

from .loss import LogitsKLLoss

if TYPE_CHECKING:
    from megatron.core.dist_checkpointing.mapping import ShardedStateDict
    from megatron.core.models.gpt.gpt_model import GPTModel as MCoreGPTModel
    from megatron.core.transformer.transformer_config import TransformerConfig

    from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec


def load_distillation_config(cfg: "TransformerConfig") -> Dict[str, Any]:
    """Create a default distillation config for MCore GPT Models."""
    logit_pair = ("output_layer", "output_layer")  # logit module names for MCoreGPTModel
    distill_cfg = {
        "criterion": {},
        "loss_balancer": _DummyLossBalancer(),  # HACK: to appease ModelOpt until validation relaxed
        "skip_lm_loss": True,
    }
    if cfg.pipeline_model_parallel_size == 1 or parallel_state.is_pipeline_last_stage():
        distill_cfg["criterion"][logit_pair] = LogitsKLLoss(cfg)

    return distill_cfg


class _DummyLossBalancer(mtd.DistillationLossBalancer):
    def forward(self, loss_dict):
        # pylint: disable=C0116
        return next(iter(loss_dict.values()))


def teacher_provider(
    config: llm.GPTConfig, ckpt_path: str, tokenizer: "TokenizerSpec", trainer: nl.Trainer
) -> "MCoreGPTModel":
    """Teacher model factory (must be a non-local function to pickle)."""
    logging.info("Distillation: Loading teacher weights...")

    # TODO(aanoosheh): Replace spec with modelopt one
    model = config.configure_model(tokenizer)

    sharded_state_dict = {"state_dict": model.sharded_state_dict(prefix="module.")}
    checkpoint = trainer.strategy.checkpoint_io.load_checkpoint(ckpt_path, sharded_state_dict)
    state_dict = {k.replace("module.", ""): v for k, v in checkpoint["state_dict"].items()}
    model.load_state_dict(state_dict)

    torch.cuda.empty_cache()
    logging.info("Distillation: ...teacher weights loaded.")
    return model


class LoopingCachedDataIterator:
    """Iterator which takes in a sequence and cycles through it when exhausted."""

    def __init__(self, data):
        self.data = data
        self.it = iter(self.data)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            return next(self.it)
        except StopIteration:
            self.it = iter(self.data)
            return next(self.it)


def adjust_distillation_model_for_mcore(
    model: mtd.DistillationModel, model_cfg: "TransformerConfig", distill_cfg: Dict[str, Any]
):
    """Extra modifcations to ``mtd.DistillationModel`` requried for Megatron-Core."""

    # HACK: Get rid of ModelOpt Distillation state
    # NOTE: If re-placed, above losses need modifcation as `TransformerConfig` has non-pickleable elements.
    mto.ModeloptStateManager(model)._state.pop()

    # HACK: Hide teacher during `sharded_state_dict` method.
    def _sharded_state_dict(self, *args, **kwargs) -> "ShardedStateDict":
        with self.hide_teacher_model():
            return self._sharded_state_dict(*args, **kwargs)

    model._sharded_state_dict = model.sharded_state_dict
    model.sharded_state_dict = MethodType(_sharded_state_dict, model)

    # HACK: Skip `lm_loss` bypassing it when training if not needed for backprop.
    def _compute_language_model_loss(self, labels, logits) -> Tensor:
        if self.training:
            return torch.zeros_like(labels, dtype=logits.dtype)
        return self._compute_language_model_loss(labels, logits)

    if distill_cfg["skip_lm_loss"]:
        model._compute_language_model_loss = model.compute_language_model_loss
        model.compute_language_model_loss = MethodType(_compute_language_model_loss, model)

    # HACK: Skip `lm_loss` always for teacher.
    def _compute_language_model_loss(self, labels, logits) -> Tensor:
        return torch.zeros_like(labels, dtype=logits.dtype)

    model.teacher_model.compute_language_model_loss = MethodType(_compute_language_model_loss, model.teacher_model)

    if model_cfg.pipeline_model_parallel_size > 1:

        def _set_input_tensor(self, input_tensor: Tensor):
            obj = self.teacher_model if self._only_teacher_fwd else self
            return type(self).set_input_tensor(obj, input_tensor)

        # HACK: Pipeline-parallel Distillation requires a way to cache input batches for subsequent
        # forward calls, as well as a way to pass through output tensors to teacher model.
        model.set_input_tensor = MethodType(_set_input_tensor, model)

        @contextmanager
        def _swap_teacher_config(self, model_wrapper):
            try:
                if hasattr(model_wrapper, "config"):
                    model_wrapper._config = model_wrapper.config
                model_wrapper.config = self.teacher_model.config
                yield
            finally:
                del model_wrapper.config
                if hasattr(model_wrapper, "_config"):
                    model_wrapper.config = model_wrapper._config
                    del model_wrapper._config

        # HACK: Pipeline-parallel forward function relies on the config in the model to know what
        # hidden size of tensor to communicate to next stage.
        model.swap_teacher_config = MethodType(_swap_teacher_config, model)


def load_cicd_models(student_path: str):
    # pylint: disable=C0116
    import os.path

    from nemo.collections.nlp.modules.common.tokenizer_utils import get_nmt_tokenizer
    from tests.collections.llm.common import Llama3ConfigCI

    tokenizer = get_nmt_tokenizer(tokenizer_model=os.path.join(student_path, "dummy_tokenizer.model"))
    student_model = llm.LlamaModel(Llama3ConfigCI(), tokenizer=tokenizer)
    teacher_model = llm.LlamaModel(Llama3ConfigCI(), tokenizer=tokenizer)

    return student_model, teacher_model, tokenizer
