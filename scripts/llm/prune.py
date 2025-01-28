# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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
"""Pruning example for Llama model.

Example usage to prune width automatically (you can skip parameters that you don't want to prune):
```python
    torchrun --nproc_per_node 8 prune.py \
        --devices 8 \
        --tp_size 1 \
        --pp_size 8 \
        --restore_path <path/to/llama3.1-8b-nemo2> \
        --seq_length 8192 \
        --data_paths 30 path/to/dataset_1_prefix 70 path/to/dataset_2_prefix \
        --prune_ffn_hidden_size 9216 \
        --prune_hidden_size 3072 \
        --prune_num_attention_heads 32 \
        --prune_num_query_groups 8 \
        --save_path llama3.1-8b-width-pruned
```

Example usage to prune depth automatically using cosine-similarity based importance metric:
```python
    torchrun --nproc_per_node 8 prune.py \
        --devices 8 \
        --tp_size 1 \
        --pp_size 8 \
        --restore_path <path/to/llama3.1-8b-nemo2> \
        --seq_length 8192 \
        --data_paths 30 path/to/dataset_1_prefix 70 path/to/dataset_2_prefix \
        --prune_num_layers 16 \
        --save_path llama3.1-8b-depth-pruned
```

NOTE: for above usages, `--tp_size` must be 1 because of the current prune API limitation. If you
do not pass `--data_paths`, the script will use mock data for calibration which will lead to incorrect
pruning results but helps in testing the pruning pipeline.

Example usage to prune depth by dropping specific model layers (1-indexed):
```python
    torchrun --nproc_per_node 8 prune.py \
        --devices 8 \
        --tp_size 8 \
        --pp_size 1 \
        --restore_path <path/to/llama3.1-8b-nemo2> \
        --drop_layers 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 \
        --save_path llama3.1-8b-dropped-layers
```
"""

import argparse
import os
import shutil

import modelopt.torch.prune as mtp
import torch
from megatron.core import dist_checkpointing
from megatron.core.transformer.module import Float16Module as MCoreFloat16Module

from nemo import lightning as nl
from nemo.collections import llm
from nemo.collections.common.tokenizers.huggingface.auto_tokenizer import AutoTokenizer
from nemo.collections.llm.inference.base import _setup_trainer_and_restore_model
from nemo.collections.nlp.models.language_modeling.megatron.gpt_layer_modelopt_spec import get_gpt_layer_modelopt_spec
from nemo.collections.nlp.modules.common.megatron.module import Float16Module
from nemo.lightning.ckpt_utils import ckpt_to_context_subdir
from nemo.lightning.io.pl import TrainerContext, ckpt_to_weights_subdir
from nemo.utils import logging
from nemo.utils.model_utils import unwrap_model

os.environ["TOKENIZERS_PARALLELISM"] = "true"

SUPPORTED_PRUNING_HPARAMS = {
    # Width pruning
    "ffn_hidden_size",
    "hidden_size",
    "num_attention_heads",
    "num_query_groups",
    # Depth pruning
    "num_layers",
}


def get_data_module(args):
    """Get data module for running validation loop on.

    Also overwrites val dataloader to return train dataloader for importance estimation.
    """
    assert args.num_train_samples % args.gbs == 0, "num_train_samples must be divisible by gbs"
    assert args.seq_length, "Sequence length must be provided for pruning"

    data_module_kwargs = {}
    if args.data_paths:
        logging.info(f"Loading pre-training data from {args.data_paths}")
        data_module_cls = llm.PreTrainingDataModule
        data_module_kwargs["paths"] = args.data_paths
        data_module_kwargs["split"] = args.split
        data_module_kwargs["index_mapping_dir"] = args.index_mapping_dir
    else:
        logging.warning("Using mock data since --data_paths is not provided.")
        data_module_cls = llm.MockDataModule
    data_module = data_module_cls(
        seq_length=args.seq_length,
        micro_batch_size=args.mbs,
        global_batch_size=args.gbs,
        num_train_samples=args.num_train_samples,
        **data_module_kwargs,
    )

    # Overwrite val dataloader to use train dataloader with llm.validate
    data_module.val_dataloader = data_module.train_dataloader

    return data_module


def load_model_with_modelopt_spec(restore_path: str, trainer: nl.Trainer) -> llm.GPTModel:
    """Load model from nemo checkpoint with modelopt spec."""
    logging.info(f"Loading model from {restore_path}...")
    model = nl.io.load_context(path=ckpt_to_context_subdir(restore_path), subpath="model")
    model.config.transformer_layer_spec = get_gpt_layer_modelopt_spec()
    del model.optim
    _setup_trainer_and_restore_model(restore_path, trainer, model)
    logging.info(f"Loaded model: {model}\n")
    return model


def save_pruned_model(model: llm.GPTModel, trainer: nl.Trainer, save_path: str):
    """Save pruned model nemo checkpoint."""
    logging.info(f"Saving pruned model to {save_path}...")
    if hasattr(model.tokenizer, "save_pretrained"):
        tokenizer_tmp_dir = "/tmp/nemo_tokenizer"
        model.tokenizer.save_pretrained(tokenizer_tmp_dir)
        model.tokenizer = AutoTokenizer(tokenizer_tmp_dir)
        shutil.rmtree(tokenizer_tmp_dir)
    if hasattr(trainer.model, "__io__") and hasattr(trainer.model.tokenizer, "__io__"):
        trainer.model.__io__.tokenizer = trainer.model.tokenizer.__io__
        # Make sure pruned hparams are saved
        for k in SUPPORTED_PRUNING_HPARAMS | {"kv_channels"}:
            setattr(trainer.model.__io__.config, k, getattr(model.config, k))

    weight_path = ckpt_to_weights_subdir(save_path, is_saving=True)
    weight_path.mkdir(parents=True, exist_ok=True)
    dist_checkpointing.save(trainer.strategy.megatron_parallel.sharded_state_dict(), weight_path)

    TrainerContext.from_trainer(trainer).io_dump(ckpt_to_context_subdir(save_path), yaml_attrs=["model"])

    logging.info(f"Pruned model saved to {save_path}\n")


def main(args):
    """Main function for pruning Llama model."""
    # Training strategy setup
    strategy = nl.MegatronStrategy(
        tensor_model_parallel_size=args.tp_size,
        pipeline_model_parallel_size=args.pp_size,
        pipeline_dtype=torch.bfloat16,
        sequence_parallel=False,
        ckpt_load_optimizer=False,
        ckpt_parallel_save_optim=False,
        setup_optimizers=False,
        ddp="pytorch",
    )

    # Trainer setup
    trainer = nl.Trainer(
        num_nodes=args.num_nodes,
        devices=args.devices,
        accelerator="gpu",
        strategy=strategy,
        plugins=nl.MegatronMixedPrecision(precision="bf16-mixed", params_dtype=torch.bfloat16, autocast_enabled=True),
        max_steps=args.num_train_samples // args.gbs,
        limit_val_batches=args.num_train_samples // args.mbs,
    )

    model = load_model_with_modelopt_spec(args.restore_path, trainer)
    export_config = {
        k: getattr(args, f"prune_{k}") for k in SUPPORTED_PRUNING_HPARAMS if getattr(args, f"prune_{k}") is not None
    }
    if args.drop_layers:
        assert not export_config, f"Cannot specify `--drop_layers` with other prune constraints. Recieved: {args}"

        unwrapped_model = unwrap_model(model.module, (Float16Module, MCoreFloat16Module))
        mtp.plugins.megatron.drop_mcore_gpt_layers(unwrapped_model, layers_to_drop=args.drop_layers)
    else:
        assert args.tp_size == 1, "Pruning currently only supports --tp_size=1"

        data_module = get_data_module(args)

        def forward_loop(model):
            llm.validate(model, data_module, trainer)

        logging.info("Pruning model...")
        mtp.prune(
            model,
            mode="mcore_gpt_minitron",
            constraints={"export_config": export_config},
            dummy_input=None,  # Not used
            config={"forward_loop": forward_loop},
        )

    save_pruned_model(model, trainer, args.save_path)

    logging.info("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Llama Pruning Script")
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--num_nodes", type=int, default=1)
    parser.add_argument(
        "--tp_size",
        type=int,
        default=1,
        help="Tensor parallel size. Can only be 1 if pruning and not dropping layers",
    )
    parser.add_argument("--pp_size", type=int, default=1, help="Pipeline parallel size")
    parser.add_argument(
        "--seq_length",
        type=int,
        help="Sequence length. Only required if pruning and not dropping layers.",
    )
    parser.add_argument("--restore_path", type=str, required=True, help="Path to restore model checkpoint from")
    parser.add_argument("--save_path", type=str, required=True, help="Path to save pruned model checkpoint to")
    # Calibration data parameters
    parser.add_argument(
        "--data_paths",
        type=str,
        metavar="N",
        nargs="*",
        help="""Paths of the data distributions. Only required if pruning and not dropping layers
            The given paths will be used to generate the train, validation and test datasets.
            The format can be either (1) a list of paths, e.g.
                ["path/to/dataset_1_prefix", "path/to/dataset_2_prefix"],
            or (2) a flattened, zipped list of weights and paths, e.g.
                ["30", "path/to/dataset_1_prefix", "70", "path/to/dataset_2_prefix"]
            Check nemo.collections.llm.PreTrainingDataModule for more info and alternative formats.
        """,
    )
    parser.add_argument(
        "--split",
        type=str,
        default="99,1,0",
        help="A string of 3 comma-separated integers denoting how much of the distribution to "
        "allocate to train, validation, and test sets, respectively",
    )
    parser.add_argument("--index_mapping_dir", type=str, help="Path to a directory to write index mapping files")
    parser.add_argument("--mbs", type=int, default=1, help="Micro batch size")
    parser.add_argument("--gbs", type=int, default=32, help="Global batch size")
    parser.add_argument(
        "--num_train_samples",
        type=int,
        default=1024,
        help="Number of training samples for importance estimation",
    )
    # Pruning parameters
    parser.add_argument("--prune_ffn_hidden_size", type=int, help="Prune MLP FFN hidden size to this value")
    parser.add_argument("--prune_hidden_size", type=int, help="Prune hidden size (embedding dim) to this value")
    parser.add_argument(
        "--prune_num_attention_heads",
        type=int,
        help="Prune number of attention heads to this value. Must be supplied with " "--prune_num_query_groups",
    )
    parser.add_argument(
        "--prune_num_query_groups",
        type=int,
        help="Prune number of query groups to this value. Must be supplied with " "--prune_num_attention_heads",
    )
    parser.add_argument(
        "--prune_num_layers",
        type=int,
        help="Prune number of transformer layers to this value based on "
        "Block Influence metric (cosine similarity) as per https://arxiv.org/abs/2403.03853",
    )
    parser.add_argument(
        "--drop_layers",
        type=int,
        metavar="N",
        nargs="*",
        help="Drop specific model layers (1-indexed). Cannot be used with rest of the pruning options",
    )

    args = parser.parse_args()
    main(args)
