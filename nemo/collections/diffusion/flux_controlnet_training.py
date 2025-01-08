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

import argparse

import torch
import torch.nn as nn
from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.optimizer import OptimizerConfig
from pytorch_lightning.loggers import WandbLogger
from transformers import AutoProcessor

from nemo import lightning as nl
from nemo.collections import llm
from nemo.collections.diffusion.data.diffusion_energon_datamodule import DiffusionDataModule
from nemo.collections.diffusion.data.diffusion_mock_datamodule import MockDataModule
from nemo.collections.diffusion.data.diffusion_taskencoder import RawImageDiffusionTaskEncoder
from nemo.collections.diffusion.models.flux_controlnet.model import FluxControlNetConfig, MegatronFluxControlNetModel
from nemo.collections.diffusion.models.flux.model import MegatronFluxModel, FluxModelParams, FluxConfig, T5Config, ClipConfig
from nemo.collections.diffusion.vae.autoencoder import AutoEncoderConfig
from nemo.collections.diffusion.utils.mcore_parallel_utils import Utils
from nemo.lightning.pytorch.optim import WarmupHoldPolicyScheduler
from nemo.lightning.pytorch.optim.megatron import MegatronOptimizerModule
from nemo.utils.exp_manager import TimingCallback, PreemptionCallback
from nemo.lightning.pytorch.callbacks.megatron_comm_overlap import MegatronCommOverlapCallback
from nemo.lightning.pytorch.callbacks.nsys import NsysCallback



from nemo.collections.diffusion.models.flux_controlnet.model import MegatronFluxControlNetModel, FluxControlNetConfig
from megatron.core.distributed import DistributedDataParallelConfig
from nemo.collections.diffusion.data.diffusion_energon_datamodule import DiffusionDataModule
from nemo.collections.diffusion.data.diffusion_taskencoder import RawImageDiffusionTaskEncoder

import nemo_run as run
import os
import pytorch_lightning as pl

@run.cli.factory
@run.autoconvert
def flux_datamodule(dataset_dir) -> pl.LightningDataModule:
    """Flux Datamodule Initialization"""
    data_module = DiffusionDataModule(
        dataset_dir,
        seq_length=4096,
        task_encoder=run.Config(RawImageDiffusionTaskEncoder,),
        micro_batch_size=1,
        global_batch_size=8,
        num_workers=23,
        use_train_split_for_val=True,
    )
    return data_module


@run.cli.factory
@run.autoconvert
def flux_mock_datamodule() -> pl.LightningDataModule:
    """Mock Datamodule Initialization"""
    data_module = MockDataModule(
        image_h=1024,
        image_w=1024,
        micro_batch_size=1,
        global_batch_size=1,
        image_precached=True,
        text_precached=True,
    )
    return data_module




@run.cli.factory(target=llm.train)
def flux_controlnet_training() -> run.Partial:
    """Flux Controlnet Training Config"""
    return run.Partial(
        llm.train,
        model=run.Config(
            MegatronFluxControlNetModel,
            flux_params=run.Config(FluxModelParams),
            flux_controlnet_config=run.Config(FluxControlNetConfig),
        ),
        data=flux_mock_datamodule(),
        trainer=run.Config(
            nl.Trainer,
            devices=1,
            num_nodes=int(os.environ.get('SLURM_NNODES', 1)),
            accelerator="gpu",
            strategy=run.Config(
                nl.MegatronStrategy,
                tensor_model_parallel_size=1,
                pipeline_model_parallel_size=1,
                context_parallel_size=1,
                sequence_parallel=False,
                pipeline_dtype=torch.bfloat16,
                ddp=run.Config(
                    DistributedDataParallelConfig,
                    use_custom_fsdp=True,
                    data_parallel_sharding_strategy='MODEL_AND_OPTIMIZER_STATES',
                    check_for_nan_in_grad=True,
                    grad_reduce_in_fp32=True,
                    overlap_grad_reduce=True,
                    overlap_param_gather=True,
                ),
            ),
            plugins=nl.MegatronMixedPrecision(precision="bf16-mixed"),
            num_sanity_val_steps=0,
            limit_val_batches=1,
            val_check_interval=1000,
            max_epochs=10000,
            log_every_n_steps=1,
            callbacks=[
                run.Config(
                    nl.ModelCheckpoint,
                    monitor='global_step',
                    filename='{global_step}',
                    every_n_train_steps=1000,
                    save_top_k=3,
                    mode='max',
                ),
                run.Config(TimingCallback),
            ],
        ),
        log=nl.NeMoLogger(wandb=(WandbLogger() if "WANDB_API_KEY" in os.environ else None)),
        optim=run.Config(
            nl.MegatronOptimizerModule,
            config=run.Config(
                OptimizerConfig,
                lr=1e-4,
                adam_beta1=0.9,
                adam_beta2=0.999,
                use_distributed_optimizer=True,
                bf16=True,
            ),
        ),
        tokenizer=None,
        resume=run.Config(
            nl.AutoResume,
            resume_if_exists=True,
            resume_ignore_no_checkpoint=True,
            resume_past_end=True,
        ),
        model_transform=None,
    )


@run.cli.factory(target=llm.train)
def convergence_test() -> run.Partial:
    recipe = flux_controlnet_training()
    recipe.model.flux_params.t5_params = run.Config(T5Config, version='/ckpts/text_encoder_2')
    recipe.model.flux_params.clip_params = run.Config(ClipConfig, version='/ckpts/text_encoder')
    recipe.model.flux_params.vae_config = run.Config(AutoEncoderConfig, ckpt='/ckpts/ae.safetensors', ch_mult=[1,2,4,4], attn_resolutions=[])
    recipe.model.flux_params.device = 'cuda'
    recipe.model.flux_params.flux_config = run.Config(FluxConfig, ckpt_path='/ckpts/nemo_flux_transformer.safetensors')
    recipe.trainer.devices=8
    recipe.data = flux_datamodule('/mingyuanm/dataset/fill50k/fill50k_tarfiles/')
    recipe.model.flux_controlnet_config.num_single_layers = 10
    recipe.model.flux_controlnet_config.num_joint_layers = 4
    return recipe

@run.cli.factory(target=llm.train)
def full_model_tp2_dp4_mock() -> run.Partial:
    recipe = flux_controlnet_training()
    recipe.model.flux_params.t5_params = None  # run.Config(T5Config, version='/ckpts/text_encoder_2')
    recipe.model.flux_params.clip_params = None  # run.Config(ClipConfig, version='/ckpts/text_encoder')
    recipe.model.flux_params.vae_config = None  # run.Config(AutoEncoderConfig, ckpt='/ckpts/ae.safetensors', ch_mult=[1,2,4,4], attn_resolutions=[])
    recipe.model.flux_params.device = 'cuda'
    recipe.trainer.strategy.tensor_model_parallel_size=2
    recipe.trainer.devices=8
    recipe.data.global_batch_size = 8
    recipe.trainer.callbacks.append(
        run.Config(
            NsysCallback,
            start_step=10,
            end_step=11,
            gen_shape=True
        )
    )
    recipe.model.flux_controlnet_config.num_single_layers = 10
    recipe.model.flux_controlnet_config.num_joint_layers = 4
    return recipe

@run.cli.factory(target=llm.train)
def full_model_tp2_dp4_mock() -> run.Partial:
    recipe = flux_controlnet_training()
    recipe.model.flux_params.t5_params = None  # run.Config(T5Config, version='/ckpts/text_encoder_2')
    recipe.model.flux_params.clip_params = None  # run.Config(ClipConfig, version='/ckpts/text_encoder')
    recipe.model.flux_params.vae_config = None  # run.Config(AutoEncoderConfig, ckpt='/ckpts/ae.safetensors', ch_mult=[1,2,4,4], attn_resolutions=[])
    recipe.model.flux_params.device = 'cuda'
    recipe.trainer.strategy.tensor_model_parallel_size=2
    recipe.trainer.devices=8
    recipe.data.global_batch_size = 8
    recipe.trainer.callbacks.append(
        run.Config(
            NsysCallback,
            start_step=10,
            end_step=11,
            gen_shape=True
        )
    )
    recipe.model.flux_controlnet_config.num_single_layers = 10
    recipe.model.flux_controlnet_config.num_joint_layers = 4
    return recipe



@run.cli.factory(target=llm.train)
def unit_test() -> run.Partial:
    '''Basic functional test, with mock dataset, text/vae encoders not initialized, ddp strategy, frozen and trainable layers both set to 1'''
    recipe = flux_controlnet_training()
    recipe.model.flux_params.t5_params = None #run.Config(T5Config, version='/ckpts/text_encoder_2')
    recipe.model.flux_params.clip_params = None #run.Config(ClipConfig, version='/ckpts/text_encoder')
    recipe.model.flux_params.vae_config = None #run.Config(AutoEncoderConfig, ckpt='/ckpts/ae.safetensors', ch_mult=[1,2,4,4], attn_resolutions=[])
    recipe.model.flux_params.device = 'cuda'
    recipe.model.flux_params.flux_config=run.Config(
        FluxConfig,
        num_joint_layers=1,
        num_single_layers=1,
    )
    recipe.model.flux_controlnet_config.num_single_layers = 1
    recipe.model.flux_controlnet_config.num_joint_layers = 1
    recipe.data.global_batch_size = 1
    recipe.trainer.strategy.ddp = run.Config(
        DistributedDataParallelConfig,
        check_for_nan_in_grad=True,
        grad_reduce_in_fp32=True,
    )

    return recipe





if __name__ == "__main__":
    OOM_DEBUG = False
    if OOM_DEBUG:
        torch.cuda.memory._record_memory_history(
            True,
            # Keep 100,000 alloc/free events from before the snapshot
            trace_alloc_max_entries=100000,
            # Record stack information for the trace events
            trace_alloc_record_context=True,
        )
    run.cli.main(llm.train, default_factory=unit_test)