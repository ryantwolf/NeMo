from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Dict, List, Literal, Optional, Tuple, Union

import torch
from megatron.core.extensions.transformer_engine import (
    TEColumnParallelLinear,
    TEDotProductAttention,
    TELayerNormColumnParallelLinear,
    TENorm,
    TERowParallelLinear,
)
from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.inference_params import InferenceParams
from megatron.core.models.multimodal.llava_model import LLaVAModel as MCoreLLaVAModel
from megatron.core.transformer.attention import CrossAttention, CrossAttentionSubmodules
from megatron.core.transformer.enums import AttnMaskType as MCoreAttnMaskType
from megatron.core.transformer.enums import ModelType
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import TransformerLayerSubmodules

from nemo.collections.multimodal.mimo.model.gpt import MimoGPTModel
from nemo.collections.multimodal.mimo.model.projection import Baseconfig, TempPoolingHead, TransformersProjector


class CustomMimoModel(MCoreLLaVAModel):
    def __init__(
        self,
        model_config: TransformerConfig,
        language_transformer_config: TransformerConfig,
        language_transformer_layer_spec: ModuleSpec,
        language_vocab_size: int,
        language_max_sequence_length: int,
        vision_transformer_config: TransformerConfig,
        vision_transformer_layer_spec: ModuleSpec,
        drop_vision_class_token: bool,
        vision_projection_config: TransformerConfig,
        vision_projection_layer_spec: ModuleSpec,
        vision_output_projection_config: TransformerConfig,
        vision_output_projection_spec: ModuleSpec,
        vision_projection_type: str = "mlp",
        allow_missing_vision_projection_checkpoint: bool = False,
        parallel_output: bool = True,
        pre_process: bool = True,
        post_process: bool = True,
        add_encoder: bool = True,
        add_decoder: bool = True,
        img_h: int = 336,
        img_w: int = 336,
        patch_dim: int = 14,
        language_rotary_base: int = 10000,
        language_rope_scaling: bool = False,
    ) -> None:
        # Temporarily disable add_decoder to prevent MCoreGPTModel initialization
        self.add_decoder = False
        super().__init__(
            language_transformer_config=language_transformer_config,
            language_transformer_layer_spec=language_transformer_layer_spec,
            language_vocab_size=language_vocab_size,
            language_max_sequence_length=language_max_sequence_length,
            vision_transformer_config=vision_transformer_config,
            vision_transformer_layer_spec=vision_transformer_layer_spec,
            drop_vision_class_token=drop_vision_class_token,
            vision_projection_config=vision_projection_config,
            vision_projection_layer_spec=vision_projection_layer_spec,
            vision_projection_type=vision_projection_type,
            allow_missing_vision_projection_checkpoint=allow_missing_vision_projection_checkpoint,
            parallel_output=parallel_output,
            language_position_embedding_type=language_transformer_config.position_embedding_type,
            language_rotary_percent=language_transformer_config.rotary_percent,
            pre_process=pre_process,
            post_process=post_process,
            add_encoder=add_encoder,
            add_decoder=False,  # Ensure GPTModel isn't initialized
            img_h=img_h,
            img_w=img_w,
            patch_dim=patch_dim,
            language_rotary_base=language_rotary_base,
            language_rope_scaling=language_rope_scaling,
        )
        self.model_config = model_config
        # Now re-enable add_decoder after parent constructor is done
        self.add_decoder = True
        self.model_type = ModelType.encoder_or_decoder

        # Initialize MimoGPTModel
        self.language_model = MimoGPTModel(
            config=language_transformer_config,
            transformer_layer_spec=language_transformer_layer_spec,
            vocab_size=language_vocab_size,
            max_sequence_length=language_max_sequence_length,
            parallel_output=parallel_output,
            position_embedding_type=language_transformer_config.position_embedding_type,
            rotary_percent=language_transformer_config.rotary_percent,
            pre_process=pre_process,
            post_process=post_process,
            rotary_base=language_rotary_base,
            rope_scaling=language_rope_scaling,
        )

        self.share_embeddings_and_output_weights = self.language_model.share_embeddings_and_output_weights
        self._language_max_sequence_length = language_max_sequence_length
        self._language_is_pipeline_parallel = language_transformer_config.pipeline_model_parallel_size > 1
        from diffusers import EulerDiscreteScheduler, StableDiffusionPipeline

        self.image_decoder_name = "stabilityai/stable-diffusion-2"
        self.scheduler = EulerDiscreteScheduler.from_pretrained(self.image_decoder_name, subfolder="scheduler")
        self.image_decoder = StableDiffusionPipeline.from_pretrained(self.image_decoder_name, scheduler=self.scheduler)

        self.image_decoder.vae.requires_grad_(False)
        self.image_decoder.unet.requires_grad_(False)
        self.image_decoder.text_encoder.requires_grad_(False)

        # output projection Megatron Module

        # self.vision_output_projection_module = MCoreMultimodalProjector(
        #         vision_output_projection_config,
        #         vision_output_projection_spec,
        #         projector_type="mlp" ,
        #         input_size=vision_output_projection_config.input_size,
        #     )
        # self.vision_output_projection_module = TransformersProjector(
        #     in_features=self.config.hidden_size, out_features=1024, num_query_token=77
        # )  # Yash : TODO Fix hard coding
        self.vision_output_projection_module = TempPoolingHead(
            config=Baseconfig(),
            submodules=TransformerLayerSubmodules(
                cross_attention=ModuleSpec(
                    module=CrossAttention,
                    params={"attn_mask_type": MCoreAttnMaskType.no_mask},
                    submodules=CrossAttentionSubmodules(
                        linear_q=TEColumnParallelLinear,
                        linear_kv=TEColumnParallelLinear,
                        core_attention=TEDotProductAttention,
                        linear_proj=TERowParallelLinear,
                    ),
                ),
                cross_attn_bda=get_bias_dropout_add,
                mlp=ModuleSpec(
                    module=MLP,
                    submodules=MLPSubmodules(
                        linear_fc1=TELayerNormColumnParallelLinear,
                        linear_fc2=TERowParallelLinear,
                    ),
                ),
                mlp_bda=get_bias_dropout_add,
            ),
            num_query_token=77,
        )

    def get_image_caption_embeddings(self, text_input):
        with torch.no_grad():
            text_inputs = self.image_decoder.tokenizer(
                text_input, padding="max_length", truncation=True, return_tensors="pt", add_special_tokens=True
            )
            # if torch.distributed.get_rank() == 0:  # or other ranks
            #     breakpoint()
            # torch.distributed.barrier()
            text_inputs = text_inputs.to(self.image_decoder.device)
            image_caption_embeddings = self.image_decoder.text_encoder(**text_inputs)[0]  # b,77,1024

            return image_caption_embeddings

    def forward(
        self,
        images: torch.Tensor,
        output_images: torch.Tensor,
        input_ids: torch.Tensor,
        input_text: str,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        loss_mask: Optional[torch.Tensor] = None,
        inference_params: Optional[InferenceParams] = None,
        num_image_tiles: Optional[List[int]] = None,
        image_token_index: Optional[int] = -200,
        runtime_gather_output: Optional[bool] = None,
    ) -> torch.Tensor:
        """Forward function of the LLaVA model.

        Args:
            images (torch.Tensor): input images of shape [num_tiles, img_h, img_w].
                num_tiles means the number of image tiles in this batch.
                num_tiles = 0 if the batch doesn't contain images.
            input_ids (torch.Tensor): input text ids [batch, text_seq_len].
            position_ids (torch.Tensor): input text position ids [batch, text_seq_len].
            attention_mask (torch.Tensor): Language model attention mask
                [batch, 1, combined_seq_len, combined_seq_len].
            labels (torch.Tensor): Optional target text labels [batch, combined_seq_len].
            loss_mask (torch.Tensor): Text loss mask [batch, text_seq_len].
            inference_params (InferenceParams): Inference-time parameters including KV cache.
            num_image_tiles (list of int): Number of tiles per image. Default 1 tile per image.
            image_token_index (int): ID for input images.
            runtime_gather_output (bool): Gather output at runtime. Default None means
                `parallel_output` arg in the constructor will be used.

        Returns:
            output (torch.Tensor): Loss of shape [b, s] if labels are provided,
                otherwise logits of shape [b, s, vocab_size].
            loss_mask (torch.Tensor): Loss mask expanded to combined sequence length. Shape [b, s].
        """
        use_inference_kv_cache = (
            inference_params is not None and "image_tokens_count" in inference_params.key_value_memory_dict
        )
        has_images = images.shape[0] > 0

        # If running inference, we can skip image token computation
        # if they were computed already earlier for this sample.
        if use_inference_kv_cache:
            image_embeddings = None
        elif self.add_encoder and not has_images:
            # If no images provided, use an empty image embeddings tensor.
            image_embeddings = torch.tensor([], dtype=images.dtype, device=images.device).reshape(0, 0, 0)
        elif self.add_encoder and has_images:
            image_embeddings = self.vision_model(images)  # [num_tiles, img_seq_len, h_vision]
            if self._drop_vision_class_token:
                image_embeddings = image_embeddings[:, self.vision_model.class_token_len :, :]
            # contiguous() required as `permute` can sparsify the tensor and this breaks pipelining
            image_embeddings = image_embeddings.permute(1, 0, 2).contiguous()  # [img_seq_len, num_tiles, h_vision]

            # map vision model output size to language model input size.
            image_embeddings = self.vision_projection(image_embeddings)  # [img_seq_len, num_tiles, h_language]

            # TODO: Support batched inference.
            # In inference, the language model KV cache will be updated for image token positions.
            # Store the image tokens sequence length to be used as an offset to the KV cache later.
            if inference_params is not None:
                inference_params.key_value_memory_dict["image_tokens_count"] = (
                    image_embeddings.shape[0] * image_embeddings.shape[1]
                )
        else:
            image_embeddings = self.encoder_hidden_state

        if not self.add_decoder:
            return image_embeddings, loss_mask

        language_embeddings = None
        if self.pre_process:
            input_ids_text = input_ids.clone()
            input_ids_text[input_ids_text == image_token_index] = 0
            # Note: This adds absolute position embedding but not RoPE.
            # Each image is counted as one position.
            # RoPE is added in language_model forward. Each image embedding is one position.
            language_embeddings = self.language_model.embedding(
                input_ids=input_ids_text, position_ids=position_ids
            )  # [text_seq_len, b, h_language]
            language_embeddings = language_embeddings.transpose(1, 0).contiguous()  # [b, text_seq_len, h_language]

        # Assume 1 tile per image if the number of tiles is not provided.
        if num_image_tiles is None:
            num_image_tiles = torch.ones(images.shape[0], dtype=torch.int, device=input_ids.device)

        # if torch.distributed.get_rank() == 0:  # or other ranks
        #     breakpoint()
        # torch.distributed.barrier()

        # Preprocess input, labels and loss mask.
        combined_embeddings, new_labels, new_loss_mask, attention_mask = self._preprocess_data(
            image_embeddings,
            language_embeddings,
            input_ids,
            loss_mask,
            labels,
            use_inference_kv_cache,
            image_token_index,
            num_image_tiles,
            attention_mask,
        )  # [combined_seq_len, b, h_language], [b, combined_seq_len], [b, combined_seq_len]
        # TODO: Yash return this hidden state for computing loss

        output, hidden_states = self.language_model(
            input_ids=None,
            position_ids=None,
            attention_mask=attention_mask,
            decoder_input=combined_embeddings,
            labels=new_labels,
            inference_params=inference_params,
            runtime_gather_output=runtime_gather_output,
        )
        # if labels is None output is logits (b,s,vocab_size) or its loss (b,s)

        # send hidden_state for special tokens to output_projection module.
        device = output_images.device
        image_decoder = self.image_decoder.to(device)
        image_caption_embeddings = self.get_image_caption_embeddings(input_text)  # (bs, 77, 1024)
        # if torch.distributed.get_rank() == 0:
        #     breakpoint()
        # torch.distributed.barrier()
        if new_labels is not None:
            special_token_mask = torch.zeros_like(new_labels, dtype=torch.bool)
            for idx in self.model_config.image_special_token_indices:
                special_token_mask |= new_labels == idx

            nonzero_indices = torch.nonzero(special_token_mask, as_tuple=False)
            special_token_positions = nonzero_indices[:, 1]
            special_token_indices = new_labels[special_token_mask]

            special_token_positions = special_token_positions.view(
                new_labels.size(0), -1
            )  # batch_size, no_special_tokens
            special_token_indices = special_token_indices.view(new_labels.size(0), -1)

            # if torch.distributed.get_rank() == 0:  # or other ranks
            #     breakpoint()
            # torch.distributed.barrier()
            special_token_mask = special_token_mask.transpose(0, 1).unsqueeze(-1)
            special_token_mask = special_token_mask.expand_as(hidden_states)
            selected_hidden_states = hidden_states[special_token_mask].view(
                hidden_states.size(1), -1, hidden_states.size(-1)
            )

            special_token_embeddings = self.language_model.embedding(
                input_ids=special_token_indices, position_ids=special_token_positions
            )
            special_token_embeddings = special_token_embeddings.transpose(0, 1)  # change to b,s,h
            if torch.distributed.get_rank() == 0:
                breakpoint()
            torch.distributed.barrier()
            output_projection_embeddings = self.vision_output_projection_module(
                selected_hidden_states + special_token_embeddings
            )  # (bs, no_special_tokens, 1024)
            if torch.distributed.get_rank() == 0:
                breakpoint()
            torch.distributed.barrier()
            # Image caption embeddings
            image_caption_embeddings = image_caption_embeddings.to(
                output_projection_embeddings.device, dtype=output_projection_embeddings.dtype
            )

        if labels is None or loss_mask is None:
            # return output
            return {
                'output': output,
                # 'output_projection_embeddings': output_projection_embeddings,
                # 'image_caption_embeddings': image_caption_embeddings,
                'hidden_states': hidden_states,
            }

        # for calcualating denoising loss
        # with torch.no_grad():

        # if torch.distributed.get_rank() == 0:  # or other ranks
        #     breakpoint()
        # torch.distributed.barrier()

        latents = image_decoder.to(device).vae.encode(output_images).latent_dist.sample()
        latents = latents * image_decoder.vae.config.scaling_factor

        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        # Sample a random timestep for each sample in the batch
        timesteps = torch.randint(0, image_decoder.scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
        timesteps = timesteps.long()

        # add noise to latents using timesteps
        noisy_latents = image_decoder.scheduler.add_noise(latents, noise, timesteps)
        # make added noise target
        target = noise
        # predict the added noise
        model_pred = image_decoder.unet(noisy_latents, timesteps, output_projection_embeddings).sample
        # model_pred = image_decoder.unet(
        #     noisy_latents, timesteps, image_caption_embeddings.to(dtype=noisy_latents.dtype)
        # # ).sample
        # snr = compute_snr(timesteps, image_decoder.scheduler)
        # mse_loss_weights = torch.stack([snr, 5 * torch.ones_like(timesteps)], dim=1).min(dim=1)[0] / snr

        # if torch.distributed.get_rank() == 0:  # or other ranks
        #     breakpoint()
        # torch.distributed.barrier()

        return {
            'output': output,
            'new_loss_mask': new_loss_mask,
            'output_projection_embeddings': output_projection_embeddings,
            'image_caption_embeddings': image_caption_embeddings,
            'hidden_states': hidden_states,
            # 'denoise_mse_loss_weights': mse_loss_weights,
            'denoise_model_pred': model_pred,
            'denoise_target': target,
        }
        # return (output,output_projection_embeddings, image_caption_embeddings), new_loss_mask
