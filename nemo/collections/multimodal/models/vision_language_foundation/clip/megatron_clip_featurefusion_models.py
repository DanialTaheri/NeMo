import sys
from nemo.collections.multimodal.data.clip.clip_dataset import get_preprocess_fns
from nemo.collections.multimodal.losses.clip_loss import InbatchContrastiveLoss
from nemo.collections.multimodal.models.vision_language_foundation.clip.megatron_clip_models import (
    MegatronCLIPModel,
)
from nemo.core.classes.common import PretrainedModelInfo
from transformers.models.t5.modeling_t5 import T5Stack
from transformers.models.t5 import T5Config

import einops
import torch
import torch.distributed.nn
from omegaconf.dictconfig import DictConfig
from pytorch_lightning.accelerators import CPUAccelerator
from pytorch_lightning.trainer.trainer import Trainer
from nemo.utils import logging

try:
    from apex.transformer.enums import AttnMaskType
    from apex.transformer.pipeline_parallel.utils import get_num_microbatches

    HAVE_APEX = True
except (ImportError, ModuleNotFoundError):
    HAVE_APEX = False


try:
    from megatron.core import parallel_state
    from megatron.core.pipeline_parallel.schedules import get_forward_backward_func

    HAVE_MEGATRON_CORE = True
except (ImportError, ModuleNotFoundError):

    HAVE_MEGATRON_CORE = False

class MegatronCLIPFeatureFusionModel(MegatronCLIPModel):
    def __init__(self, cfg: DictConfig, trainer: Trainer, pre_process=True, post_process=True):
        super().__init__(cfg, trainer)

    # TODO add dataset support
    def setup(self, stage=None):
        """ PTL hook that is executed after DDP spawns.
            We setup datasets here as megatron datasets require DDP to instantiate.
            See https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#setup for more information.
        Args:
            stage (str, optional): Can be 'fit', 'validate', 'test' or 'predict'. Defaults to None.
        """

        # log number of parameters
        if isinstance(self.model, list):
            num_parameters_on_device = sum(
                [sum([p.nelement() for p in model_module.parameters()]) for model_module in self.model]
            )
        else:
            num_parameters_on_device = sum([p.nelement() for p in self.model.parameters()])

        # to be summed across data parallel group
        total_num_parameters = torch.tensor(num_parameters_on_device).cuda()

        torch.distributed.all_reduce(total_num_parameters, group=parallel_state.get_model_parallel_group())

        logging.info(
            f'Pipeline model parallel rank: {parallel_state.get_pipeline_model_parallel_rank()}, '
            f'Tensor model parallel rank: {parallel_state.get_tensor_model_parallel_rank()}, '
            f'Number of model parameters on device: {num_parameters_on_device:.2e}. '
            f'Total number of model parameters: {total_num_parameters:.2e}.'
        )

        resume_checkpoint_path = self.trainer.ckpt_path
        if resume_checkpoint_path:
            init_consumed_samples = self._extract_consumed_samples_from_ckpt(resume_checkpoint_path)
        else:
            init_consumed_samples = 0
        self.init_consumed_samples = init_consumed_samples
        self.init_global_step = self.trainer.global_step

        # Batch size need to be provided for dataset
        self._num_micro_batches = get_num_microbatches()
        self._micro_batch_size = self.cfg.micro_batch_size

        # when using pipeline model parallel the final stage need to initialize word embeddings
        if parallel_state.get_pipeline_model_parallel_world_size() > 1:
            if isinstance(self.model, list):
                for i, module in enumerate(self.model):
                    parallel_state.set_virtual_pipeline_model_parallel_rank(i)
                parallel_state.set_virtual_pipeline_model_parallel_rank(0)
        #ToDo: check the configuration of megatron_T5 model and replace this
        conf_t5 = T5Config()
        conf_t5.num_layers = 2
        conf_t5.num_decoder_layers = 2
        conf_t5.num_heads = 12
        conf_t5.d_model = 512
        conf_t5.d_kv = 64
        self.t5_layers = T5Stack(conf_t5)

    def encode_text(self, text_tensor):
        """ 
        text_tensor [bs, seq_len]
        output_tensor [bs, seq_len, hidden_size]
        """
        output_tensor = self.model.text_encoder(text_tensor)
        x = self.model.text_encoder.language_model.embedding.word_embeddings(text_tensor) # [batch_size, seq_len, d_model]
        x = x + self.model.text_encoder.language_model.embedding.position_embeddings.weight
        x = x.permute(1, 0, 2)  # NLD -> LND
        #TOdo: Check if the attn_mask is incorporated correctly here. 
        attn_mask = self.model.text_encoder.build_attention_mask(self.cfg.text.max_position_embeddings)
        x = self.model.text_encoder.language_model.encoder(x, attn_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD
        output_tensor = self.model.text_encoder.head(x)

        return output_tensor

    def encode_image(self, image_tensor):
        """ 
        image_tensor [bs, channel, image_h, image_w]
        output_tensor [bs, seq_len + class_token_len, hidden_size]
        """
        # Borrowed scripts from NeMo/nemo/collections/vision/modules/vit/vit_backbone.py
        if self.cfg.vision.pre_process:
            rearranged_input = einops.rearrange(
                image_tensor, "b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=self.cfg.vision.patch_dim, p2=self.cfg.vision.patch_dim,
            )
            # [b num_patch patch_dim*patch_dim*c] ->  [b, s, h]; s:=num_patch, h:=hidden
            encoder_output = self.model.vision_encoder.backbone.linear_encoder(rearranged_input)
            concatenated_tokens = encoder_output
            if self.cfg.vision.class_token_length:
                cls_tokens = self.model.vision_encoder.backbone.cls_token.expand(encoder_output.shape[0], -1, -1)
                concatenated_tokens = torch.cat((cls_tokens, encoder_output), dim=1)

            if self.cfg.vision.position_embedding_type == "learned_absolute":
                token_embeddings = concatenated_tokens + self.model.vision_encoder.backbone.position_embeddings(
                    torch.arange(concatenated_tokens.shape[1]).expand(1, -1).cuda()[:, : concatenated_tokens.shape[1]]
                )
            elif self.cfg.vision.position_embedding_type == "learned_parameters":
                token_embeddings = concatenated_tokens + self.model.vision_encoder.backbone.interpolate_pos_encoding(concatenated_tokens)
            else:
                raise ValueError(f"Unrecognized position embedding type: {self.cfg.vision.position_embedding_type}.")

            # a patch_dropout of 0. would mean it is disabled and this function would do nothing but return what was passed in
            token_embeddings = self.model.vision_encoder.backbone.drop_patch(token_embeddings)

            if self.cfg.vision.preprocess_layernorm is not None:
                token_embeddings = self.model.vision_encoder.backbone.preprocess_layernorm(token_embeddings)

            # [b s h] => [s b h]
            token_embeddings = token_embeddings.transpose(0, 1).contiguous()
            hidden_states = self.model.vision_encoder.backbone.embedding_dropout(token_embeddings)
        else:
            hidden_states = image_tensor

        hidden_states = self.model.vision_encoder.backbone.transformer(hidden_states, None)
        if self.cfg.vision.post_process:
            # [s b h] => [b s h]
            hidden_states = hidden_states.transpose(0, 1).contiguous()
        output_tensor = self.model.vision_encoder.head(hidden_states)

        return output_tensor

    def forward(self, batch):

        txt_batched = batch["txt_batched"]
        image_batched = batch["image_batched"] #[bs, width, img_h, img_w]
        index_mapping = batch["index_mapping"]
        text_features = self.encode_text(txt_batched)
        image_features = self.encode_image(image_batched)
        combined_features = torch.cat([text_features, image_features], dim=1) # shape: [batch_size, seq_len, embed_dim]
        transformer_output = self.t5_layers(
            inputs_embeds=combined_features,
            attention_mask=None,
            use_cache=False,
            return_dict=True
        )
        def mean_pooling(embeddings):
            return torch.mean(embeddings, dim=1)
        # Pool the output of the T5 transformer to get the final features
        embeddings = mean_pooling(transformer_output.last_hidden_state)
        query_embeds = embeddings[torch.tensor(index_mapping["query"]).flatten()]
        pos_cand_embeds = embeddings[torch.tensor(index_mapping["pos_cand"]).flatten()]
        output_tensor = query_embeds, pos_cand_embeds

        return output_tensor

    def get_forward_output_and_loss_func(self):
        loss_func = InbatchContrastiveLoss(
            local_loss=self.cfg.local_loss, gather_with_grad=self.cfg.gather_with_grad, enable_hard_neg=False
        )

        def fwd_output_and_loss_func(dataloader_iter, model):
            batch = next(dataloader_iter)
            if parallel_state.get_pipeline_model_parallel_world_size() == 1:

                batch["txt_batched"] = batch["txt_batched"].to(device='cuda', non_blocking=True)
                batch["image_batched"] = batch["image_batched"].to(device='cuda', non_blocking=True)
                batch["txt_mask_batched"] = batch["txt_mask_batched"].to(device='cuda', non_blocking=True)
                batch["image_mask_batched"] = batch["image_mask_batched"].to(device='cuda', non_blocking=True)

            else:
                # GPT3 uses only causal mask, which doesn't need attention mask
                if parallel_state.is_pipeline_first_stage():
                    # Fist pipeline stage needs only the tokens and position_ids
                    batch["txt_batched"] = batch["txt_batched"].to(device='cuda', non_blocking=True)
                    batch["image_batched"] = batch["image_batched"].to(device='cuda', non_blocking=True)
                    batch["txt_mask_batched"] = batch["txt_mask_batched"].to(device='cuda', non_blocking=True)
                    batch["image_mask_batched"] = batch["image_mask_batched"].to(device='cuda', non_blocking=True)
                else:
                    # Intermediate / Last pipeline stage doesn't need any inputs
                    batch = None
            outputs = self.forward(batch)
            return outputs, loss_func

        return fwd_output_and_loss_func
