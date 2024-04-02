import sys
from nemo.collections.multimodal.data.clip.clip_dataset import get_preprocess_fns
from nemo.collections.multimodal.losses.clip_loss import InbatchContrastiveLoss
from nemo.collections.multimodal.models.vision_language_foundation.clip.megatron_clip_models import (
    MegatronCLIPModel,
)
from torch.utils.data import DataLoader, DistributedSampler
from nemo.collections.multimodal.data.clip.mbeir_dataset import (
    MBEIRMainCollator,
    MBEIRMainDataset,
    Mode,
)
from nemo.core.classes.common import PretrainedModelInfo
from transformers.models.t5.modeling_t5 import T5Stack
from transformers.models.t5 import T5Config

import einops
import torch
import clip
import torch.distributed.nn
from omegaconf.dictconfig import DictConfig
from pytorch_lightning.accelerators import CPUAccelerator
from pytorch_lightning.trainer.trainer import Trainer
from nemo.utils import logging


from nemo.collections.nlp.models.language_modeling.megatron_t5_model import MegatronT5Model
from nemo.collections.nlp.modules.common.megatron.token_level_encoder_decoder import (
    MegatronTokenLevelEncoderDecoderModule,
)
from nemo.collections.nlp.modules.common.megatron.megatron_encoders import get_encoder_model
from nemo.collections.nlp.modules.common.megatron.megatron_decoders import get_decoder_model
from nemo.collections.nlp.modules.common.megatron.megatron_encoder_decoder import (
    MegatronTransformerEncoderDecoderModule,
)
from nemo.collections.nlp.modules.common.megatron.utils import (
    ApexGuardDefaults,
    build_position_ids,
    init_method_normal,
    parallel_lm_logits,
    scaled_init_method_normal,
)
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
try:
    from apex.transformer.enums import AttnMaskType, ModelType

    HAVE_APEX = True

except (ImportError, ModuleNotFoundError):

    # fake missing classes with None attributes
    AttnMaskType = ApexGuardDefaults()
    ModelType = ApexGuardDefaults()

    HAVE_APEX = False
class MegatronCLIPFeatureFusionModel(MegatronCLIPModel):
    def __init__(self, cfg: DictConfig, trainer: Trainer, pre_process=True, post_process=True):
        super().__init__(cfg, trainer)

        self.tokenizer = clip.tokenize
        #self.t5_layers = MegatronT5Model(self.cfg.t5, trainer)

    def get_tokenizer(self):
        def tokenizer_wrapper(txt):
            tokenizer = self.tokenizer
            txt_tensor = tokenizer(txt, context_length=77, truncate=True)
            return txt_tensor
        return tokenizer_wrapper

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
        self.setup_training_data()
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

        # encoder = get_encoder_model(
        #         config=self.cfg.t5,
        #         arch=self.cfg.t5.arch,
        #         hidden_size=self.cfg.t5.hidden_size,
        #         ffn_hidden_size=self.cfg.t5.ffn_hidden_size,
        #         num_layers=self.cfg.t5.num_layers,
        #         num_attention_heads=self.cfg.t5.num_attention_heads,
        #         apply_query_key_layer_scaling=self.cfg.t5.get('apply_query_key_layer_scaling', True),
        #         kv_channels=self.cfg.t5.kv_channels,
        #         init_method=init_method_normal(self.cfg.t5.get('init_method_std', 0.02)),
        #         scaled_init_method=scaled_init_method_normal(
        #             self.cfg.t5.get('init_method_std', 0.02), self.cfg.t5.num_layers
        #         ),
        #         encoder_attn_mask_type=AttnMaskType.padding,
        #         pre_process=True,
        #         post_process=False,
        #         init_method_std=self.cfg.t5.get('init_method_std', 0.02),
        #         hidden_dropout=self.cfg.t5.get('hidden_dropout', 0.1),
        #         attention_dropout=self.cfg.t5.get('attention_dropout', 0.1),
        #         ffn_dropout=self.cfg.t5.get('ffn_dropout', 0.0),
        #         precision=self.cfg.get('precision', 16),
        #         fp32_residual_connection=self.cfg.t5.get('fp32_residual_connection', False),
        #         activations_checkpoint_method=self.cfg.t5.get('activations_checkpoint_method', None),
        #         activations_checkpoint_num_layers=self.cfg.t5.get('activations_checkpoint_num_layers', 1),
        #         activations_checkpoint_granularity=self.cfg.t5.get('activations_checkpoint_granularity', None),
        #         layernorm_epsilon=self.cfg.t5.get('layernorm_epsilon', 1e-5),
        #         bias_activation_fusion=self.cfg.t5.get('bias_activation_fusion', True),
        #         bias_dropout_add_fusion=self.cfg.t5.get('bias_dropout_add_fusion', True),
        #         masked_softmax_fusion=self.cfg.t5.get('masked_softmax_fusion', True),
        #         persist_layer_norm=self.cfg.t5.get('persist_layer_norm', True),
        #         openai_gelu=self.cfg.t5.get('openai_gelu', False),
        #         onnx_safe=self.cfg.t5.get('onnx_safe', False),
        #         hidden_steps=self.cfg.t5.get('hidden_steps', -1),
        #         activation=self.cfg.t5.get('activation', 'gelu'),
        #         bias=self.cfg.t5.get('bias', True),
        #         normalization=self.cfg.t5.get('normalization', 'layernorm'),
        #         transformer_block_type=self.cfg.t5.get('transformer_block_type', 'pre_ln'),
        #         headscale=self.cfg.t5.get('headscale', False),
        #         parent_model_type=ModelType.encoder_and_decoder,
        #         num_self_attention_per_cross_attention=self.cfg.t5.get('num_self_attention_per_cross_attention', 1),
        #         megatron_legacy=self.cfg.t5.get('megatron_legacy', False),
        #         normalize_attention_scores=self.cfg.t5.get('normalize_attention_scores', True),
        #         num_moe_experts=self.cfg.t5.get('num_moe_experts', 1),
        #         moe_frequency=self.cfg.t5.get('moe_frequency', 1),
        #         moe_dropout=self.cfg.t5.get('moe_dropout', 0.0),
        #         position_embedding_type=self.cfg.t5.get('position_embedding_type', 'learned_absolute'),
        #         use_flash_attention=self.cfg.t5.get('use_flash_attention', False),
        #     )

        # decoder = get_decoder_model(
        #         config=self.cfg.t5,
        #         arch=self.cfg.t5.arch,
        #         hidden_size=self.cfg.t5.hidden_size,
        #         ffn_hidden_size=self.cfg.t5.ffn_hidden_size,
        #         num_layers=self.cfg.t5.num_layers,
        #         num_attention_heads=self.cfg.t5.num_attention_heads,
        #         apply_query_key_layer_scaling=self.cfg.t5.get('apply_query_key_layer_scaling', True),
        #         kv_channels=self.cfg.t5.kv_channels,
        #         init_method=init_method_normal(self.cfg.t5.get('init_method_std', 0.02)),
        #         scaled_init_method=scaled_init_method_normal(
        #             self.cfg.t5.get('init_method_std', 0.02), self.cfg.t5.num_layers
        #         ),
        #         decoder_attn_mask_type=AttnMaskType.causal,
        #         pre_process=False,
        #         post_process=False,
        #         init_method_std=self.cfg.t5.get('init_method_std', 0.02),
        #         hidden_dropout=self.cfg.t5.get('hidden_dropout', 0.1),
        #         attention_dropout=self.cfg.t5.get('attention_dropout', 0.1),
        #         ffn_dropout=self.cfg.t5.get('ffn_dropout', 0.0),
        #         precision=self.cfg.get('precision', 16),
        #         fp32_residual_connection=self.cfg.t5.get('fp32_residual_connection', False),
        #         activations_checkpoint_method=self.cfg.t5.get('activations_checkpoint_method', None),
        #         activations_checkpoint_num_layers=self.cfg.t5.get('activations_checkpoint_num_layers', 1),
        #         activations_checkpoint_granularity=self.cfg.t5.get('activations_checkpoint_granularity', None),
        #         layernorm_epsilon=self.cfg.t5.get('layernorm_epsilon', 1e-5),
        #         bias_activation_fusion=self.cfg.t5.get('bias_activation_fusion', True),
        #         bias_dropout_add_fusion=self.cfg.t5.get('bias_dropout_add_fusion', True),
        #         masked_softmax_fusion=self.cfg.t5.get('masked_softmax_fusion', True),
        #         persist_layer_norm=self.cfg.t5.get('persist_layer_norm', True),
        #         openai_gelu=self.cfg.t5.get('openai_gelu', False),
        #         onnx_safe=self.cfg.t5.get('onnx_safe', False),
        #         hidden_steps=self.cfg.t5.get('hidden_steps', -1),
        #         activation=self.cfg.t5.get('activation', 'gelu'),
        #         bias=self.cfg.t5.get('bias', True),
        #         normalization=self.cfg.t5.get('normalization', 'layernorm'),
        #         transformer_block_type=self.cfg.t5.get('transformer_block_type', 'pre_ln'),
        #         headscale=self.cfg.t5.get('headscale', False),
        #         parent_model_type=ModelType.encoder_and_decoder,
        #         megatron_legacy=self.cfg.t5.get('megatron_legacy', False),
        #         normalize_attention_scores=self.cfg.t5.get('normalize_attention_scores', True),
        #         num_moe_experts=self.cfg.t5.get('num_moe_experts', 1),
        #         moe_frequency=self.cfg.t5.get('moe_frequency', 1),
        #         moe_dropout=self.cfg.t5.get('moe_dropout', 0.0),
        #         position_embedding_type=self.cfg.t5.get('position_embedding_type', 'learned_absolute'),
        #         use_flash_attention=self.cfg.t5.get('use_flash_attention', False),
        #     )
        # self.enc_dec_model = MegatronTransformerEncoderDecoderModule(
        #     config=self.cfg.t5,
        #     encoder=encoder,
        #     decoder=decoder,
        #     hidden_steps=encoder_cfg.get('hidden_steps', -1),
        #     hiddens_module=hiddens_module,
        # )
        
        # import pdb; pdb.set_trace()
        # self.t5_layers = MegatronTokenLevelEncoderDecoderModule(
        #     config=self.model_parallel_config,
        #     encoder_cfg=self.cfg.t5,
        #     decoder_cfg=self.cfg.t5,
        #     vocab_size=self.padded_vocab_size,
        #     max_position_embeddings=512,
        #     num_tokentypes=0,
        #     parallel_output=True,
        #     pre_process=True,
        #     post_process=True,
        #     fp16_cross_entropy=self.cfg.get('fp16_lm_cross_entropy', False),
        #     precision=self.cfg.get('precision', 16),
        #     embedding_init_method_std=0.02,
        #     embedding_dropout=0.1,
        #     label_smoothing=self.cfg.get('label_smoothing', 0.0),
        #     add_encoder=True,
        #     add_decoder=False,
        #     share_token_embeddings=self.cfg.get('share_token_embeddings', True),
        #     share_decoder_tokens_head_embeddings=self.cfg.get('share_decoder_tokens_head_embeddings', True),
        #     tokens_head_bias=self.cfg.get('tokens_head_bias', True),
        #     hiddens_cfg=self.cfg.get('hiddens', None),
        # )
    def encode_text(self, text_tensor):
        """ 
        text_tensor [bs, seq_len]
        output_tensor [bs, seq_len, hidden_size]
        """
        #output_tensor = self.model.text_encoder(text_tensor)
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
        txt_mask_batched = batch["txt_mask_batched"]
        image_mask_batched = batch["image_mask_batched"]
        index_mapping = batch["index_mapping"]

        ###########
        #This is for getting [bs, seq_len, embed_dimension]
        text_features = self.encode_text(txt_batched)
        image_features = self.encode_image(image_batched)
        # output_tensor = self.model(image_batched, txt_batched)
        # image_features, text_features, _ = output_tensor
        # image_features = image_features.unsqueeze(dim=1)
        # text_features = text_features.unsqueeze(dim=1)
        combined_features = torch.cat([text_features, image_features], dim=1) # shape: [batch_size, seq_len, embed_dim]
        # transformer_output = self.t5_layers(
        #     inputs_embeds=combined_features,
        #     attention_mask=None,
        #     use_cache=False,
        #     return_dict=True
        # )
        # enc_attn_mask = torch.ones(combined_features.shape[0], combined_features.shape[1], dtype=torch.bool).to('cuda')
        # dec_attn_mask = torch.ones(combined_features.shape[0], combined_features.shape[1], dtype=torch.bool).to('cuda')
        #import pdb; pdb.set_trace()
        #transformer_output = self.t5_layers(enc_input=combined_features, enc_attn_mask=enc_attn_mask, dec_attn_mask=dec_attn_mask)
        def mean_pooling(embeddings):
            return torch.mean(embeddings, dim=1)
        # Pool the output of the T5 transformer to get the final features
        embeddings = mean_pooling(transformer_output.last_hidden_state)
        ########
        # output_tensor = self.model(image_batched, txt_batched)
        # image_features, text_features, _ = output_tensor
        # embeddings = image_features * image_mask_batched.unsqueeze(-1) + text_features * txt_mask_batched.unsqueeze(-1)

        query_embeds = embeddings[torch.tensor(index_mapping["query"]).flatten()]
        pos_cand_embeds = embeddings[torch.tensor(index_mapping["pos_cand"]).flatten()]
        output_tensors = query_embeds, pos_cand_embeds

        return output_tensors

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

    def setup_training_data(self):

        val_image_transform, text_transform = get_preprocess_fns(self.cfg, self.tokenizer, is_train=False,)
        #data loaders
        self._train_ds = MBEIRMainDataset(
                        mbeir_data_dir=self.cfg.data_config.mbeir_data_dir,
                        query_data_path=self.cfg.data_config.train_query_data_path,
                        cand_pool_path=self.cfg.data_config.train_cand_pool_path,
                        query_instruct_path=self.cfg.data_config.query_instruct_path,
                        img_preprocess_fn=val_image_transform,
                        mode=Mode.TRAIN,
                        enable_query_instruct=self.cfg.data_config.enable_query_instruct,
                        shuffle_cand=self.cfg.data_config.shuffle_cand,
                        hard_neg_num=0, # TODO 
                        returns=self.cfg.data_config.returns,
                        ) 
        train_collector = MBEIRMainCollator(
                        tokenizer=self.get_tokenizer(),
                        image_size=tuple(map(int, self.cfg.data_config.image_size.split(','))),
                        mode=Mode.TRAIN,
                        )
        train_sampler = DistributedSampler(
                        dataset=self._train_ds,
                        num_replicas=1,
                        rank=0,
                        shuffle=True,
                        )
        self._train_dl = DataLoader(
                    dataset=self._train_ds,
                    batch_size=self.cfg.dataloader_config.train_batch_size,
                    num_workers=self.cfg.dataloader_config.num_workers,
                    pin_memory=True,
                    sampler=train_sampler,
                    shuffle=False,  # Note: since we use sampler, shuffle should be False
                    collate_fn=train_collector,
                    drop_last=True,
                    )