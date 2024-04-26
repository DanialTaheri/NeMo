import sys
from nemo.collections.multimodal.data.clip.clip_dataset import get_preprocess_fns
from nemo.collections.multimodal.losses.clip_loss import InbatchContrastiveLoss
from nemo.collections.multimodal.models.vision_language_foundation.clip.megatron_clip_models import (
    MegatronCLIPModel,
    CLIPVisionTransformer,
    CLIPTextTransformer,
    CLIPModel
)
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler
from nemo.collections.multimodal.data.clip.mbeir_dataset import (
    MBEIRMainCollator,
    MBEIRMainDataset,
    Mode,
)
from typing import Any, Optional, Union, Dict, Iterator, List, Tuple

from nemo.core.classes.common import PretrainedModelInfo
from transformers.models.t5.modeling_t5 import T5Stack
from transformers.models.t5 import T5Config

import einops
import numpy as np
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
from nemo.collections.nlp.parts.utils_funcs import get_last_rank, torch_dtype_from_precision
from nemo.collections.nlp.modules.common.megatron.megatron_encoders import get_encoder_model
from nemo.collections.nlp.modules.common.megatron.megatron_decoders import get_decoder_model
from nemo.collections.nlp.modules.common.megatron.megatron_encoder_decoder import (
    MegatronTransformerEncoderDecoderModule,
)
from nemo.collections.nlp.modules.common.megatron.build_model import build_model
from nemo.collections.nlp.models.language_modeling.megatron_base_model import MegatronBaseModel
from nemo.collections.nlp.modules.common.megatron.module import Float16Module, MegatronModule
from nemo.collections.nlp.modules.common.megatron.utils import (
    average_losses_across_data_parallel_group,
    get_all_params_for_weight_decay_optimization,
    get_params_for_weight_decay_optimization,
    init_method_normal,
    scaled_init_method_normal,
)
from nemo.collections.multimodal.parts.utils import load_nemo_model_weights
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
        
class CLIPFeatureFusionModel(MegatronModule):
    def __init__(self, model_cfg, model_parallel_config, padded_vocab_size, pre_process=True, post_process=True):
        super().__init__()

        self.model_cfg = model_cfg
        self.config = model_parallel_config
        self.pre_process = pre_process
        self.post_process = post_process

        self.vision_encoder = CLIPVisionTransformer(
            model_cfg.vision,
            model_parallel_config, 
            pre_process=self.pre_process, 
            post_process=self.post_process,
            )

        self.text_encoder = CLIPTextTransformer(
            model_cfg.text,
            model_parallel_config,
            padded_vocab_size,
            pre_process=self.pre_process,
            post_process=self.post_process,
            )
        # self.clip_model = CLIPModel(
        #     model_cfg,
        #     model_parallel_config,
        #     padded_vocab_size,
        #     pre_process=self.pre_process,
        #     post_process=self.post_process,
        # )
        if model_cfg.restore_from_path.endswith(".nemo") or os.path.isdir(model_cfg.restore_from_path):
            state_dict = load_nemo_model_weights(model_cfg.restore_from_path)[0]
            import collections
            vision_state_dict = collections.OrderedDict()
            text_state_dict = collections.OrderedDict()
            
            for key, val in state_dict.items():
                encoder_type = '.'.join(key.split('.')[:2])
                if encoder_type == 'model.vision_encoder':
                    vision_state_dict['.'.join(key.split('.')[2:])] = val
                elif encoder_type == 'model.text_encoder':
                    text_state_dict['.'.join(key.split('.')[2:])] = val
            self.vision_encoder.load_state_dict(vision_state_dict)
            self.text_encoder.load_state_dict(text_state_dict)

        self.transformer_layer = MegatronTokenLevelEncoderDecoderModule(
            config=model_parallel_config,
            encoder_cfg=model_cfg.t5,
            decoder_cfg=model_cfg.t5,
            vocab_size=padded_vocab_size,
            max_position_embeddings=512,
            num_tokentypes=0,
            parallel_output=True,
            pre_process=pre_process,
            post_process=post_process,
            fp16_cross_entropy=model_cfg.get('fp16_lm_cross_entropy', False),
            precision=model_cfg.get('precision', 32),
            embedding_init_method_std=0.02,
            embedding_dropout=0.1,
            label_smoothing=model_cfg.get('label_smoothing', 0.0),
            add_encoder=True,
            add_decoder=False,
            share_token_embeddings=model_cfg.get('share_token_embeddings', True),
            share_decoder_tokens_head_embeddings=model_cfg.get('share_decoder_tokens_head_embeddings', True),
            tokens_head_bias=model_cfg.get('tokens_head_bias', True),
            hiddens_cfg=model_cfg.get('hiddens', None),
        )

        self.logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def set_input_tensor(self, input_tensor):
        """See megatron.model.transformer.set_input_tensor()"""
        pass

    def encode_image(self, image_tensor):
        """ 
        image_tensor [bs, channel, image_h, image_w]
        output_tensor [bs, seq_len + class_token_len, hidden_size]
        """
        # Borrowed scripts from NeMo/nemo/collections/vision/modules/vit/vit_backbone.py
        hidden_states = self.vision_encoder.backbone(image_tensor)
        # if self.model_cfg.vision.pre_process:
        #     rearranged_input = self.vision_encoder.backbone.conv1(image_tensor)
        #     rearranged_input = rearranged_input.reshape(rearranged_input.shape[0], rearranged_input.shape[1], -1)
        #     encoder_output = rearranged_input.permute(0, 2, 1)

        #     concatenated_tokens = encoder_output
        #     if self.model_cfg.vision.class_token_length:
        #         cls_tokens = self.vision_encoder.backbone.cls_token.expand(encoder_output.shape[0], -1, -1)
        #         concatenated_tokens = torch.cat((cls_tokens, encoder_output), dim=1)

        #     if self.model_cfg.vision.position_embedding_type == "learned_absolute":
        #         token_embeddings = concatenated_tokens + self.vision_encoder.backbone.position_embeddings(
        #             torch.arange(concatenated_tokens.shape[1]).expand(1, -1).cuda()[:, : concatenated_tokens.shape[1]]
        #         )
        #     elif self.model_cfg.vision.position_embedding_type == "learned_parameters":
        #         token_embeddings = concatenated_tokens + self.vision_encoder.backbone.interpolate_pos_encoding(concatenated_tokens)
        #     else:
        #         raise ValueError(f"Unrecognized position embedding type: {self.model_cfg.vision.position_embedding_type}.")

        #     # a patch_dropout of 0. would mean it is disabled and this function would do nothing but return what was passed in
        #     token_embeddings = self.vision_encoder.backbone.drop_patch(token_embeddings)

        #     if self.model_cfg.vision.preprocess_layernorm is not None:
        #         token_embeddings = self.vision_encoder.backbone.preprocess_layernorm(token_embeddings)

        #     # [b s h] => [s b h]
        #     token_embeddings = token_embeddings.transpose(0, 1).contiguous()
        #     hidden_states = self.vision_encoder.backbone.embedding_dropout(token_embeddings)
        # else:
        #     hidden_states = image_tensor

        # hidden_states = self.vision_encoder.backbone.transformer(hidden_states, None)
        if self.model_cfg.vision.post_process:
            # [s b h] => [b s h]
            #hidden_states = hidden_states.transpose(0, 1).contiguous()
            output_tensor = self.vision_encoder.head(hidden_states)
        return output_tensor

    def encode_text(self, text_tensor):
        """ 
        text_tensor [bs, seq_len]
        output_tensor [bs, seq_len, hidden_size]
        """
        output_tensor = self.text_encoder.language_model(text_tensor,
                self.text_encoder.position_ids, 
                self.text_encoder.attn_mask)
        output_tensor = output_tensor.permute(1, 0, 2)  # LND -> NLD
        return output_tensor

    def forward(self, images, captions):
        image_features = self.encode_image(images)
        text_features = self.encode_text(captions)
        #import pdb; pdb.set_trace()
        combined_features = torch.cat([text_features, image_features], dim=1) # shape: [batch_size, seq_len, embed_dim]
        
        enc_attn_mask = torch.ones(combined_features.shape[0], combined_features.shape[1], device=combined_features.device, dtype=torch.long)
        transformer_output = self.transformer_layer(enc_input=combined_features, enc_attn_mask=enc_attn_mask)
        transformer_output = transformer_output.permute(1, 0, 2)
        def mean_pooling(embeddings):
            return torch.mean(embeddings, dim=1)
        embeddings = mean_pooling(transformer_output)
        return embeddings

class MegatronCLIPFeatureFusionModel(MegatronBaseModel):
    def __init__(self, cfg: DictConfig, trainer: Trainer, pre_process=True, post_process=True):
        super().__init__(cfg, trainer)
        if not HAVE_APEX:
            raise ImportError(
                "Apex was not found. Please see the NeMo README for installation instructions: https://github.com/NVIDIA/NeMo#megatron-gpt."
            )
        if not HAVE_MEGATRON_CORE:
            raise ImportError(
                "megatron-core was not found. Please see the NeMo README for installation instructions: https://github.com/NVIDIA/NeMo#megatron-gpt."
            )
        self.tokenizer = clip.tokenize
        self._validate_trainer()

        self.megatron_amp_O2 = cfg.get('megatron_amp_O2', False)

        if not self.megatron_amp_O2 and self.cfg.get('virtual_pipeline_model_parallel_size', None):
            raise ValueError('Virtual pipeline model parallel is only supported when using megatron_amp_O2')
        # build_model returns a list of modules which are used for interleaved pipeline parallelism
        if isinstance(self.trainer.accelerator, CPUAccelerator):
            self.model = build_model(
                model_provider_func=self.model_provider_func,
                wrap_with_ddp=False,
                on_cpu=True,
                virtual_pipeline_model_parallel_size=self.cfg.get('virtual_pipeline_model_parallel_size', None),
            )
        else:
            self.model = build_model(
                model_provider_func=self.model_provider_func,
                wrap_with_ddp=False,
                virtual_pipeline_model_parallel_size=self.cfg.get('virtual_pipeline_model_parallel_size', None),
            )
        # if we're not using interleaved, then self.model is a module.
        if self.cfg.get('virtual_pipeline_model_parallel_size', None) is None:
            self.model = self.model[0]

        if self.megatron_amp_O2:

            if not self.with_distributed_adam:
                # Pre-allocate the model on GPU to have master parameters allocated on the same device with matching data type
                if isinstance(self.model, list):
                    for module in self.model:
                        module.cuda(torch.cuda.current_device())
                else:
                    self.model.cuda(torch.cuda.current_device())

            # Model wrapper to convert both model and inputs to half precision
            # TODO (yuya): check this; FP16 Module might not work; when self.model is a list?
            if isinstance(self.model, list):
                converted_model = []
                for module in self.model:
                    converted_model.append(
                        Float16Module(config=self.model_parallel_config, module=module, precision=cfg.precision)
                    )
                    self.model = converted_model
            else:
                self.model = Float16Module(
                    config=self.model_parallel_config, module=self.model, precision=cfg.precision
                )

        self.autocast_dtype = torch_dtype_from_precision(self.trainer.precision)
        self.enable_autocast = (
            True if (not self.megatron_amp_O2) and (self.autocast_dtype in [torch.float16, torch.bfloat16]) else False
        )

        self.transformer_engine = cfg.get('transformer_engine', False)

        # Convert the global-batch-based profile index to micro-batch index
        if hasattr(self, '_nsys_profile_enabled'):
            mp_size = cfg.get('tensor_model_parallel_size', 1) * cfg.get('pipeline_model_parallel_size', 1)
            data_parallel_world_size = trainer.world_size // mp_size
            grad_accum_steps = cfg.get('global_batch_size') // (cfg.get('micro_batch_size') * data_parallel_world_size)
            self._nsys_profile_start_step *= grad_accum_steps
            self._nsys_profile_end_step *= grad_accum_steps
        self.get_attention_mask_from_fusion = self.cfg.get('get_attention_mask_from_fusion', True)
        self.initialize_ub = self.cfg.get('ub_tp_comm_overlap', False)

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
        self.setup_validation_data()
        # when using pipeline model parallel the final stage need to initialize word embeddings
        if parallel_state.get_pipeline_model_parallel_world_size() > 1:
            if isinstance(self.model, list):
                for i, module in enumerate(self.model):
                    parallel_state.set_virtual_pipeline_model_parallel_rank(i)
                parallel_state.set_virtual_pipeline_model_parallel_rank(0)

    def get_module_list(self):
        if isinstance(self.model, list):
            return [model.module if isinstance(model, Float16Module) else model for model in self.model]
        elif isinstance(self.model, Float16Module):
            return [self.model.module]
        else:
            return [self.model]

    def model_provider_func(self, pre_process, post_process):
        """Model depends on pipeline paralellism."""
        model = CLIPFeatureFusionModel(
            model_cfg=self.cfg,
            model_parallel_config=self.model_parallel_config,
            padded_vocab_size=self.padded_vocab_size,
            pre_process=pre_process,
            post_process=post_process,
        )
        return model

    def setup_optimizer_param_groups(self):
        """ModelPT override. Optimizer will get self._optimizer_param_groups"""
        if self.cfg.get('do_layer_norm_weight_decay', False):
            if isinstance(self.model, list):
                self._optimizer_param_groups = get_all_params_for_weight_decay_optimization(self.model)
            else:
                self._optimizer_param_groups = get_all_params_for_weight_decay_optimization([self.model])

        else:
            self._optimizer_param_groups = get_params_for_weight_decay_optimization(self.model)
        # filter out params doesn't have grad
        for param_group in self._optimizer_param_groups:
            params_with_grad = [param for param in param_group['params'] if param.requires_grad]
            param_group['params'] = params_with_grad
        # Mapping from parameter objects to their names
        param_to_name = {
            param: name
            for name, param in self.model.named_parameters()
        }
        group1_params, group2_params = [], []
        for param in self._optimizer_param_groups[0]['params']:
            param_name = param_to_name.get(param)
            if 'transformer_layer' in param_name.split(".")[0]:
                group2_params.append(param)
            else:
                group1_params.append(param)

        group1_nodecay_params, group2_nodecay_params = [], []
        for param in self._optimizer_param_groups[1]['params']:
            param_name = param_to_name.get(param)
            if 'transformer_layer' in param_name.split(".")[0]:
                group2_nodecay_params.append(param)
            else:
                group1_nodecay_params.append(param)
        
        base_lr = self._cfg.optim.get('lr')
        transformer_layer_lr_ratio = 15
        # Create two new optimizer param groups
        self._optimizer_param_groups = [
            {'params': group1_params, 'lr': base_lr},
            {'params': group2_params, 'lr': base_lr * transformer_layer_lr_ratio},
            {'params': group1_nodecay_params, "weight_decay": 0.0, 'lr': base_lr},
            {'params': group2_nodecay_params, "weight_decay": 0.0, 'lr': base_lr * transformer_layer_lr_ratio},
        ]

    def configure_optimizers(self):
        #check with Ali about this 
        if self.with_distributed_adam:

            # Disable overlapped grad sync for layer norm grads when
            # sequence parallelism is enabled
            for param in self.parameters():
                if getattr(param, 'sequence_parallel', False):
                    param._disable_greedy_grad_copy = not self.megatron_amp_O2
                    param._disable_overlap_grad_sync = True

            # Initialize parameter buckets for overlapped grad and param syncs
            # Note: Params with disabled overlapping are put in the
            # last param bucket
            buckets = []
            if self.cfg.get('virtual_pipeline_model_parallel_size', None) is not None:
                # Initialize a bucket for each virtual pipeline stage
                for module in self.model:
                    if isinstance(module, Float16Module):
                        module = module.module
                    stage_bucket = []
                    for layer in itertools.chain(
                        module.vision_encoder.backbone.transformer.layers,
                        module.text_encoder.language_model.encoder.layers,
                    ):
                        stage_bucket.extend(
                            p for p in layer.parameters() if not getattr(p, '_disable_overlap_grad_sync', False)
                        )
                    buckets.append(stage_bucket)
            else:
                # Initialize a bucket for each Transformer layer
                modules = self.model if isinstance(self.model, list) else [self.model]
                for module in modules:
                    if isinstance(module, Float16Module):
                        module = module.module
                    for layer in itertools.chain(
                        module.vision_encoder.backbone.transformer.layers,
                        module.text_encoder.language_model.encoder.layers,
                    ):
                        buckets.append(
                            [p for p in layer.parameters() if not getattr(p, '_disable_overlap_grad_sync', False)]
                        )
            buckets.reverse()
            used_params = set()
            for bucket in buckets:
                used_params.update(bucket)
            buckets[-1].extend(p for p in self.parameters() if p not in used_params)
            self.distributed_adam_buckets = buckets

        return super().configure_optimizers()

    def training_step(self, dataloader_iter, batch_idx):
        """
            Our dataloaders produce a micro-batch and then we fetch
            a number of microbatches depending on the global batch size and model parallel size
            from the dataloader to produce a list of microbatches.
            Batch should be a list of microbatches and those microbatches should on CPU.
            Microbatches are then moved to GPU during the pipeline.
            The list of microbatches is then piped through the pipeline using Apex fwd/bwd functions.
        """
        # Initialize userbuffer communicators.
        if self.initialize_ub:
            self.initialize_ub_func()

        # we zero grads here because we also call backward in the megatron-core fwd/bwd functions
        self._optimizer.zero_grad()

        if self.with_distributed_adam:
            # hack to enable overlapping param sync and forward compute
            # note: the distributed optimizer monkey-patches each
            # parameter's __getattribute__ function so that it can
            # launch parameter all-gathers the first time the
            # parameter is accessed after the optimizer step. However,
            # PyTorch directly passes embedding parameters into a C++,
            # bypassing this process. A quick-and-dirty hack is to
            # manually interact with the parameter.
            modules = self.model if isinstance(self.model, list) else [self.model]
            for module in modules:
                if isinstance(module, Float16Module):
                    module = module.module
                module = module.text_encoder.language_model
                if hasattr(module, 'embedding'):
                    for param in module.embedding.parameters():
                        param.data_ptr()

        loss_mean = self.fwd_bwd_step(dataloader_iter, batch_idx, False)

        # when using sequence parallelism, the sequence parallel layernorm grads must be all-reduced
        if self.cfg.get('tensor_model_parallel_size', 1) > 1 and self.cfg.get('sequence_parallel', False):
            self.allreduce_sequence_parallel_gradients()

        if self.with_distributed_adam:
            # synchronize asynchronous grad reductions
            # note: not necessary, but reduces performance degradation
            # from multiple simultaneous NCCL calls
            self._optimizer._finish_bucket_grad_sync()
        elif self.megatron_amp_O2:
            # when using pipeline parallelism grads must be all-reduced after the pipeline (not asynchronously)
            # if self.cfg.get('pipeline_model_parallel_size', 1) > 1 or self.cfg.get('sequence_parallel', False):
            #     # main grads are stored in the MainParamsOptimizer wrapper
            self._optimizer.allreduce_main_grads()
        else:
            # async grad allreduce is not currently implemented for O1/autocasting mixed precision training
            # so we all-reduce gradients after the pipeline
            self.allreduce_gradients()  # @sangkug we think this is causing memory to blow up (hurts perf)

        ## logging
        # we can only log on one rank if it is rank zero so we broadcast from last rank
        # we can avoid this broadcast by updating the PTL log function to accept specific ranks
        torch.distributed.broadcast(loss_mean, get_last_rank())

        if self.cfg.precision in [16, '16', '16-mixed']:
            loss_scale = self.trainer.precision_plugin.scaler._scale
            if loss_scale is not None:
                self.log('loss_scale', loss_scale, batch_size=1)

        self.log('reduced_train_loss', loss_mean, prog_bar=True, rank_zero_only=True, batch_size=1)
        lr = self._optimizer.param_groups[0]['lr']
        self.log('lr', lr, rank_zero_only=True, batch_size=1)
        self.log('global_step', self.trainer.global_step + 1, prog_bar=True, rank_zero_only=True, batch_size=1)
        self.log(
            'consumed_samples',
            self.compute_consumed_samples(self.trainer.global_step + 1 - self.init_global_step),
            prog_bar=True,
            rank_zero_only=True,
            batch_size=1,
        )

        return loss_mean

    def forward(self, batch):

        txt_batched = batch["txt_batched"]
        image_batched = batch["image_batched"] #[bs, width, img_h, img_w]
        txt_mask_batched = batch["txt_mask_batched"]
        image_mask_batched = batch["image_mask_batched"]
        index_mapping = batch["index_mapping"]

        embeddings = self.model(image_batched, txt_batched)
        query_embeds = embeddings[torch.tensor(index_mapping["query"]).flatten()]
        pos_cand_embeds = embeddings[torch.tensor(index_mapping["pos_cand"]).flatten()]
        output_tensors = query_embeds, pos_cand_embeds

        return output_tensors

    def fwd_bwd_step(self, dataloader_iter, batch_idx, forward_only):

        # handle asynchronous grad reduction
        no_sync_func = None
        grad_sync_func = None
        param_sync_func = None
        if not forward_only and self.with_distributed_adam:
            no_sync_func = partial(self._optimizer.no_sync, greedy_grad_copy=self.megatron_amp_O2,)
            grad_sync_func = self.reduce_overlap_gradients
            param_sync_func = self.sync_overlap_parameters

        # pipeline schedules will get these from self.model.config
        for module in self.get_module_list():
            module.config.no_sync_func = no_sync_func
            module.config.grad_sync_func = grad_sync_func
            module.config.param_sync_func = param_sync_func

        # run forward and backwards passes for an entire global batch
        # we do this inside training_step to support pipeline parallelism
        fwd_bwd_function = get_forward_backward_func()

        # TODO @akhattar: add num_micro_batches_with_partial_activation_checkpoints when ready
        losses_reduced_per_micro_batch = fwd_bwd_function(
            forward_step_func=self.get_forward_output_and_loss_func(),
            data_iterator=dataloader_iter,
            model=self.model,
            num_microbatches=get_num_microbatches(),
            forward_only=forward_only,
            seq_length=None,
            micro_batch_size=self.cfg.micro_batch_size,
        )

        # only the last stages of the pipeline return losses
        if losses_reduced_per_micro_batch:
            if (not forward_only) or self.cfg.data.get('validation_drop_last', True):
                # average loss across micro batches
                loss_tensors_list = [loss_reduced['loss'] for loss_reduced in losses_reduced_per_micro_batch]
                loss_tensor = torch.stack(loss_tensors_list)
                loss_mean = loss_tensor.mean()
            else:
                # Get the total loss since micro batches sizes are not uniform
                raise NotImplementedError("Losses of micro batches sizes must be uniform!")
        else:
            # we're not on the last pipeline stage so no losses
            if forward_only:
                loss_mean = []
            else:
                loss_mean = torch.tensor(0.0).cuda()

        return loss_mean

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

    def backward(self, *args, **kwargs):
        """ LightningModule hook to do backward.
            We want this to do nothing since we run backward in the fwd/bwd functions from apex.
            No need to call it here.
        """
        pass

    def optimizer_zero_grad(self, *args, **kwargs):
        """ LightningModule hook to zero grad.
            We want this to do nothing as we are zeroing grads during the training_step.
        """
        pass

    def _append_sequence_parallel_module_grads(self, module, grads):
        """ Helper method for allreduce_sequence_parallel_gradients"""

        for param in module.parameters():
            sequence_parallel_param = getattr(param, 'sequence_parallel', False)
            if sequence_parallel_param and param.requires_grad:
                if self.megatron_amp_O2:
                    grad = param.main_grad
                else:
                    grad = param.grad
                grads.append(grad.data)

    def allreduce_sequence_parallel_gradients(self):
        """ All-reduce layernorm parameters across model parallel nodes when sequence parallelism is used.
            Modified from megatron-lm:
            https://gitlab-master.nvidia.com/ADLR/megatron-lm/-/blob/3f91f09bb2ab32f9904b47f46f19d2fc3f518ed8/megatron/training.py#L425
        """

        grads = []
        if isinstance(self.model, list):
            for module in self.model:
                self._append_sequence_parallel_module_grads(module, grads)
        else:
            self._append_sequence_parallel_module_grads(self.model, grads)

        coalesced = torch._utils._flatten_dense_tensors(grads)
        torch.distributed.all_reduce(coalesced, group=parallel_state.get_tensor_model_parallel_group())
        for buf, synced in zip(grads, torch._utils._unflatten_dense_tensors(coalesced, grads)):
            buf.copy_(synced)

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
    def setup_validation_data(self):
        val_image_transform, text_transform = get_preprocess_fns(self.cfg, self.tokenizer, is_train=False,)
        #data loaders
        self._eval_ds = MBEIRMainDataset(
                        mbeir_data_dir=self.cfg.data_config.mbeir_data_dir,
                        query_data_path=self.cfg.data_config.val_query_data_path,
                        cand_pool_path=self.cfg.data_config.val_cand_pool_path,
                        query_instruct_path=self.cfg.data_config.query_instruct_path,
                        img_preprocess_fn=val_image_transform,
                        mode=Mode.EVAL,
                        enable_query_instruct=self.cfg.data_config.enable_query_instruct,
                        shuffle_cand=self.cfg.data_config.shuffle_cand,
                        hard_neg_num=0, # TODO 
                        returns=self.cfg.data_config.returns,
                        ) 
        valid_collector = MBEIRMainCollator(
                        tokenizer=self.get_tokenizer(),
                        image_size=tuple(map(int, self.cfg.data_config.image_size.split(','))),
                        mode=Mode.EVAL,
                        )
        valid_sampler = DistributedSampler(
                        dataset=self._eval_ds,
                        num_replicas=1,
                        rank=0,
                        shuffle=True,
                        )
        self._valid_dl = DataLoader(
                    dataset=self._eval_ds,
                    batch_size=self.cfg.dataloader_config.valid_batch_size,
                    num_workers=self.cfg.dataloader_config.num_workers,
                    pin_memory=True,
                    sampler=valid_sampler,
                    shuffle=False,  # Note: since we use sampler, shuffle should be False
                    collate_fn=valid_collector,
                    drop_last=True,
                    )

    def _validate_trainer(self):
        """ Certain trainer configurations can break training.
            Here we try to catch them and raise an error.
        """
        if self.trainer.accumulate_grad_batches > 1:
            raise ValueError(
                f'Gradient accumulation is done within training_step. trainer.accumulate_grad_batches must equal 1'
            )
    @classmethod
    def list_available_models(cls) -> Optional[PretrainedModelInfo]:
        return None