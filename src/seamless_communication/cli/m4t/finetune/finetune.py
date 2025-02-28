# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the license found in the
# MIT_LICENSE file in the root directory of this source tree.

import argparse
import logging
import os
import sys
sys.path.append("/data/donggukang/seamless_test/seamless_communication/src")
os.environ["CUDA_VISIBLE_DEVICES"]="1,2,3,5"
from pathlib import Path
import torch.distributed as dist

import torch

from seamless_communication.cli.m4t.finetune import dataloader, dist_utils, trainer
from seamless_communication.cli.m4t.finetune.trainer import UnitYFinetuneWrapper
from seamless_communication.models.unity import (
    load_unity_model,
    load_unity_text_tokenizer,
    load_unity_unit_tokenizer,
)
from seamless_communication.models.unity.model import UnitYModel, TransformerFrontend
from fairseq2.models.transformer.frontend import TransformerEmbeddingFrontend
from fairseq2.models.wav2vec2 import Wav2Vec2Frontend
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import (
    ShardingStrategy,
    MixedPrecision
)
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    CPUOffload,
    BackwardPrefetch,
)
from torch.distributed.fsdp.wrap import (
    size_based_auto_wrap_policy,
    transformer_auto_wrap_policy,
    enable_wrap,
    wrap,
)
import functools
import torch.multiprocessing as mp


def setup_logging(rank):
    logger = logging.getLogger("finetune")
    if rank==0:
        logging.basicConfig(
            level=logging.INFO,
            format=f"%(asctime)s %(levelname)s -- %(name)s.{os.getpid()}: %(message)s",
            filename="finetune.log",
            filemode="w"
        )
    else:
        logger.addHandler(logging.NullHandler())  # 다른 프로세스는 로그를 무시

    return logger

def init_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Example finetuning script for M4T models"
    )
    parser.add_argument(
        "--train_dataset",
        type=Path,
        required=True,
        help="Path to manifest with train samples",
    )
    parser.add_argument(
        "--eval_dataset",
        type=Path,
        required=True,
        help="Path to manifest with eval samples",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="seamlessM4T_medium",
        help="Base model name (`seamlessM4T_medium`, `seamlessM4T_large`, 'seamlessM4T_v2_large')",
    )
    parser.add_argument(
        "--save_model_to",
        type=Path,
        required=True,
        help="Path to save best finetuned model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2343,
        help="Randomizer seed value",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for training and evaluation",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help=(
            "Set early termination after `patience` number of evaluations "
            "without eval loss improvements"
        ),
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=10,
        help=("Max number of training epochs"),
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-7,
        help=("Finetuning learning rate"),
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=100,
        help=("Number of steps with linearly increasing learning rate"),
    )
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=50,
        help=("Get eval loss after each `eval_steps` training steps "),
    )
    parser.add_argument(
        "--log_steps",
        type=int,
        default=5,
        help=("Log inner loss after each `log_steps` training steps"),
    )
    parser.add_argument(
        "--max_src_tokens",
        type=int,
        default=7000,
        help=("Maximum number of src_tokens per batch, used to avoid GPU OOM and maximize the effective batch size"),
    )
    parser.add_argument(
        "--mode",
        type=trainer.FinetuneMode,
        choices=list(trainer.FinetuneMode),
        default=trainer.FinetuneMode.SPEECH_TO_TEXT,
        help=(
            "* `SPEECH_TO_SPEECH` -- finetune S2T and T2U parts of the model; "
            "* `TEXT_TO_SPEECH` -- finetune only T2U; "
            "* `SPEECH_TO_TEXT` -- finetune only S2T"
        ),
    )
    parser.add_argument(
        "--freeze_layers",
        nargs="*",
        required=False,
        default=None,
        # TODO: better description
        help=("A list of modules to freeze in the model. If empty, everything will be trained."),
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help=("Device to fine-tune on. See `torch.device`."),
    )
    return parser

my_auto_wrap_policy = functools.partial(
        # transformer_auto_wrap_policy, 
        # transformer_layer_cls = {
        #     TransformerEmbeddingFrontend,
        #     Wav2Vec2Frontend,
        # },
        # recurse=True,
        # TransformerEmbeddingFrontend, UnityEncdoerAdaptor, Wav2Vec2Frontend
        size_based_auto_wrap_policy, min_num_params=30000 # 이수치가 높을수록 메모리는 많이 소요되지만, 학습속도는 빨라짐.
    )

def custom_auto_wrap_policy(module, recurse, **kwargs):
    SPECIAL_CLASSES = (TransformerEmbeddingFrontend, Wav2Vec2Frontend) # StrandardTransformerDecoder
    # 만약 module이 SPECIAL_CLASSES 중 하나의 인스턴스라면 무조건 통째로 wrap
    if isinstance(module, SPECIAL_CLASSES):
        return True
    # 그렇지 않다면, size 기반 auto wrap 정책을 적용합니다.
    # 예: 최소 파라미터 수 20000 이상일 때 wrap
    return size_based_auto_wrap_policy(module, recurse, kwargs['nonwrapped_numel'], min_num_params=20000)


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    rank = int(rank)
    world_size = int(world_size)
    local_rank = int(rank)
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=world_size,
        rank=rank,
    )

    torch.cuda.set_device(local_rank)
    dist.barrier()

def cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()

def fsdp_main(rank, world_size) -> None:
    args = init_parser().parse_args()
    logger = setup_logging(rank)
    setup(rank, world_size) 
    float_dtype = torch.float16 if torch.device(args.device).type != "cpu" else torch.bfloat16
    
    text_tokenizer = load_unity_text_tokenizer(args.model_name)
    unit_tokenizer = load_unity_unit_tokenizer(args.model_name)

    finetune_params = trainer.FinetuneParams(
        model_name=args.model_name,
        finetune_mode=args.mode,
        save_model_path=args.save_model_to,
        device=torch.device(rank),
        float_dtype=float_dtype,
        train_batch_size=args.batch_size,
        eval_batch_size=args.batch_size,
        patience=args.patience,
        max_epochs=args.max_epochs,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        eval_steps=args.eval_steps,
        log_steps=args.log_steps,
    )
    
    logger.info(f"Finetune Params: {finetune_params}")
    
    model = load_unity_model(args.model_name, device=torch.device("cpu"), dtype=torch.float32)
    assert model.target_vocab_info == text_tokenizer.vocab_info
    
    if (
        finetune_params.finetune_mode == trainer.FinetuneMode.SPEECH_TO_TEXT
        and model.t2u_model is not None
    ):
        model.t2u_model = None
        model.text_encoder = None
        model.text_encoder_frontend = None
    
    if model.text_encoder is not None:
        model.text_encoder = None

    wrapped_model = UnitYFinetuneWrapper(
        model=model, mode=finetune_params.finetune_mode, device=torch.device(rank))
    
    fpSixteen = MixedPrecision(
        param_dtype=float_dtype,
        # Gradient communication precision.
        reduce_dtype=float_dtype,
        # Buffer precision.
        buffer_dtype=float_dtype,
    )

    model = FSDP(wrapped_model,
                auto_wrap_policy=my_auto_wrap_policy,#custom_auto_wrap_policy,
                mixed_precision=fpSixteen, #amp 설정. 해당 기능 사용시 학습 내부 amp코드를 전부 제거할것.
                device_id=torch.cuda.current_device(), # 해당 기능 사용시 CPU기반 초기화보다 몇 배 더 빠르게 초기화 가능하고, OOM문제 방지. FSDP 단위 기준으로 모델을 지정된 장치로 이동.
                # cpu_offload=CPUOffload(offload_params=True), # 모델이 너무커서 gpu들에 쪼개서도 안들어갈때 넘치는 부분을 cpu에 보내는것. 파라미터와 gradient만 cpu offload가능.
                # sharding_strategy=ShardingStrategy.SHARD_GRAD_OP # optimizer's state는 shard하지 않으려고 할때
                backward_prefetch = BackwardPrefetch.BACKWARD_PRE, # 메모리는 약 12% 증가하고, 속도는 2~10%감소. 
            )

    # TODO: delete unused params to reduce GPU memory consumption. dataloader
    train_dataloader = dataloader.UnitYDataLoader(
        text_tokenizer=text_tokenizer,
        unit_tokenizer=unit_tokenizer,
        batching_config=dataloader.BatchingConfig(
            batch_size=finetune_params.train_batch_size,
            rank=rank,
            world_size=dist_utils.get_world_size(),
            num_workers=dist_utils.get_world_size(),
            max_audio_length_sec=15.0,
            float_dtype=finetune_params.float_dtype,
        ),
        dataset_manifest_path=args.train_dataset,
        max_src_tokens_per_batch=args.max_src_tokens)
    
    eval_dataloader = dataloader.UnitYDataLoader(
        text_tokenizer=text_tokenizer,
        unit_tokenizer=unit_tokenizer,
        batching_config=dataloader.BatchingConfig(
            batch_size=finetune_params.eval_batch_size,
            rank=rank,
            world_size=dist_utils.get_world_size(),
            max_audio_length_sec=75.0,
            float_dtype=finetune_params.float_dtype,
        ),
        dataset_manifest_path=args.eval_dataset)
    
    finetune = trainer.UnitYFinetune(
        model=model,
        params=finetune_params,
        train_data_loader=train_dataloader,
        eval_data_loader=eval_dataloader,
        freeze_modules=args.freeze_layers,
        )
    
    finetune.run()

    dist.barrier()
    cleanup()

if __name__ == "__main__":
    WORLD_SIZE = torch.cuda.device_count()

    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        # 이미 설정되어 있다면 무시합니다.
        pass

    mp.spawn(fsdp_main,
        args=(WORLD_SIZE,),
        nprocs=WORLD_SIZE
        )
