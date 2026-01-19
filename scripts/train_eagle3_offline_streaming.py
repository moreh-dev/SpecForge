import argparse
import hashlib
import math
import os
import time
from collections import defaultdict

import torch
import torch.distributed as dist
from accelerate.utils import set_seed
from datasets import load_dataset
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from tqdm import tqdm
from transformers import AutoTokenizer

from specforge import AutoDraftModelConfig, AutoEagle3DraftModel, OfflineEagle3Model
from specforge.data import (
    build_eagle3_dataset,
    build_offline_eagle3_dataset,
    generate_vocab_mapping_file,
    prepare_dp_dataloaders,
)
from specforge.distributed import (
    destroy_distributed,
    get_dp_device_mesh,
    get_dp_group,
    init_distributed,
)
from specforge.modeling.target.target_head import TargetHead
from specforge.optimizer import BF16Optimizer
from specforge.tracker import create_tracker, get_tracker_class
from specforge.utils import (
    create_draft_config_from_target,
    get_full_optimizer_state,
    get_last_checkpoint,
    print_on_rank0,
    print_with_rank,
    rank_0_priority,
    shard_optimizer_state_with_dtensor,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train Eagle3 with offline data")

    # add model-related arguments
    parser.add_argument("--target-model-path", type=str, required=True)
    parser.add_argument(
        "--draft-model-config",
        type=str,
        required=False,
        help="Draft model config path. If not provided, will auto-generate from target model.",
    )
    parser.add_argument(
        "--embedding-key",
        type=str,
        default="model.embed_tokens.weight",
        help="The key of the embedding weight to load from the target model",
    )
    parser.add_argument(
        "--lm-head-key",
        type=str,
        default="lm_head.weight",
        help="The key of the lm head weight to load from the target model",
    )

    # add training-related arguments
    # parser.add_argument("--train-data-path", type=str, required=True)
    parser.add_argument("--train-hidden-states-path", type=str, required=True)
    parser.add_argument("--eval-data-path", type=str, default=None)
    parser.add_argument("--eval-hidden-states-path", type=str, default=None)
    parser.add_argument("--baseline-dir", type=str, default=None)
    parser.add_argument("--num-epochs", type=int, default=10)
    parser.add_argument("--draft-global-batch-size", type=int, default=16)
    parser.add_argument("--draft-micro-batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--warmup-ratio", type=float, default=0.015)
    parser.add_argument(
        "--total-steps",
        type=int,
        default=1e12,
        help="Total training steps. If not provided, will be calculated as num_epochs * steps_per_epoch",
    )
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument(
        "--log-steps", type=int, default=50, help="Log training metrics every N steps"
    )
    parser.add_argument(
        "--ttt-length",
        type=int,
        default=7,
        help="The length for Test-Time Training (TTT).",
    )
    parser.add_argument("--draft-attention-backend", type=str, default="flex_attention")
    # data processing type
    parser.add_argument("--chat-template", type=str, default="llama3")
    parser.add_argument(
        "--is-preformatted",
        action="store_true",
        help="Whether the input data is preformatted text with the chat template already applied to the conversation messages.",
    )

    # distributed training
    parser.add_argument("--tp-size", type=int, default=1)

    # other args
    parser.add_argument("--cache-key", type=str, default=None)
    parser.add_argument("--cache-dir", type=str, default="./cache")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--eval-interval", type=int, default=1)
    parser.add_argument("--save-interval", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--dist-timeout",
        type=int,
        default=20,
        help="Timeout for collective communication in minutes",
    )
    # resume
    parser.add_argument("--resume", action="store_true")

    # report backend
    parser.add_argument(
        "--report-to",
        type=str,
        default="none",
        choices=["wandb", "tensorboard", "swanlab", "mlflow", "none"],
        help="The integration to report results and logs to.",
    )
    # wandb-specific args
    parser.add_argument(
        "--wandb-project", type=str, default=None, help="The project name for W&B."
    )
    parser.add_argument(
        "--wandb-name", type=str, default=None, help="The run name for W&B."
    )
    parser.add_argument("--wandb-key", type=str, default=None, help="W&B API key.")
    # add swanlab-specific args ---
    parser.add_argument(
        "--swanlab-project",
        type=str,
        default=None,
        help="The project name for swanlab.",
    )
    parser.add_argument(
        "--swanlab-name",
        type=str,
        default=None,
        help="The experiment name for swanlab.",
    )
    parser.add_argument(
        "--swanlab-key",
        type=str,
        default=None,
        help="The API key for swanlab non-interactive login.",
    )
    # mlflow-specific args
    parser.add_argument(
        "--mlflow-tracking-uri",
        type=str,
        default=None,
        help="The MLflow tracking URI. If not set, uses MLFLOW_TRACKING_URI environment variable or defaults to local './mlruns'.",
    )
    parser.add_argument(
        "--mlflow-experiment-name",
        type=str,
        default=None,
        help="The MLflow experiment name. If not set, uses MLFLOW_EXPERIMENT_NAME environment variable.",
    )
    parser.add_argument(
        "--mlflow-run-name",
        type=str,
        default=None,
        help="The MLflow run name. If not set, MLflow will auto-generate one.",
    )

    parser.add_argument("--build-dataset-num-proc", type=int, default=8)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--profile-start-step", type=int, default=30)
    parser.add_argument("--profile-num-steps", type=int, default=4)
    parser.add_argument("--profile-record-shapes", action="store_true")

    args = parser.parse_args()

    return parser, args


def main():
    # initialize
    parser, args = parse_args()
    set_seed(args.seed)
    init_distributed(timeout=args.dist_timeout, tp_size=args.tp_size)
    print("Initialized distributed environment")
    args.dp_size = dist.get_world_size() // args.tp_size
    args.draft_accumulation_steps = (
        args.draft_global_batch_size // args.dp_size // args.draft_micro_batch_size
    )
    assert (
        args.draft_accumulation_steps * args.draft_micro_batch_size * args.dp_size
        == args.draft_global_batch_size
    ), f"draft_global_batch_size={args.draft_global_batch_size} must be divisible by dp_size={args.dp_size} and micro_batch_size={args.draft_micro_batch_size}"
    print(
        f"draft_accumulation_steps={args.draft_global_batch_size} // {args.dp_size} // {args.draft_micro_batch_size}={args.draft_accumulation_steps}"
    )

    # Validate report backend arguments
    tracker_class = get_tracker_class(args.report_to)
    if tracker_class:
        tracker_class.validate_args(parser, args)
    else:
        parser.error(f"Unknown tracker: {args.report_to}")

    tracker = create_tracker(args, args.output_dir)

    # detecting last ckpt for draft model
    draft_model_last_checkpoint = None
    if args.baseline_dir is not None:
        if os.path.isdir(args.baseline_dir):
            draft_model_last_checkpoint = args.baseline_dir
            print(
                f"Finetuning from baseline model: {draft_model_last_checkpoint}"
            )
            args.draft_model_config = os.path.join(args.baseline_dir, "config.json")
        else:
            raise ValueError(
                f"Provided baseline-dir {args.baseline_dir} is not a valid directory."
            )
    if args.resume and os.path.isdir(args.output_dir):
        print(args.output_dir)
        draft_model_last_checkpoint = get_last_checkpoint(args.output_dir)
        print(f"Last checkpoint detected: {draft_model_last_checkpoint}")

    # build target and draft model
    target_head = TargetHead(args.target_model_path)
    target_head.load_weights(
        model_path=args.target_model_path,
        lm_head_key=args.lm_head_key,
        cache_dir=args.cache_dir,
    )
    target_head.freeze_weights()
    target_head = target_head.eval().cuda().to(torch.bfloat16)
    print("Initialized target head")

    # Handle draft model config
    if args.draft_model_config is None:
        # Auto-generate and save config file
        auto_config_path = create_draft_config_from_target(
            target_model_path=args.target_model_path, cache_dir=args.cache_dir
        )
        draft_model_config = AutoDraftModelConfig.from_file(auto_config_path)
    else:
        # Use provided config file
        draft_model_config = AutoDraftModelConfig.from_file(args.draft_model_config)

    if draft_model_last_checkpoint:
        draft_model = (
            AutoEagle3DraftModel.from_pretrained(
                draft_model_last_checkpoint,
                attention_backend=args.draft_attention_backend,
            )
            .cuda()
            .to(torch.bfloat16)
        )
    else:
        draft_model = (
            AutoEagle3DraftModel.from_config(
                draft_model_config, attention_backend=args.draft_attention_backend
            )
            .cuda()
            .to(torch.bfloat16)
        )
    


    draft_model.load_embedding(args.target_model_path, embedding_key=args.embedding_key)
    draft_model.freeze_embedding()
    print("Initialized draft model")

    # build dataloaders
    tokenizer = AutoTokenizer.from_pretrained(args.target_model_path)
    eagle3_model = OfflineEagle3Model(
        target_head=target_head,
        draft_model=draft_model,
        length=args.ttt_length,
        attention_backend=args.draft_attention_backend,
    )
    mp_policy = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16, reduce_dtype=torch.float32
    )
    fsdp_config = {"mesh": get_dp_device_mesh(), "mp_policy": mp_policy}
    fully_shard(eagle3_model, **fsdp_config)
    optimizer = BF16Optimizer(
        eagle3_model,
        lr=args.learning_rate,
        max_grad_norm=args.max_grad_norm,
        warmup_ratio=0.0,
        total_steps=args.total_steps,
        streaming=True
    )
    print("Initialized optimizer and scheduler")
    

    last_time = time.time()

    def get_latest_hidden_state_paths(hidden_states_dir, max_count=4096):
        # chunk 디렉토리/파일 기준 최신순 정렬 후 max_count만 반환
        all_chunks = []
        for entry in os.listdir(hidden_states_dir):
            path = os.path.join(hidden_states_dir, entry)
            if entry.startswith("chunk_") and os.path.isdir(path):
                all_chunks.append((os.path.getmtime(path), path))
            elif entry.endswith(".ckpt") and os.path.isfile(path):
                all_chunks.append((os.path.getmtime(path), path))
        all_chunks.sort(reverse=True)
        return [p for _, p in all_chunks[:max_count]]

    print("[Streaming Mode] Infinite training loop, always using latest 4096 samples.")
    global_step = 0
    while True:
        latest_paths = get_latest_hidden_state_paths(args.train_hidden_states_path, max_count=4096)
        if not latest_paths:
            print("No hidden state data found. Waiting...")
            time.sleep(10)
            continue
        print_on_rank0(f"Found {len(latest_paths)} latest hidden state chunks/files.")
        # 최신 데이터로 dataset/dataloader 생성
        with rank_0_priority():
            train_eagle3_dataset = build_offline_eagle3_dataset(
                hidden_states_list=latest_paths,
                max_len=args.max_length,
            )
        train_dataloader = prepare_dp_dataloaders(
            train_eagle3_dataset,
            args.draft_micro_batch_size,
            num_workers=4,
            shuffle=True,
            process_group=get_dp_group(),
            pin_memory=True,
        )
        print_on_rank0("[Streaming] Initialized train dataloader with latest data.")
        for i in range(args.num_epochs):
            draft_model.train()
            epoch_acces = [[] for _ in range(eagle3_model.length)]
            epoch_plosses = [[] for _ in range(eagle3_model.length)]
            if dist.get_rank() == 0:
                progress_bar = tqdm(
                    train_dataloader, desc=f"Training Step {global_step}", leave=True
                )
            else:
                progress_bar = train_dataloader
            
            batch_index = 0
            log_dict = defaultdict(float)
            for data in progress_bar:
                batch_index += 1
                plosses, _, acces = eagle3_model(
                    input_ids=data["input_ids"].cuda(),
                    attention_mask=data["attention_mask"].cuda(),
                    loss_mask=data["loss_mask"].unsqueeze(-1).cuda(),
                    hidden_states=data["hidden_state"].cuda(),
                    target=data["target"].cuda(),
                )
                acces = torch.stack(acces).cpu().tolist()
                ploss_weight = [0.8**i for i in range(len(plosses))]
                ploss = (
                    sum([ploss_weight[i] * plosses[i] for i in range(len(plosses))])
                    / args.draft_accumulation_steps
                )
                ploss.backward()
                log_dict["train/lr"] = optimizer.get_learning_rate()
                for i in range(len(plosses)):
                    log_dict[f"train/ploss_{i}"] += (
                        plosses[i].item() / args.draft_accumulation_steps
                    )
                for i in range(len(acces)):
                    log_dict[f"train/acc_{i}"] += acces[i] / args.draft_accumulation_steps
                if batch_index % args.draft_accumulation_steps == 0:
                    optimizer.step()
                    global_step += 1
                    if global_step % args.log_steps == 0:
                        tracker.log(log_dict, step=global_step)
                    log_dict = defaultdict(float)
                epoch_acces = [epoch_acces[i] + [acces[i]] for i in range(len(acces))]
                epoch_plosses = [
                    epoch_plosses[i] + [plosses[i].item()] for i in range(len(plosses))
                ]
                if dist.get_rank() == 0:
                    avg_loss = sum(pl.item() for pl in plosses) / len(plosses)
                    avg_acc = sum(acces) / len(acces)
                    progress_bar.set_postfix(
                        {"loss": f"{avg_loss:.2f}", "acc": f"{avg_acc:.2f}"}
                    )

            train_epoch_logdict = {}
            for i in range(len(epoch_acces)):
                acc_i = torch.tensor(epoch_acces[i]).cuda().mean()
                dist.all_reduce(acc_i)
                acc_i = (acc_i / dist.get_world_size()).item()
                train_epoch_logdict[f"train/epoch_acc_{i}"] = acc_i
                print_on_rank0(
                    f"Train Step [{global_step}], position {i},  Acc: {acc_i:.2f}"
                )
            for i in range(len(epoch_plosses)):
                loss_i = torch.tensor(epoch_plosses[i]).cuda().mean()
                dist.all_reduce(loss_i)
                loss_i = (loss_i / dist.get_world_size()).item()
                train_epoch_logdict[f"train/epoch_ploss_{i}"] = loss_i
                print_on_rank0(
                    f"Train Step [{global_step}], position {i}, pLoss: {loss_i:.2f}"
                )
                print_on_rank0("[Streaming] One epoch finished. Checking for new data...")
                time.sleep(5)
            tracker.log(train_epoch_logdict, step=global_step)
            
            if global_step % args.save_interval == 0:
                # Save the model
                epoch_output_dir = os.path.join(args.output_dir, f"global_step_{global_step}")

                if dist.get_rank() == 0:
                    os.makedirs(epoch_output_dir, exist_ok=True)
                dist.barrier()

                model_state_dict = eagle3_model.state_dict()

                state_to_save = {
                    "global_step": global_step,
                    "args": args,
                }

                optimizer_state_dict = optimizer.state_dict()
                optimizer_state_dict["optimizer_state_dict"] = get_full_optimizer_state(
                    optimizer_state_dict["optimizer_state_dict"]
                )

                state_to_save.update(optimizer_state_dict)

                draft_model_state_dict = {
                    k.replace("draft_model.", ""): (
                        v.full_tensor()
                        if isinstance(v, torch.distributed.tensor.DTensor)
                        else v
                    )
                    for k, v in model_state_dict.items()
                    if "draft_model." in k and "embed" not in k.lower()
                }

                if dist.get_rank() == 0:
                    torch.save(
                        state_to_save,
                        os.path.join(epoch_output_dir, "training_state.pt"),
                    )
                    print_on_rank0(
                        f"Saved full training state to {epoch_output_dir}/training_state.pt"
                    )
                    draft_model.save_pretrained(
                        epoch_output_dir,
                        state_dict=draft_model_state_dict,
                    )
                    print_on_rank0(f"Saved model configuration to {epoch_output_dir}")
                dist.barrier()

    try:
        # Keep training loop running
        pass
    except KeyboardInterrupt:
        print("KeyboardInterrupt received. Gracefully shutting down...")
    finally:
        # Close the tracker at the end of training
        tracker.close()
        destroy_distributed()
    return


if __name__ == "__main__":
    main()
