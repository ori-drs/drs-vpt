# Adapted from MapAnything (https://github.com/facebookresearch/map-anything), mapanything/train/training.py and
# mapanything/utils/inference.py (loss_of_one_batch_multi_view).
# Copyright (c) Meta Platforms, Inc. and affiliates. Apache License 2.0, see drs_vpt/_vendor/mapanything/LICENSE.
# Modified for DRS VPT (Oxford Robotics Institute): builds DRSVPT (from the Hub weights, a training checkpoint
# or the config), optionally freezes the image encoder, makes Weights & Biases optional and picks the autocast device
# from the training device.

"""
Training loop for DRS VPT.

References:
DUSt3R: https://github.com/naver/dust3r
"""

import datetime
import json
import math
import os
import pickle
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Sized

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from omegaconf import OmegaConf
from torch.utils.tensorboard import SummaryWriter

import drs_vpt.train.train_tools as train_tools
from drs_vpt.datasets.loaders import get_test_data_loader, get_train_data_loader
from drs_vpt.train import losses
from drs_vpt.train.train_tools import NativeScalerWithGradNormCount as NativeScaler

# Enable TF32 precision if supported (for GPU >= Ampere and PyTorch >= 1.12)
if hasattr(torch.backends.cuda, "matmul") and hasattr(
    torch.backends.cuda.matmul, "allow_tf32"
):
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def build_model(model_args):
    """
    Build DRSVPT for training.

    model_args.pretrained selects the starting weights:
        a Hub repo id or a save_pretrained directory (default: the released weights, fulkast/drs_vpt),
        a training checkpoint (.pth with a "model" state dict),
        or null to start from the config with pretrained DINOv2 and Sonata encoders only.
    """
    from drs_vpt.model import DRSVPT

    pretrained = model_args.pretrained
    if pretrained and not Path(pretrained).is_file():
        print(f"Loading pretrained weights: {pretrained}")
        return DRSVPT.from_pretrained(pretrained, strict=True)

    model_config = OmegaConf.to_container(model_args.model_config, resolve=True)
    model = DRSVPT(
        **model_config,
        torch_hub_force_reload=model_args.torch_hub_force_reload,
        skip_encoder_pretrain=bool(pretrained),
    )
    if pretrained:
        print(f"Loading training checkpoint: {pretrained}")
        state_dict = torch.load(pretrained, map_location="cpu", weights_only=False)["model"]
        print(model.load_state_dict(state_dict, strict=True))
    return model


def train(args):
    """
    Main training function that handles the entire training process.

    This function initializes the distributed training environment, sets up datasets,
    initializes the model, optimizer, and loss functions, and manages the training
    and evaluation loop across multiple epochs.

    In this training, an epoch is just a chunk of the entire dataset.

    Args:
        args: Configuration object containing all training parameters including
              dataset configs, model configs, training parameters, and loss functions.
    """
    # Initialize distributed training if required
    train_tools.init_distributed_mode(args.distributed)
    global_rank = train_tools.get_rank()

    wandb = None
    if args.wandb.enabled and train_tools.is_main_process():
        import wandb

        wandb.init(
            project=args.wandb.project,
            name=args.wandb.run_name or Path(args.output_dir).name,
            config=OmegaConf.to_container(args, resolve=False),
            dir=args.output_dir,
            resume=args.wandb.resume,
            entity=args.wandb.entity,
        )

    # Init output directory and device
    print("output_dir: " + args.output_dir)
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    print("{}".format(args).replace(", ", ",\n"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Fix the seed
    seed = args.train_params.seed + train_tools.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = not args.train_params.disable_cudnn_benchmark

    # Datasets and Dataloaders
    print("Building train dataset {:s}".format(args.dataset.train_dataset))
    data_loader_train = build_dataset(
        dataset=args.dataset.train_dataset,
        num_workers=args.dataset.num_workers,
        test=False,
        max_num_of_imgs_per_gpu=args.train_params.max_num_of_imgs_per_gpu,
    )
    print("Building test dataset {:s}".format(args.dataset.test_dataset))
    test_batch_size = (
        args.train_params.max_num_of_imgs_per_gpu // args.dataset.num_views
    )  # Since we don't have any backward overhead
    data_loader_test = {
        dataset.split("(")[0].split("@")[-1].strip(): build_dataset(
            dataset=dataset,
            num_workers=args.dataset.num_workers,
            test=True,
            batch_size=test_batch_size,
        )
        for dataset in args.dataset.test_dataset.split("+")
        if "(" in dataset
    }

    # Load Model (rank 0 first, so that downloads happen once)
    if global_rank == 0:
        model = build_model(args.model)
    if torch.distributed.is_initialized():
        torch.distributed.barrier()  # Make sure the model is initialized before proceeding
    if global_rank != 0:
        model = build_model(args.model)
    model.to(device)  # Move model to device
    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))

    # Criterion
    print(f">> Creating train criterion = {args.loss.train_criterion}")
    train_criterion = eval(args.loss.train_criterion, vars(losses)).to(device)
    print(
        f">> Creating test criterion = {args.loss.test_criterion or args.loss.train_criterion}"
    )
    test_criterion = eval(
        args.loss.test_criterion or args.loss.train_criterion, vars(losses)
    ).to(device)

    # Freeze the image encoder (DINOv2); the released model was trained this way
    if args.train_params.freeze_image_encoder:
        for name, param in model_without_ddp.named_parameters():
            if name.startswith("encoder."):
                param.requires_grad = False

    # Init model for DDP training
    if args.distributed.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.distributed.gpu],
            find_unused_parameters=True,
            static_graph=False,
        )
        model_without_ddp = model.module

    # Optimizer and loss scaler for gradient accumulation
    # Following timm: set wd as 0 for bias and norm layers
    param_groups, param_groups_name_to_idx_map, param_groups_idx_to_name_map = (
        train_tools.get_parameter_groups(
            model_without_ddp,
            args.train_params.lr,
            args.train_params.weight_decay,
            submodule_configs=args.train_params.submodule_configs,
            warn_not_in_submodule=args.train_params.warn_not_in_submodule,
        )
    )
    optimizer = torch.optim.AdamW(
        param_groups, lr=args.train_params.lr, betas=(0.9, 0.95)
    )
    print(optimizer)
    loss_scaler = NativeScaler()

    def write_log_stats(epoch, train_stats, test_stats):
        """
        Writes training and testing statistics to log files and TensorBoard.

        Args:
            epoch: Current epoch number.
            train_stats: Dictionary containing training metrics.
            test_stats: Dictionary containing testing metrics for each test dataset.
        """
        if train_tools.is_main_process():
            if log_writer is not None:
                log_writer.flush()

            log_stats = dict(
                epoch=epoch, **{f"train_{k}": v for k, v in train_stats.items()}
            )
            for test_name in data_loader_test:
                if test_name not in test_stats:
                    continue
                log_stats.update(
                    {test_name + "_" + k: v for k, v in test_stats[test_name].items()}
                )

            with open(
                os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8"
            ) as f:
                f.write(json.dumps(log_stats) + "\n")

    def save_model(epoch, fname, best_so_far):
        """
        Saves model checkpoint to disk.

        Args:
            epoch: Current epoch number.
            fname: Filename or identifier for the checkpoint.
            best_so_far: Best validation metric achieved so far.
        """
        train_tools.save_model(
            args=args,
            model_without_ddp=model_without_ddp,
            optimizer=optimizer,
            loss_scaler=loss_scaler,
            epoch=epoch,
            fname=fname,
            best_so_far=best_so_far,
        )

    # Resume from a checkpoint if needed
    last_ckpt_fname = os.path.join(args.output_dir, "checkpoint-last.pth")
    if args.train_params.resume and os.path.isfile(last_ckpt_fname):
        args.train_params.resume_ckpt = last_ckpt_fname
    else:
        args.train_params.resume_ckpt = None
    best_so_far = train_tools.load_model(
        train_args=args.train_params,
        model_without_ddp=model_without_ddp,
        optimizer=optimizer,
        loss_scaler=loss_scaler,
    )
    if best_so_far is None:
        best_so_far = float("inf")

    if global_rank == 0 and args.output_dir is not None:
        log_writer = SummaryWriter(log_dir=args.output_dir)
    else:
        log_writer = None

    print(f"Start training for {args.train_params.epochs} epochs")
    start_time = time.time()
    train_stats = test_stats = {}
    for epoch in range(args.train_params.start_epoch, args.train_params.epochs + 1):
        # Save immediately the last checkpoint
        if epoch > args.train_params.start_epoch:
            if (
                args.train_params.save_freq
                and epoch % args.train_params.save_freq == 0
                or epoch == args.train_params.epochs
            ):
                save_model(epoch - 1, "last", best_so_far)

        # Test on multiple datasets
        new_best = False
        test_stats = {}
        if (
            args.train_params.eval_freq > 0
            and epoch % args.train_params.eval_freq == 0
            and epoch > args.train_params.start_epoch
        ):
            for test_name, testset in data_loader_test.items():
                print(f"Testing on {test_name} ...")
                stats = test_one_epoch(
                    model,
                    test_criterion,
                    testset,
                    device,
                    epoch,
                    log_writer=log_writer,
                    args=args,
                    prefix=test_name,
                )
                test_stats[test_name] = stats
                if wandb is not None:
                    wandb.log(
                        {
                            f"eval/{test_name}/{k}": v
                            for k, v in stats.items()
                            if "_med" in k and "view" not in k
                        },
                        step=epoch * 1000,
                    )

            # Calculate average test loss median
            avg_test_loss_med = np.mean(
                [stats["loss_med"] for stats in test_stats.values()]
            )
            test_stats["Average Test Loss Median"] = avg_test_loss_med
            # Save best
            if avg_test_loss_med < best_so_far:
                best_so_far = avg_test_loss_med
                new_best = True

        # Save more stuff
        write_log_stats(epoch, train_stats, test_stats)

        if epoch > args.train_params.start_epoch:
            if args.train_params.keep_freq and epoch % args.train_params.keep_freq == 0:
                save_model(epoch - 1, str(epoch), best_so_far)
            if new_best:
                save_model(epoch - 1, "best", best_so_far)
        if epoch >= args.train_params.epochs:
            break  # exit after writing last test to disk

        # Train
        train_stats = train_one_epoch(
            model,
            train_criterion,
            data_loader_train,
            optimizer,
            device,
            epoch,
            loss_scaler,
            log_writer=log_writer,
            args=args,
            param_groups_name_to_idx_map=param_groups_name_to_idx_map,
            param_groups_idx_to_name_map=param_groups_idx_to_name_map,
            model_without_ddp=model_without_ddp,
            wandb=wandb,
        )

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))

    save_final_model(
        args, args.train_params.epochs, model_without_ddp, best_so_far=best_so_far
    )
    if wandb is not None:
        wandb.finish()


def save_final_model(args, epoch, model_without_ddp, best_so_far=None):
    """
    Saves the final model checkpoint after training completion.

    Args:
        args: Configuration object containing output directory information.
        epoch: Current epoch number.
        model_without_ddp: Model state dictionary or model instance without DistributedDataParallel wrapper.
        best_so_far: Optional; Best validation metric achieved during training.
    """
    output_dir = Path(args.output_dir)
    checkpoint_path = output_dir / "checkpoint-final.pth"
    to_save = {
        "args": args,
        "model": model_without_ddp
        if isinstance(model_without_ddp, dict)
        else model_without_ddp.cpu().state_dict(),
        "epoch": epoch,
    }
    if best_so_far is not None:
        to_save["best_so_far"] = best_so_far
    print(f">> Saving model to {checkpoint_path} ...")
    train_tools.save_on_master(to_save, checkpoint_path)


def build_dataset(
    dataset, num_workers, test, batch_size=None, max_num_of_imgs_per_gpu=None
):
    """
    Builds data loaders for training or testing.

    Args:
        dataset: Dataset specification string.
        num_workers: Number of worker processes for data loading.
        test: Boolean flag indicating whether this is a test dataset.
        batch_size: Number of samples per batch. Defaults to None. Used only for testing.
        max_num_of_imgs_per_gpu: Maximum number of images per GPU. Defaults to None. Used only for training.

    Returns:
        DataLoader: PyTorch DataLoader configured for the specified dataset.
    """
    split = ["Train", "Test"][test]
    print(f"Building {split} Data loader for dataset: ", dataset)
    if test:
        assert batch_size is not None, (
            "batch_size must be specified for testing dataloader"
        )
        loader = get_test_data_loader(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_mem=True,
            shuffle=False,
            drop_last=False,
        )
    else:
        assert max_num_of_imgs_per_gpu is not None, (
            "max_num_of_imgs_per_gpu must be specified for training dataloader"
        )
        loader = get_train_data_loader(
            dataset=dataset,
            max_num_of_imgs_per_gpu=max_num_of_imgs_per_gpu,
            num_workers=num_workers,
            pin_mem=True,
            shuffle=True,
            drop_last=True,
        )

    print(f"{split} dataset length: ", len(loader))
    return loader


def loss_of_one_batch_multi_view(
    batch,
    model,
    criterion,
    device,
    use_amp=False,
    amp_dtype="bf16",
    ret=None,
    ignore_keys=None,
):
    """
    Calculate loss for a batch with multiple views.

    Args:
        batch (dict): {"image_views": list of view dicts, "cloud": the collated cloud input}.
        model (torch.nn.Module): Model to run inference with.
        criterion (callable, optional): Loss function to compute the loss.
        device (torch.device): Device to run the computation on.
        use_amp (bool, optional): Whether to use automatic mixed precision (CUDA only). Defaults to False.
        amp_dtype (str, optional): Floating point type to use for automatic mixed precision. Options: ["fp32", "fp16", "bf16"]. Defaults to "bf16".
        ret (str, optional): If provided, return only the specified key from the result dictionary.
        ignore_keys (set, optional): Set of keys to ignore when moving tensors to device.
                                   Defaults to {"depthmap", "dataset", "label", "instance",
                                   "idx", "true_shape", "rng", "data_norm_type"}.

    Returns:
        dict or Any: If ret is None, returns a dictionary containing views, predictions, and loss.
                     Otherwise, returns the value associated with the ret key.
    """
    # Move necessary tensors to device
    if ignore_keys is None:
        ignore_keys = {
            "depthmap",
            "dataset",
            "label",
            "instance",
            "idx",
            "true_shape",
            "rng",
            "data_norm_type",
        }
    for view in batch["image_views"]:
        for name in view.keys():
            if name in ignore_keys:
                continue
            view[name] = view[name].to(device, non_blocking=True)
    cloud = batch["cloud"]
    for name, value in cloud.items():
        if torch.is_tensor(value):
            cloud[name] = value.to(device, non_blocking=True)

    # Determine the mixed precision floating point type
    use_amp = bool(use_amp) and device.type == "cuda"
    if use_amp:
        if amp_dtype == "fp16":
            amp_dtype = torch.float16
        elif amp_dtype == "bf16":
            if torch.cuda.is_bf16_supported():
                amp_dtype = torch.bfloat16
            else:
                warnings.warn(
                    "bf16 is not supported on this device. Using fp16 instead."
                )
                amp_dtype = torch.float16
        elif amp_dtype == "fp32":
            amp_dtype = torch.float32
    else:
        amp_dtype = torch.float32

    # Run model and compute loss
    with torch.autocast(device.type, enabled=use_amp, dtype=amp_dtype):
        preds = model(batch["image_views"], batch["cloud"])
        with torch.autocast(device.type, enabled=False):
            loss = criterion(batch, preds) if criterion is not None else None

    result = {f"view{i + 1}": view for i, view in enumerate(batch["image_views"])}
    result.update({f"pred{i + 1}": pred for i, pred in enumerate(preds["image_views"])})
    result.update(cloud=batch["cloud"], pred_cloud=preds["cloud"])
    result["loss"] = loss

    return result[ret] if ret else result


@torch.no_grad()
def dump_top_named_grad_norms(model, topk=20, print_fn=print):
    rows = []
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        g = p.grad
        if g.is_sparse:
            g = g.coalesce().values()
        gn = float(g.norm(2).item())
        gmax = float(g.abs().max().item())
        rows.append((gn, gmax, name, tuple(p.shape), str(p.dtype), str(p.device)))

    rows.sort(key=lambda x: x[0], reverse=True)
    print_fn(f"[grad-debug] showing top {min(topk, len(rows))}/{len(rows)} params by grad L2 norm")
    for gn, gmax, name, shape, dtype, device in rows[:topk]:
        print_fn(f"  {gn:12.5e}  max|g|={gmax:12.5e}  {name}  shape={shape}  {dtype}  {device}")


def train_one_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: Sized,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler,
    args,
    log_writer=None,
    param_groups_name_to_idx_map=None,
    param_groups_idx_to_name_map=None,
    model_without_ddp=None,
    wandb=None,
):
    """
    Trains the model for one epoch.
    Epoch is just a chunk of the entire dataset.

    This function handles the training loop for a single epoch, including forward/backward passes,
    gradient accumulation, learning rate scheduling, and logging metrics.

    Args:
        model: The neural network model to train.
        criterion: Loss function to optimize.
        data_loader: DataLoader providing the training data.
        optimizer: Optimizer for updating model parameters.
        device: Device to run training on (CPU or GPU).
        epoch: Current epoch number.
        loss_scaler: Scaler for gradient accumulation and mixed precision training.
        args: Configuration object containing training parameters.
        log_writer: Optional; TensorBoard SummaryWriter for logging.
        param_groups_name_to_idx_map: Mapping from parameter group names to indices.
        param_groups_idx_to_name_map: Mapping from parameter group indices to names.
        model_without_ddp: Model without DistributedDataParallel wrapper for debugging.
        wandb: Optional; the wandb module when Weights & Biases logging is enabled.

    Returns:
        dict: Dictionary containing training metrics averaged over the epoch.
    """
    model.train(True)
    metric_logger = train_tools.MetricLogger(delimiter="  ")
    for submodule_name in param_groups_name_to_idx_map:
        lr_name = f"lr_{submodule_name}" if submodule_name != "default" else "lr"
        metric_logger.add_meter(
            lr_name, train_tools.SmoothedValue(window_size=1, fmt="{value:.6f}")
        )
    header = "Epoch: [{}]".format(epoch)
    accum_iter = args.train_params.accum_iter

    if log_writer is not None:
        print("log_dir: {}".format(log_writer.log_dir))

    if hasattr(data_loader, "dataset") and hasattr(data_loader.dataset, "set_epoch"):
        data_loader.dataset.set_epoch(epoch)
    if hasattr(data_loader, "sampler") and hasattr(data_loader.sampler, "set_epoch"):
        data_loader.sampler.set_epoch(epoch)
    if hasattr(data_loader, "batch_sampler") and hasattr(
        data_loader.batch_sampler, "set_epoch"
    ):
        data_loader.batch_sampler.set_epoch(epoch)

    optimizer.zero_grad()

    for data_iter_step, batch in enumerate(
        metric_logger.log_every(data_loader, args.train_params.print_freq, header)
    ):
        n_views = len(batch["image_views"])
        epoch_f = epoch + data_iter_step / len(data_loader)

        # We use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            train_tools.adjust_learning_rate(
                optimizer,
                epoch_f,
                args.train_params,
                param_groups_idx_to_name_map,
                args.train_params.submodule_configs,
            )

        loss_tuple = loss_of_one_batch_multi_view(
            batch,
            model,
            criterion,
            device,
            use_amp=bool(args.train_params.amp),
            amp_dtype=args.train_params.amp_dtype,
            ret="loss",
        )
        loss, loss_details = loss_tuple  # criterion returns two values
        if n_views > 2:
            loss = loss * (
                2 / n_views
            )  # scale the loss relative to the number of views (base is 2 views)
        loss_value = float(loss)

        if not math.isfinite(loss_value) or (loss_value > 1000):
            print("Loss is {}, stopping training".format(loss_value), force=True)
            print(f"Loss Details: {loss_details}", force=True)
            print(f"Epoch: {epoch}, Data Iteration: {data_iter_step}", force=True)
            # Save the current batch to the output folder for further inspection
            parts = {f"view_{i}": view for i, view in enumerate(batch["image_views"])} | {"cloud": batch["cloud"]}
            for part_name, part in parts.items():
                part_cpu = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in part.items()}
                with open(os.path.join(args.output_dir, f"batch_{part_name}.pkl"), "wb") as f:
                    pickle.dump(part_cpu, f)
            # Save the model to the output folder for further inspection
            checkpoint_debug_path = os.path.join(
                args.output_dir, "checkpoint-debug.pth"
            )
            to_save_debug = {
                "args": args,
                "model": (
                    model_without_ddp
                    if isinstance(model_without_ddp, dict)
                    else model_without_ddp.cpu().state_dict()
                ),
                "epoch": epoch,
                "data_iter_step": data_iter_step,
            }
            torch.save(to_save_debug, checkpoint_debug_path)
            print(f"Saved debugging material to {args.output_dir}", force=True)
            sys.exit(1)

        # Scale the loss by the number of gradient accumulation iterations
        loss /= accum_iter

        # Compute the scaled gradients (also clip the gradients to max norm of 1)
        gradient_norm = loss_scaler(
            loss,
            optimizer,
            parameters=model.parameters(),
            update_grad=(data_iter_step + 1) % accum_iter == 0,
            clip_grad=1.0,
        )

        if gradient_norm is not None and gradient_norm > 100_000_000:
            print(f"Gradient norm: {gradient_norm}")
            dump_top_named_grad_norms(model, topk=60)
        # Zero out the gradients to prepare for the next iteration of gradient descent
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        del loss
        del batch

        metric_logger.update(epoch=epoch_f)
        for submodule_name in param_groups_name_to_idx_map:
            lr_name = f"lr_{submodule_name}" if submodule_name != "default" else "lr"
            log_lr = optimizer.param_groups[
                param_groups_name_to_idx_map[submodule_name][0]
            ]["lr"]
            metric_logger.meters[lr_name].update(log_lr)
        metric_logger.update(loss=loss_value, **loss_details)

        if (data_iter_step + 1) % accum_iter == 0 and (
            (data_iter_step + 1) % (accum_iter * args.train_params.print_freq)
        ) == 0:
            loss_value_reduce = train_tools.all_reduce_mean(loss_value)
            epoch_1000x = int(epoch_f * 1000)

            if train_tools.is_main_process():
                metrics = {
                    "train/loss": loss_value_reduce,
                    "train/grad_norm": float(gradient_norm or 0.0),
                    "train/iter": epoch_1000x,
                }
                for submodule_name in param_groups_name_to_idx_map:
                    lr_name = (
                        f"train/lr_{submodule_name}"
                        if submodule_name != "default"
                        else "train/lr"
                    )
                    log_lr = optimizer.param_groups[
                        param_groups_name_to_idx_map[submodule_name][0]
                    ]["lr"]
                    metrics[lr_name] = log_lr
                for name, val in loss_details.items():
                    metrics[f"train/{name}"] = val

                # Log to TensorBoard
                if log_writer is not None:
                    for k, v in metrics.items():
                        log_writer.add_scalar(k, v, epoch_1000x)

                # Log to Weights & Biases
                if wandb is not None:
                    wandb.log(metrics, step=epoch_1000x)

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def test_one_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: Sized,
    device: torch.device,
    epoch: int,
    args,
    log_writer=None,
    prefix="test",
):
    """
    Evaluates the model on a test dataset for one epoch.
    Epoch is just a chunk of the entire dataset.

    This function runs evaluation on the test dataset without computing gradients,
    and collects metrics for model performance assessment.

    Args:
        model: The neural network model to evaluate.
        criterion: Loss function for evaluation.
        data_loader: DataLoader providing the test data.
        device: Device to run evaluation on (CPU or GPU).
        epoch: Current epoch number.
        args: Configuration object containing evaluation parameters.
        log_writer: Optional; TensorBoard SummaryWriter for logging.
        prefix: String prefix for logging metrics.

    Returns:
        dict: Dictionary containing evaluation metrics (average and median values).
    """
    model.eval()
    metric_logger = train_tools.MetricLogger(delimiter="  ")
    metric_logger.meters = defaultdict(
        lambda: train_tools.SmoothedValue(window_size=9**9)
    )
    header = "Test Epoch: [{}]".format(epoch)

    if log_writer is not None:
        print("log_dir: {}".format(log_writer.log_dir))

    if args.train_params.freeze_val_samples_across_all_epochs:
        dataloader_epoch = 0
    else:
        dataloader_epoch = epoch
    if hasattr(data_loader, "dataset") and hasattr(data_loader.dataset, "set_epoch"):
        data_loader.dataset.set_epoch(dataloader_epoch)
    if hasattr(data_loader, "sampler") and hasattr(data_loader.sampler, "set_epoch"):
        data_loader.sampler.set_epoch(dataloader_epoch)
    if hasattr(data_loader, "batch_sampler") and hasattr(
        data_loader.batch_sampler, "set_epoch"
    ):
        data_loader.batch_sampler.set_epoch(dataloader_epoch)

    for _, batch in enumerate(
        metric_logger.log_every(data_loader, args.train_params.print_freq, header)
    ):
        n_views = len(batch["image_views"])
        loss_tuple = loss_of_one_batch_multi_view(
            batch,
            model,
            criterion,
            device,
            use_amp=bool(args.train_params.amp),
            amp_dtype=args.train_params.amp_dtype,
            ret="loss",
        )
        loss_value, loss_details = loss_tuple  # criterion returns two values
        if n_views > 2:
            loss_value = loss_value * (
                2 / n_views
            )  # scale the loss relative to the number of views (base is 2 views)
        metric_logger.update(loss=float(loss_value), **loss_details)

    aggs = [("avg", "global_avg"), ("med", "median")]
    results = {
        f"{k}_{tag}": getattr(meter, attr)
        for k, meter in metric_logger.meters.items()
        for tag, attr in aggs
    }

    if log_writer is not None:
        for name, val in results.items():
            log_writer.add_scalar(prefix + "_" + name, val, 1000 * epoch)

    return results
