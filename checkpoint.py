# coding=utf-8
# Copyright 2017-2020 The THUMT Authors
import os
import glob
import torch


def oldest_checkpoint(path):
    names = glob.glob(os.path.join(path, "*.pt"))

    if not names:
        return None

    oldest_counter = 10000000
    checkpoint_name = names[0]

    for name in names:
        counter = name.rstrip(".pt").split("-")[-1]

        if not counter.isdigit():
            continue
        else:
            counter = int(counter)

        if counter < oldest_counter:
            checkpoint_name = name
            oldest_counter = counter

    return checkpoint_name


def latest_checkpoint(path):
    names = glob.glob(os.path.join(path, "*.pt"))

    if not names:
        return None

    latest_counter = 0
    checkpoint_name = names[0]

    for name in names:
        counter = name.rstrip(".pt").split("-")[-1]

        if not counter.isdigit():
            continue
        else:
            counter = int(counter)

        if counter > latest_counter:
            checkpoint_name = name
            latest_counter = counter

    return checkpoint_name


def save(state, path, max_to_keep=None):
    """
    调用该方法，保存 state
    max_to_keep 为最多保存多少个ckpt
    """
    checkpoints = glob.glob(os.path.join(path, "*.pt"))

    if not checkpoints:
        counter = 1
    else:
        checkpoint = latest_checkpoint(path)
        counter = int(checkpoint.rstrip(".pt").split("-")[-1]) + 1

    if max_to_keep and len(checkpoints) >= max_to_keep:
        checkpoint = oldest_checkpoint(path)
        os.remove(checkpoint)

    checkpoint = os.path.join(path, "model-%d.pt" % counter)
    print("Saving checkpoint: %s" % checkpoint)
    torch.save(state, checkpoint)
    
def save_checkpoint(step, epoch, model, optimizer, params):
    """
    存的是 step, epoch, model，optimizer
    可以按照实际情况进行修改
    """
    if dist.get_rank() == 0:
        state = {
            "step": step,
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict()
        }
        utils.save(state, params.output, params.keep_checkpoint_max)
        
def load_checkpoint():
    checkpoint = utils.latest_checkpoint(params.output)

    if args.checkpoint is not None:
        # 超参数传入的 checkpoint
        state = torch.load(args.checkpoint, map_location="cpu")
        model.load_state_dict(state["model"])
        step = params.initial_step
        epoch = 0  # 默认从 0 开始
    elif checkpoint is not None:
        # 默认地址的 checkpoints
        state = torch.load(checkpoint, map_location="cpu")
        step = state["step"]  # 保存以前没训练完的步数
        epoch = state["epoch"]
        model.load_state_dict(state["model"])

        if "optimizer" in state:
            optimizer.load_state_dict(state["optimizer"])
    else:
        step = 0  #
        epoch = 0  # 真实 epoch 数量
