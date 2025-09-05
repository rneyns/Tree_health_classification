"""
training.py

This module contains functions related to the training and validation of the model.
It handles the training loop, loss computation, gradient calculation, and evaluation.

Functions:
    - train_epoch: Performs one training epoch, including forward pass, loss calculation,
      backward pass, and optimization step.
    - valid: Evaluates the model on a validation or test set and returns metrics like accuracy and loss.

Author: Robbe Neyns
Created: Tue Sep 19 15:07:11 2023
"""


# training.py - for training and evaluation logic
from utils_.utils import setup_seed
from augmentations import embed_data_mask
from augmentations import add_noise
import torch
import torch.nn as nn
import wandb

def prepare_data_embedding(data, args, model, device, cat_mask=None, con_mask=None):
    # x_categ is the the categorical data, with y appended as last feature. x_cont has continuous data. cat_mask is an array of ones same shape as x_categ except for last column(corresponding to y's) set to 0s. con_mask is an array of ones same shape as x_cont.
    image, ids, DOY, x_categ, x_cont, y_gts = data[0].to(device), data[1].to(device), data[2].to(
        device).type(torch.float32), data[3].to(device).type(torch.float32), data[4].type(torch.float32).to(
        device),data[5].to(device).type(torch.long)#,data[6].to(device).type(torch.float32)
    if args.train_noise_type is not None and args.train_noise_level > 0:
        noise_dict = {
            'noise_type': args.train_noise_type,
            'lambda': args.train_noise_level
        }
        if args.train_noise_type == 'cutmix':
            x_categ, x_cont = add_noise(x_categ, x_cont, noise_params=noise_dict)
        elif args.train_noise_type == 'missing':
            cat_mask, con_mask = add_noise(cat_mask, con_mask, noise_params=noise_dict)
    # We are converting the data to embeddings in the next step
    _, x_categ_enc, x_cont_enc, con_mask = embed_data_mask(x_categ, x_cont, model, False, DOY)
    return image, x_categ_enc, x_cont_enc, con_mask, y_gts


def train_epoch(args, epoch, model, device, dataloader, optimizer, scheduler, ratio_a, writer=None):
    criterion = nn.CrossEntropyLoss()
    softmax = nn.Softmax(dim=1)
    relu = nn.ReLU(inplace=True)
    tanh = nn.Tanh()

    model.train()
    print("Start training ... ")

    _loss = 0
    _loss_a = 0
    _loss_v = 0

    _a_angle = 0
    _v_angle = 0
    _a_diff = 0
    _v_diff = 0
    _ratio_a = 0

    for step, batch in enumerate(dataloader):
        # Unpack
        image, ids, DOY, x_categ, x_cont, y_gts = batch

        # Move + cast
        image = image.to(device, non_blocking=True).float()
        DOY     = DOY.to(device, non_blocking=True).float()
        x_categ = x_categ.to(device, non_blocking=True).float()
        x_cont  = x_cont.to(device, non_blocking=True).float()
        label  = y_gts.to(device, non_blocking=True).long().view(-1)  # CE expects [N] Long

        optimizer.zero_grad(set_to_none=True)
        # Forward pass
        _, x_categ_enc, x_cont_enc, con_mask = embed_data_mask(x_categ, x_cont, model.tab_net, False, DOY=DOY)

        a, v, out = model(image, x_categ_enc, x_cont_enc, con_mask)

        if args.fusion_method == 'sum':
            out_v = (torch.mm(v, torch.transpose(model.fusion_module.ffc_y.weight, 0, 1)) +
                     model.fusion_module.ffc_y.bias)
            out_a = (torch.mm(a, torch.transpose(model.fusion_module.ffc_x.weight, 0, 1)) +
                     model.fusion_module.ffc_x.bias)
        elif args.fusion_method == 'concat':
            weight_size = model.fusion_module.ffc_out.weight.size(1)
            out_v = (torch.mm(v, torch.transpose(model.fusion_module.ffc_out.weight[:, weight_size // 2:], 0, 1))
                     + model.fusion_module.ffc_out.bias / 2)
            out_a = (torch.mm(a, torch.transpose(model.fusion_module.ffc_out.weight[:, :weight_size // 2], 0, 1))
                     + model.fusion_module.ffc_out.bias / 2)
        elif args.fusion_method == 'film' or args.fusion_method == 'gated':
            out_v = out
            out_a = out

        # Loss calculation
        loss = criterion(out, label)
        loss_v = criterion(v, label)
        loss_a = criterion(a, label)

        loss.backward()  # Backward pass

        if args.modulation == 'Normal':
            score_v = sum([softmax(out_v)[i][label[i]] for i in range(out_v.size(0))])
            score_a = sum([softmax(out_a)[i][label[i]] for i in range(out_a.size(0))])

            ratio_v = score_v / score_a
            ratio_a = 1 / ratio_v
        else:
            # Modulation starts here !
            score_v = sum([softmax(out_v)[i][label[i]] for i in range(out_v.size(0))])
            score_a = sum([softmax(out_a)[i][label[i]] for i in range(out_a.size(0))])

            ratio_v = score_v / score_a
            ratio_a = 1 / ratio_v
            # ratio_v = 1 / ratio_a

            """
            Below is the Eq.(10) in our CVPR paper:
                    1 - tanh(alpha * rho_t_u), if rho_t_u > 1
            k_t_u =
                    1,                         else
            coeff_u is k_t_u, where t means iteration steps and u is modality indicator, either a or v.
            """
            if ratio_v > 1:
                coeff_v = 1 - tanh(args.alpha * relu(ratio_v))
                coeff_a = 1
                acc_v = 1
                acc_a = 1 + tanh(args.alpha * relu(ratio_v))
            else:
                coeff_a = 1 - tanh(args.alpha * relu(ratio_a))
                coeff_v = 1
                acc_a = 1
                acc_v = 1 + tanh(args.alpha * relu(ratio_a))

            # TODO: check again what the modulation does exactly and if it's not a problem that the final layers are now also being modulated
            if args.modulation_starts <= epoch <= args.modulation_ends:  # bug fixed
                for name, parms in model.named_parameters():
                    layer = str(name).split('.')[0]
                    if parms.grad is not None:
                        if 'img' in layer and len(parms.grad.size()) == 4:
                            if args.modulation == 'OGM_GE':  # bug fixed
                                parms.grad = parms.grad * coeff_v + \
                                             torch.zeros_like(parms.grad).normal_(0, parms.grad.std().item() + 1e-8)
                            elif args.modulation == 'OGM':
                                parms.grad *= coeff_v
                            elif args.modulation == 'Acc':
                                parms.grad *= acc_v
                        elif 'ffc' not in layer and len(parms.grad.size()) == 4:
                            if args.modulation == 'OGM_GE':  # bug fixed
                                parms.grad = parms.grad * coeff_a + \
                                             torch.zeros_like(parms.grad).normal_(0, parms.grad.std().item() + 1e-8)
                            elif args.modulation == 'OGM':
                                parms.grad *= coeff_a
                            elif args.modulation == 'Acc':
                                parms.grad *= acc_a

            else:
                pass

        optimizer.step()

        # Metrics tracking
        _loss += loss.item()
        _loss_a += loss_a.item()
        _loss_v += loss_v.item()

        if step % 10 == 0:
            wandb.log({"loss": loss, "loss_img": loss_a, "loss_tab": loss_v})

    return _loss / len(dataloader), _loss_a / len(dataloader), _loss_v / len(dataloader), _a_angle / len(dataloader), \
           _v_angle / len(dataloader), _ratio_a / len(dataloader)

def maybe_retain_grad(t):
    if isinstance(t, torch.Tensor) and t.requires_grad:
        t.retain_grad()

def train_epoch_tab(args, epoch, model, device, dataloader, optimizer, scheduler, ratio_a, writer=None,
                    debug_step0=True):
    """
    Single-epoch trainer for the tabular branch (SAINT). Adds robust sanity checks and
    lightweight debug probes that run only on the first batch of the epoch.

    - Targets shaped/typed correctly for CrossEntropyLoss
    - Optional class weights via `ratio_a`
    - Prints whether grads reach the head/backbone
    - Probes that parameters actually change after optimizer.step()
    - Scheduler stepped per-batch only if scheduler.by_step == True, else per-epoch at the end
    - Logs scalar loss to wandb every 10 steps (if available)
    """
    import torch
    from torch import nn

    # Loss (weighted CE if you pass a 1D weight tensor in ratio_a)
    if isinstance(ratio_a, torch.Tensor) and ratio_a.numel() > 1:
        criterion = nn.CrossEntropyLoss(weight=ratio_a.to(device))
    else:
        criterion = nn.CrossEntropyLoss()

    model.train()
    print("Start training tab transformer ... ")

    running_loss = 0.0
    steps = 0

    for step, batch in enumerate(dataloader):
        # Unpack
        image, ids, DOY, x_categ, x_cont, y_gts = batch

        # Move + cast
        DOY     = DOY.to(device, non_blocking=True).float()
        x_categ = x_categ.to(device, non_blocking=True).float()
        x_cont  = x_cont.to(device, non_blocking=True).float()
        y_gts   = y_gts.to(device, non_blocking=True).long().view(-1)  # CE expects [N] Long

        optimizer.zero_grad(set_to_none=True)

        # Forward through embeddings and transformer
        _, x_categ_enc, x_cont_enc, con_mask = embed_data_mask(x_categ, x_cont, model, False, DOY=DOY)
        reps   = model.transformer(x_categ_enc, x_cont_enc, con_mask)     # [N, T, D]
        logits = model.mlpfory(reps[:, 0, :])                             # [N, C] (y-token)

        # Shape & label sanity
        assert logits.ndim == 2, f"logits must be [N,C], got {logits.shape}"
        assert logits.size(0) == y_gts.size(0), f"N mismatch: {logits.size(0)} vs {y_gts.size(0)}"
        C = logits.size(1)
        assert y_gts.min().item() >= 0 and y_gts.max().item() < C, \
            f"target out of range [{y_gts.min().item()}, {y_gts.max().item()}] for C={C}"

        # Optional: retain grads on intermediates, only if they require grad
        if debug_step0 and step == 0:
            def maybe_retain_grad(t):
                if isinstance(t, torch.Tensor) and t.requires_grad:
                    t.retain_grad()
            maybe_retain_grad(x_categ_enc)
            maybe_retain_grad(x_cont_enc)
            maybe_retain_grad(reps)
            maybe_retain_grad(logits)
            with torch.no_grad():
                w_before = torch.nn.utils.parameters_to_vector(
                    [p for p in model.parameters() if p.requires_grad]
                ).clone()

        # Loss / backward
        loss = criterion(logits, y_gts)
        loss.backward()

        # One-time debug readouts
        if debug_step0 and step == 0:
            head_has_grad = any(p.grad is not None for p in model.mlpfory.parameters() if p.requires_grad)
            backbone_has_grad = any(p.grad is not None for p in model.transformer.parameters() if p.requires_grad)
            print(f"[debug] requires_grad: logits={logits.requires_grad} reps={reps.requires_grad} "
                  f"x_cont_enc={getattr(x_cont_enc, 'requires_grad', None)} "
                  f"x_categ_enc={getattr(x_categ_enc, 'requires_grad', None)}")
            print(f"[debug] head_has_grad={head_has_grad} backbone_has_grad={backbone_has_grad}")
            if getattr(logits, "grad", None) is not None:
                print(f"[debug] logits.grad |max|={(logits.grad.abs().max().item()):.3e}")
            if getattr(reps, "grad", None) is not None:
                print(f"[debug] reps.grad   |max|={(reps.grad.abs().max().item()):.3e}")

        # Optimizer step
        optimizer.step()

        # Param delta probe (once)
        if debug_step0 and step == 0:
            with torch.no_grad():
                w_after = torch.nn.utils.parameters_to_vector(
                    [p for p in model.parameters() if p.requires_grad]
                )
                max_delta = (w_after - w_before).abs().max().item()
            print(f"[debug] max param delta after step 0: {max_delta:.3e}")

        # (Optional) step scheduler per-batch if it declares that behavior
        if scheduler is not None and getattr(scheduler, "by_step", False):
            scheduler.step()

        # Logging
        running_loss += loss.item()
        steps += 1
        if step % 10 == 0:
            try:
                import wandb
                wandb.log({"loss_tab": loss.item(), "epoch": epoch, "step": step})
            except Exception:
                pass

    # Per-epoch scheduler step (default)
    if scheduler is not None and not getattr(scheduler, "by_step", False):
        try:
            scheduler.step()
        except TypeError:
            # some schedulers require metrics; handle upstream if needed
            pass

    return running_loss / max(1, steps)


def train_epoch_img(args, epoch, model, device, dataloader, optimizer, scheduler, ratio_a, writer=None):
    criterion = nn.CrossEntropyLoss()
    softmax = nn.Softmax(dim=1)
    relu = nn.ReLU(inplace=True)
    tanh = nn.Tanh()

    model.train()
    print("Start training ResNet ... ")

    for step, data in enumerate(dataloader):
        # Forward pass
        optimizer.zero_grad()
        # x_categ is the the categorical data, with y appended as last feature. x_cont has continuous data. cat_mask is an array of ones same shape as x_categ except for last column(corresponding to y's) set to 0s. con_mask is an array of ones same shape as x_cont.
        image, _, _, _, _, y_gts = data[0].to(device), data[1].to(device).type(torch.float32), data[2].to(
            device).type(torch.float32), data[3].to(device).type(torch.float32), data[4].type(torch.float32).to(
            device),data[5].to(device, dtype=torch.long)#,data[6].to(device).type(torch.float32)
        # We are converting the data to embeddings in the next step

        y_outs = model(image)

        loss = criterion(y_outs, y_gts.squeeze())
        loss.backward()
        optimizer.step()
        # print(running_loss)

        if step % 10 == 0:
            wandb.log({"loss_tab": loss})


