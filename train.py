# Created by Baole Fang at 3/29/23

import argparse
import gc
import os

from torchmetrics.classification import BinaryAccuracy
import wandb
import yaml
import torch
from torchsummaryX import summary

import models
import utils
from torch.optim import lr_scheduler
from torch import nn
from tqdm import tqdm
from data import create_dataloader


def prepare_training():
    if config.get('resume') is not None:
        sv_file = torch.load(config['resume'])
        model = models.make(sv_file['model'], load_sd=True).cuda()
        optimizer = utils.make_optimizer(
            model.parameters(), sv_file['optimizer'], load_sd=True)
        if config.get('optimizer').get('force_lr'):
            optimizer.param_groups[0]['lr'] = config.get('optimizer').get('force_lr')
        epoch_start = sv_file['epoch'] + 1
        if config.get('scheduler') is None:
            scheduler = None
        else:
            scheduler = utils.make_scheduler(optimizer, sv_file['scheduler'], True)
        max_val_v = sv_file['accuracy']
    else:
        model = models.make(config['model']).cuda()
        optimizer = utils.make_optimizer(
            model.parameters(), config['optimizer'])
        epoch_start = 1
        if config.get('scheduler') is None:
            scheduler = None
        else:
            scheduler_class = lr_scheduler.__dict__[config.get('scheduler')['name']]
            scheduler = scheduler_class(optimizer=optimizer, **config.get('scheduler')['args'])
        if config['precision'] == 'half':
            model = model.half()
        max_val_v = -1e18

    log('model: #params={}'.format(utils.compute_num_params(model, text=True)))
    return model, optimizer, epoch_start, scheduler, max_val_v


def train_model(model, train_loader, optimizer, criterion, metrics):
    model.train()
    batch_bar = tqdm(total=len(train_loader), dynamic_ncols=True, leave=False, position=0, desc='Train')

    total_loss = 0
    metrics.reset()

    for i, data in enumerate(train_loader):
        optimizer.zero_grad()

        x, y = data
        x, y = x.cuda(), y.cuda()

        pred = model(x)
        loss = criterion(pred, y)

        total_loss += loss.item()
        metrics.update(pred, y)

        batch_bar.set_postfix(
            loss="{:.04f}".format(float(total_loss / (i + 1))),
            acc="{:.04f}".format(float(metrics.compute())),
            lr="{:.06f}".format(float(optimizer.param_groups[0]['lr'])))

        batch_bar.update()  # Update tqdm bar

        # Another couple things you need for FP16.
        scaler.scale(loss).backward()  # This is a replacement for loss.backward()
        scaler.step(optimizer)  # This is a replacement for optimizer.step()
        scaler.update()  # This is something added just for FP16

        del x, y, pred, loss
        torch.cuda.empty_cache()

    batch_bar.close()  # You need this to close the tqdm bar

    return metrics.compute(), total_loss / len(train_loader)


def validate_model(model, val_loader, criterion, metrics):
    model.eval()
    batch_bar = tqdm(total=len(val_loader), dynamic_ncols=True, position=0, leave=False, desc='Val')

    total_loss = 0
    metrics.reset()

    for i, data in enumerate(val_loader):
        x, y = data
        x, y = x.cuda(), y.cuda()

        with torch.inference_mode():
            pred = model(x)
            loss = criterion(pred, y)

        total_loss += float(loss)
        metrics.update(pred, y)

        batch_bar.set_postfix(loss="{:.04f}".format(float(total_loss / (i + 1))),
                              acc="{:.04f}".format(float(metrics.compute())))

        batch_bar.update()

        del x, y, pred, loss
        torch.cuda.empty_cache()

    batch_bar.close()
    total_loss = total_loss / len(val_loader)
    return metrics.compute(), total_loss


def main(config_, save_path_, use_wandb=True):
    global config, log, scaler
    config = config_
    remove = config.get('resume') is None
    log = utils.set_save_path(save_path, remove)
    with open(os.path.join(save_path_, 'config.yaml'), 'w') as f:
        yaml.dump(config, f, sort_keys=False)

    train_loader, valid_loader, test_loader = create_dataloader(**config.get('dataset'))
    model, optimizer, epoch_start, scheduler, max_val_v = prepare_training()

    for data in train_loader:
        x, y = data
        print(x.shape, y.shape)
        break
    summary(model, x.cuda())

    n_gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    if n_gpus > 1:
        model = nn.parallel.DataParallel(model)

    epoch_max = config['epoch_max']
    epoch_save = config.get('epoch_save')

    criterion = nn.BCELoss()
    metrics = BinaryAccuracy().cuda()

    timer = utils.Timer()
    scaler = torch.cuda.amp.GradScaler()

    if use_wandb:
        wandb.login(key="key")
        if epoch_start == 1:
            run = wandb.init(
                name=config.get('name'),  ## Wandb creates random run names if you skip this field
                reinit=True,  ### Allows reinitalizing runs when you re-run this cell
                # run_id = ### Insert specific run id here if you want to resume a previous run
                # resume = "must" ### You need this to resume previous runs, but comment out reinit = True when using this
                project="project",  ### Project should be created in your wandb account
                config=config  ### Wandb Config for your run
            )
        else:
            with open(os.path.join(save_path_, 'id.txt')) as f:
                id = f.readline().rstrip('\n')
            run = wandb.init(
                name=config.get('name'),  ## Wandb creates random run names if you skip this field
                reinit=True,  ### Allows reinitalizing runs when you re-run this cell
                id=id,  # Insert specific run id here if you want to resume a previous run
                resume="must",  ### You need this to resume previous runs, but comment out reinit = True when using this
                project="project",  ### Project should be created in your wandb account
                config=config  ### Wandb Config for your run
            )
        log(run.id, 'id.txt')

    gc.collect()  # These commands help you when you face CUDA OOM error
    torch.cuda.empty_cache()

    for epoch in range(epoch_start, epoch_max + 1):
        t_epoch_start = timer.t()
        log_info = ['epoch {}/{}'.format(epoch, epoch_max)]

        curr_lr = float(optimizer.param_groups[0]['lr'])
        train_acc, train_loss = train_model(model, train_loader, optimizer, criterion, metrics)
        if scheduler is not None:
            scheduler.step()

        log_info.append('train: acc={:.4f} loss={:.4f} lr={:.4f}'.format(train_acc, train_loss, curr_lr))

        val_acc, val_loss = validate_model(model, valid_loader, criterion, metrics)
        log_info.append("val: acc={:.04f} loss={:.04f}".format(val_acc, val_loss))

        if use_wandb:
            wandb.log({"train_loss": train_loss, 'train_Acc': train_acc, 'validation_Acc': val_acc,
                       'validation_loss': val_loss, "learning_Rate": curr_lr})

        if n_gpus > 1:
            model_ = model.module
        else:
            model_ = model
        model_spec = config['model']
        model_spec['sd'] = model_.state_dict()
        optimizer_spec = config['optimizer']
        optimizer_spec['sd'] = optimizer.state_dict()
        if scheduler:
            scheduler_spec = config['scheduler']
            scheduler_spec['sd'] = scheduler.state_dict()
            sv_file = {
                'model': model_spec,
                'optimizer': optimizer_spec,
                'scheduler': scheduler_spec,
                'epoch': epoch,
                'accuracy': min(max_val_v, val_acc)
            }
        else:
            sv_file = {
                'model': model_spec,
                'optimizer': optimizer_spec,
                'scheduler': None,
                'epoch': epoch,
                'accuracy': min(max_val_v, val_acc)
            }

        torch.save(sv_file, os.path.join(save_path_, 'epoch-last.pth'))

        if (epoch_save is not None) and (epoch % epoch_save == 0):
            torch.save(sv_file, os.path.join(save_path_, 'epoch-{}.pth'.format(epoch)))

        if n_gpus > 1 and (config.get('eval_bsize') is not None):
            model_ = model.module
        else:
            model_ = model

        if val_acc < max_val_v:
            max_val_v = val_acc
            torch.save(sv_file, os.path.join(save_path_, 'epoch-best.pth'))

        t = timer.t()
        prog = (epoch - epoch_start + 1) / (epoch_max - epoch_start + 1)
        t_epoch = utils.time_text(t - t_epoch_start)
        t_elapsed, t_all = utils.time_text(t), utils.time_text(t / prog)
        log_info.append('{} {}/{}\n'.format(t_epoch, t_elapsed, t_all))

        log(', '.join(log_info))

    log('max_memory_allocated=' + str(torch.cuda.max_memory_allocated()))
    log('max_memory_reserved=' + str(torch.cuda.max_memory_reserved()))
    if use_wandb:
        run.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--name', default=None)
    parser.add_argument('--tag', default=None)
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--wandb', action='store_true')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        print('config loaded.')

    save_name = args.name
    if save_name is None:
        save_name = '_' + args.config.split('/')[-1][:-len('.yaml')]
        config['name'] = save_name.split('_')[-1]
    if args.tag is not None:
        save_name += '_' + args.tag
    save_path = os.path.join('./save', save_name)
    main(config, save_path, args.wandb)
