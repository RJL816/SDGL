import argparse
import csv
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset.AVEDataset import AVEDataset
from dataset.CramedDataset import CramedDataset, CramedDataset_swin
from dataset.Kinect400 import Kinect400
from dataset.KSDataset import KSDataset
from dataset.VGGSoundDataset import VGGSound
from models.basic_model import AVClassifier_DGL, AVClassifier_SDGL
from utils.result_summary import write_run_summary
from utils.utils import setup_seed, weight_init


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='CREMAD', type=str,
                        help='VGGSound, KineticSound, CREMAD, AVE')
    parser.add_argument('--modulation', default='OGM_GE', type=str,
                        choices=['Normal', 'OGM', 'OGM_GE'])
    parser.add_argument('--fusion_method', default='concat', type=str,
                        choices=['sum', 'concat', 'gated', 'film'])
    parser.add_argument('--fps', default=1, type=int)
    parser.add_argument('--use_video_frames', default=3, type=int)
    parser.add_argument('--num_frame', default=1, type=int, help='use how many frames for train')

    parser.add_argument('--audio_path', default='./train_test_data/CREMA-D/AudioWAV', type=str)
    parser.add_argument('--visual_path', default='./train_test_data/CREMA-D', type=str)

    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epochs', default=100, type=int)

    parser.add_argument('--optimizer', default='sgd', type=str)
    parser.add_argument('--learning_rate', default=0.001, type=float, help='initial learning rate')
    parser.add_argument('--lr_decay_step', default='[70]', type=str, help='where learning rate decays')
    parser.add_argument('--lr_decay_ratio', default=0.1, type=float, help='decay coefficient')

    parser.add_argument('--modulation_starts', default=0, type=int, help='where modulation begins')
    parser.add_argument('--modulation_ends', default=50, type=int, help='where modulation ends')
    parser.add_argument('--alpha', default=4.0, type=float, help='alpha in DGL')

    parser.add_argument('--grad_strategy', default='dgl', type=str, choices=['dgl', 'sdgl'])
    parser.add_argument('--selector_type', default='hard', type=str, choices=['hard'])
    parser.add_argument('--selector_level', default='feature', type=str, choices=['feature'])
    parser.add_argument('--selector_tau', default=0.0, type=float)
    parser.add_argument('--selector_lambda', default=1.0, type=float)
    # 不对称门控机制的四个参数
    parser.add_argument('--selector_tau_audio', default=None, type=float)
    parser.add_argument('--selector_tau_visual', default=None, type=float)
    parser.add_argument('--selector_lambda_audio', default=None, type=float)
    parser.add_argument('--selector_lambda_visual', default=None, type=float)
    # start_epoch 是什么时候开始进行选择性截断
    parser.add_argument('--selector_start_epoch', default=0, type=int)
    parser.add_argument('--selector_log_path', default='', type=str,
                        help='path to save SDGL selector logs')

    parser.add_argument('--ckpt_path', required=True, type=str, help='path to save trained models')
    parser.add_argument('--train', action='store_true', help='turn on train mode')

    parser.add_argument('--use_tensorboard', default=False, type=bool, help='whether to visualize')
    parser.add_argument('--tensorboard_path', type=str, help='path to save tensorboard logs')

    parser.add_argument('--random_seed', default=0, type=int)
    parser.add_argument('--gpu_ids', default='1', type=str, help='GPU ids')
    parser.add_argument('--modality', type=str, default='full')
    parser.add_argument('--backbone', type=str, default='resnet')
    parser.add_argument('--total_epoch', default=10, type=int)
    parser.add_argument('--drop', default=0, type=int)

    return parser.parse_args()


def get_num_classes(dataset):
    if dataset == 'VGGSound':
        return 309
    elif dataset == 'KineticSound':
        return 34
    elif dataset == 'kinect400':
        return 400
    elif dataset == 'CREMAD':
        return 6
    elif dataset == 'AVE':
        return 28
    else:
        raise NotImplementedError('Incorrect dataset name {}'.format(dataset))


def unwrap_model(model):
    if hasattr(model, 'module'):
        return model.module
    return model


def unpack_logits(model_output):
    if isinstance(model_output, dict):
        logits = model_output['logits']
        return logits['multi'], logits['audio'], logits['visual']
    return model_output


def get_features(model_output):
    if isinstance(model_output, dict):
        return model_output['features']
    return None


def get_selector_log_path(args):
    if args.selector_log_path:
        return args.selector_log_path
    return os.path.join(
        args.ckpt_path,
        '{}_{}_{}_selector.csv'.format(args.dataset, args.modality, args.grad_strategy)
    )


def ensure_parent_dir(path):
    parent_dir = os.path.dirname(path)
    if parent_dir and not os.path.exists(parent_dir):
        os.makedirs(parent_dir)


def ensure_csv_header(path, header):
    ensure_parent_dir(path)
    if not os.path.exists(path):
        with open(path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=",")
            writer.writerow(header)


def append_csv_row(path, row):
    ensure_parent_dir(path)
    with open(path, 'a+', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=",")
        writer.writerow(row)


def cosine_stats(grad_multi, grad_uni, tau):
    if grad_multi is None or grad_uni is None:
        return 0.0, 0.0, 0.0

    grad_multi = grad_multi.detach().reshape(grad_multi.size(0), -1)
    grad_uni = grad_uni.detach().reshape(grad_uni.size(0), -1)
    similarity = F.cosine_similarity(grad_multi, grad_uni, dim=1, eps=1e-8)
    sim_mean = similarity.mean().item()
    pos_ratio = (similarity > 0).float().mean().item()
    sample_useful_ratio = (similarity > tau).float().mean().item()
    return sim_mean, pos_ratio, sample_useful_ratio

# 获取选择器的参数
def get_selector_tau(args, modality):
    modality_tau = getattr(args, 'selector_tau_{}'.format(modality), None)
    # 如果没有使用不对称的门控机制，则退化为对称门控机制
    if modality_tau is None:
        return args.selector_tau
    return modality_tau


def get_selector_lambda(args, modality):
    modality_lambda = getattr(args, 'selector_lambda_{}'.format(modality), None)
    if modality_lambda is None:
        return args.selector_lambda
    return modality_lambda


def select_beta(args, epoch, similarity, tau, lambda_value):
    if epoch < args.selector_start_epoch:
        return 0.0
    if args.selector_type == 'hard':
        if similarity > tau:
            return lambda_value
        return 0.0
    raise NotImplementedError('Incorrect selector type: {}!'.format(args.selector_type))


def grad_list_norm(grads):
    grad_sq_sum = 0.0
    for grad in grads:
        if grad is None:
            continue
        grad_sq_sum += grad.detach().pow(2).sum().item()
    return grad_sq_sum ** 0.5

# 手动合并梯度
def merge_grad_lists(grads_uni, grads_multi, alpha, beta):
    merged_grads = []
    for grad_uni, grad_multi in zip(grads_uni, grads_multi):
        if grad_uni is None and grad_multi is None:
            merged_grads.append(None)
        elif grad_uni is None:
            merged_grads.append(grad_multi.detach() * beta)
        elif grad_multi is None:
            merged_grads.append(grad_uni.detach() * alpha)
        else:
            merged_grads.append(grad_uni.detach() * alpha + grad_multi.detach() * beta)
    return merged_grads


def assign_param_grads(params, grads):
    for param, grad in zip(params, grads):
        if grad is None:
            param.grad = None
        else:
            param.grad = grad.detach().clone()


def log_selector_stats(path, row):
    append_csv_row(path, row)


def print_epoch_stats(args, batch_loss, acc, best_acc, acc_a, acc_v, a_diveristy, v_diveristy, a_re, v_re):
    print("Loss: {:.3f}, Acc: {:.3f}, Best Acc: {:.3f}".format(batch_loss, acc, best_acc))
    print("Audio Acc: {:.3f}, Visual Acc: {:.3f}".format(acc_a, acc_v))
    if args.grad_strategy == 'sdgl':
        print("Audio sim: {:.3f}, Visual sim: {:.3f}".format(a_diveristy, v_diveristy))
        print("Audio sample-useful ratio: {:.3f}, Visual sample-useful ratio: {:.3f}".format(a_re, v_re))
    else:
        print("Audio similar: {:.3f}, Visual similar: {:.3f}".format(a_diveristy, v_diveristy))
        print("Audio regurize: {:.3f}, Visual regurize: {:.3f}".format(a_re, v_re))


def write_training_summary(args, log_path):
    selector_path = get_selector_log_path(args) if args.grad_strategy == 'sdgl' else None
    meta = {
        'dataset': args.dataset,
        'modality': args.modality,
        'grad_strategy': args.grad_strategy,
        'alpha': args.alpha,
        'selector_tau': args.selector_tau,
        'selector_lambda': args.selector_lambda,
        'selector_tau_audio': get_selector_tau(args, 'audio'),
        'selector_tau_visual': get_selector_tau(args, 'visual'),
        'selector_lambda_audio': get_selector_lambda(args, 'audio'),
        'selector_lambda_visual': get_selector_lambda(args, 'visual'),
        'selector_start_epoch': args.selector_start_epoch,
    }
    summary_text, text_path, csv_path = write_run_summary(
        args.ckpt_path, log_path, selector_path, meta
    )
    print(summary_text)
    print('Run summary saved to {} and {}.'.format(text_path, csv_path))


def train_epoch(args, epoch, model, device, dataloader, optimizer, scheduler, writer=None):
    criterion = nn.CrossEntropyLoss()

    if scheduler is not None:
        scheduler.step()

    if epoch < 20:
        print(epoch, optimizer.param_groups[0]['lr'])

    model.train()
    print("Start training ... ")
    _loss = 0
    _loss_a = 0
    _loss_v = 0
    _a_diveristy = 0
    _v_diveristy = 0
    _a_re = 0
    _v_re = 0

    module = unwrap_model(model)

    for step, (spec, image, label) in enumerate(tqdm(dataloader, desc="Epoch {}/{}".format(epoch, args.epochs))):
        spec = spec.to(device)
        image = image.to(device)
        label = label.to(device)

        optimizer.zero_grad()

        output = model(spec.unsqueeze(1).float(), image.float())
        out, out_a, out_v = unpack_logits(output)

        loss_v = criterion(out_v, label)
        loss_a = criterion(out_a, label)
        loss_f = criterion(out, label)

        loss_unimodal = (loss_a + loss_v) * args.alpha
        loss_unimodal.backward(retain_graph=True)

        for name, parms in model.named_parameters():
            layer = str(name).split('.')[1]
            if 'fusion' in layer:
                parms.grad = None

        loss_f.backward()

        if step % 100 == 0:
            print("unimodal_loss:", (loss_a + loss_v).item(), "cls_loss:", loss_f.item())
        # 梯度裁剪
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=40, norm_type=2)

        audio_grad_sum = 0
        for param in module.audio_net.parameters():
            if param.grad is not None:
                audio_grad_sum += torch.abs(param.grad).mean().item()

        visual_grad_sum = 0
        for param in module.visual_net.parameters():
            if param.grad is not None:
                visual_grad_sum += torch.abs(param.grad).mean().item()

        if step % 100 == 0:
            print("grad:", audio_grad_sum, visual_grad_sum)
            print("unimodal", torch.abs(out_a).mean().item(), torch.abs(out_v).mean().item())

        file_name = 'audio_visual_grad_vanilla.csv'
        with open(file_name, 'a', newline='') as csvfile:
            vanilla_writer = csv.writer(csvfile)
            vanilla_writer.writerow([audio_grad_sum, visual_grad_sum])

        optimizer.step()

        _loss += loss_f.item()
        _loss_a += loss_a.item()
        _loss_v += loss_v.item()

    return _loss / len(dataloader), _loss_a / len(dataloader), _loss_v / len(dataloader), _a_diveristy / len(
        dataloader), _v_diveristy / len(dataloader), _a_re / len(dataloader), _v_re / len(dataloader)


def train_epoch_sdgl(args, epoch, model, device, dataloader, optimizer, scheduler, writer=None):
    criterion = nn.CrossEntropyLoss()

    if scheduler is not None:
        scheduler.step()

    if epoch < 20:
        print(epoch, optimizer.param_groups[0]['lr'])

    model.train()
    print("Start training ... ")

    module = unwrap_model(model)
    # 获取模型参数
    audio_params = list(module.audio_net.parameters())
    visual_params = list(module.visual_net.parameters())
    fusion_params = list(module.fusion_module.parameters())
    tau_audio = get_selector_tau(args, 'audio')
    tau_visual = get_selector_tau(args, 'visual')
    lambda_audio = get_selector_lambda(args, 'audio')
    lambda_visual = get_selector_lambda(args, 'visual')
    selector_log_path = get_selector_log_path(args)
    # 初始化日志文件
    selector_log_path = get_selector_log_path(args)

    selector_header = [
        'phase', 'epoch', 'step',
        'loss_f', 'loss_a', 'loss_v',
        'sim_audio', 'sim_visual',
        'pos_ratio_audio', 'pos_ratio_visual',
        'sample_useful_ratio_audio', 'sample_useful_ratio_visual',
        'tau_audio', 'tau_visual',
        'lambda_audio', 'lambda_visual',
        'beta_audio', 'beta_visual',
        'actual_keep_audio', 'actual_keep_visual',
        'grad_norm_audio_uni', 'grad_norm_audio_multi', 'grad_norm_audio_final',
        'grad_norm_visual_uni', 'grad_norm_visual_multi', 'grad_norm_visual_final',
        'grad_norm_fusion', 'total_grad_norm'
    ]
    ensure_csv_header(selector_log_path, selector_header)

    _loss = 0
    _loss_a = 0
    _loss_v = 0
    _a_diveristy = 0
    _v_diveristy = 0
    _a_re = 0
    _v_re = 0
    _a_pos = 0
    _v_pos = 0
    _a_beta = 0
    _v_beta = 0

    for step, (spec, image, label) in enumerate(tqdm(dataloader, desc="Epoch {}/{}".format(epoch, args.epochs))):
        # 音频频谱、视频帧和标签获取
        spec = spec.to(device)
        image = image.to(device)
        label = label.to(device)

        optimizer.zero_grad()
        # 前向传播进入basic_model文件，返回字典{'features': {'audio': ..., 'visual': ...},'logits': ...'}
        output = model(spec.unsqueeze(1).float(), image.float())
        features = get_features(output)
        if features is None:
            raise TypeError('SDGL expects dict outputs with features and logits.')
        # 提取特征和logits
        out, out_a, out_v = unpack_logits(output) # 多模态、音频和视频的预测
        z_audio = features['audio'] # 音频特征
        z_visual = features['visual'] # 视频特征 
        # 计算损失
        loss_a = criterion(out_a, label)
        loss_v = criterion(out_v, label)
        loss_f = criterion(out, label)

        # 音频特征梯度
        # autograd.grad()函数是手动计算梯度函数，第一个参数是需要求导的标量输出，第二个参数是对哪个输入求导
        # 返回值是一个tuple，[0]是取出第一个梯度
        grad_audio_uni_feat = torch.autograd.grad(
            loss_a, z_audio, retain_graph=True, allow_unused=True
        )[0]
        grad_audio_multi_feat = torch.autograd.grad(
            loss_f, z_audio, retain_graph=True, allow_unused=True
        )[0]
        # 视觉特征梯度
        grad_visual_uni_feat = torch.autograd.grad(
            loss_v, z_visual, retain_graph=True, allow_unused=True
        )[0]
        grad_visual_multi_feat = torch.autograd.grad(
            loss_f, z_visual, retain_graph=True, allow_unused=True
        )[0]
        # 计算单模态梯度和多模态梯度的余弦相似度
        sim_audio, pos_ratio_audio, sample_useful_ratio_audio = cosine_stats(
            grad_audio_multi_feat, grad_audio_uni_feat, tau_audio
        )
        sim_visual, pos_ratio_visual, sample_useful_ratio_visual = cosine_stats(
            grad_visual_multi_feat, grad_visual_uni_feat, tau_visual
        )
        # 选择beta，逻辑是如果余弦相似度>tau则保留多模态beta不为0，否则为0
        # beta不为0的时候值等于lambda
        beta_audio = select_beta(args, epoch, sim_audio, tau_audio, lambda_audio)
        beta_visual = select_beta(args, epoch, sim_visual, tau_visual, lambda_visual)

        # # 参数级别梯度
        # # 单模态梯度
        # audio_grads_uni = torch.autograd.grad(
        #     loss_a, audio_params, retain_graph=True, allow_unused=True
        # )
        # visual_grads_uni = torch.autograd.grad(
        #     loss_v, visual_params, retain_graph=True, allow_unused=True
        # )
        # # 多模态梯度
        # audio_grads_multi = torch.autograd.grad(
        #     loss_f, audio_params, retain_graph=True, allow_unused=True
        # )
        # visual_grads_multi = torch.autograd.grad(
        #     loss_f, visual_params, retain_graph=True, allow_unused=True
        # )
        # fusion_grads = torch.autograd.grad(
        #     loss_f, fusion_params, retain_graph=False, allow_unused=True
        # )

        # 合并梯度 final_grad=alpha*unimodal_grad+beta*multimodal_grad
        merged_audio_grads = merge_grad_lists(audio_grads_uni, audio_grads_multi, args.alpha, beta_audio)
        merged_visual_grads = merge_grad_lists(visual_grads_uni, visual_grads_multi, args.alpha, beta_visual)
        # 合并后的梯度手动赋值给参数
        assign_param_grads(audio_params, merged_audio_grads)
        assign_param_grads(visual_params, merged_visual_grads)
        assign_param_grads(fusion_params, fusion_grads)

        # 梯度裁剪
        total_grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=40, norm_type=2)
        # 传回梯度，使用手动赋值的梯度更新参数而不是使用backward
        optimizer.step()

        # 日志记录损失值、相似度、beta值、各模态梯度范数、样本有用比例
        grad_norm_audio_uni = grad_list_norm(audio_grads_uni)
        grad_norm_audio_multi = grad_list_norm(audio_grads_multi)
        grad_norm_audio_final = grad_list_norm(merged_audio_grads)
        grad_norm_visual_uni = grad_list_norm(visual_grads_uni)
        grad_norm_visual_multi = grad_list_norm(visual_grads_multi)
        grad_norm_visual_final = grad_list_norm(merged_visual_grads)
        grad_norm_fusion = grad_list_norm(fusion_grads)

        log_row = [
            'step', epoch, step,
            loss_f.item(), loss_a.item(), loss_v.item(),
            sim_audio, sim_visual,
            pos_ratio_audio, pos_ratio_visual,
            sample_useful_ratio_audio, sample_useful_ratio_visual,
            tau_audio, tau_visual,
            lambda_audio, lambda_visual,
            beta_audio, beta_visual,
            int(beta_audio > 0), int(beta_visual > 0),
            grad_norm_audio_uni, grad_norm_audio_multi, grad_norm_audio_final,
            grad_norm_visual_uni, grad_norm_visual_multi, grad_norm_visual_final,
            grad_norm_fusion, float(total_grad_norm)
        ]
        log_selector_stats(selector_log_path, log_row)

        if writer is not None:
            iteration = epoch * len(dataloader) + step
            writer.add_scalar('Selector/audio_sim', sim_audio, iteration)
            writer.add_scalar('Selector/visual_sim', sim_visual, iteration)
            writer.add_scalar('Selector/audio_beta', beta_audio, iteration)
            writer.add_scalar('Selector/visual_beta', beta_visual, iteration)

        if step % 100 == 0:
            print("loss_a:", loss_a.item(), "loss_v:", loss_v.item(), "loss_f:", loss_f.item())
            print("audio sim:", sim_audio, "visual sim:", sim_visual)
            print("audio beta:", beta_audio, "visual beta:", beta_visual)
            print("audio sample useful:", sample_useful_ratio_audio, "visual sample useful:", sample_useful_ratio_visual)

        _loss += loss_f.item()
        _loss_a += loss_a.item()
        _loss_v += loss_v.item()
        _a_diveristy += sim_audio
        _v_diveristy += sim_visual
        _a_re += sample_useful_ratio_audio
        _v_re += sample_useful_ratio_visual
        _a_pos += pos_ratio_audio
        _v_pos += pos_ratio_visual
        _a_beta += beta_audio
        _v_beta += beta_visual

    step_count = len(dataloader)
    epoch_row = [
        'epoch', epoch, -1,
        _loss / step_count, _loss_a / step_count, _loss_v / step_count,
        _a_diveristy / step_count, _v_diveristy / step_count,
        _a_pos / step_count, _v_pos / step_count,
        _a_re / step_count, _v_re / step_count,
        tau_audio, tau_visual,
        lambda_audio, lambda_visual,
        _a_beta / step_count, _v_beta / step_count,
        0, 0,
        0, 0, 0,
        0, 0, 0,
        0, 0
    ]
    log_selector_stats(selector_log_path, epoch_row)

    return _loss / step_count, _loss_a / step_count, _loss_v / step_count, _a_diveristy / step_count, \
        _v_diveristy / step_count, _a_re / step_count, _v_re / step_count


def valid(args, model, device, dataloader):
    softmax = nn.Softmax(dim=1)
    n_classes = get_num_classes(args.dataset)

    module = unwrap_model(model)
    module.args.drop = 0
    with torch.no_grad():
        model.eval()
        print(module.args.drop)
        num = [0.0 for _ in range(n_classes)]
        acc = [0.0 for _ in range(n_classes)]
        acc_a = [0.0 for _ in range(n_classes)]
        acc_v = [0.0 for _ in range(n_classes)]

        for step, (spec, image, label) in enumerate(dataloader):
            spec = spec.to(device)
            image = image.to(device)
            label = label.to(device)

            output = model(spec.unsqueeze(1).float(), image.float())
            out, out_a, out_v = unpack_logits(output)

            prediction = softmax(out)
            pred_v = softmax(out_v)
            pred_a = softmax(out_a)

            for i in range(image.shape[0]):
                label_i = int(label[i].item())
                ma = np.argmax(prediction[i].cpu().data.numpy())
                v = np.argmax(pred_v[i].cpu().data.numpy())
                a = np.argmax(pred_a[i].cpu().data.numpy())
                num[label_i] += 1.0

                if label_i == ma:
                    acc[label_i] += 1.0
                if label_i == v:
                    acc_v[label_i] += 1.0
                if label_i == a:
                    acc_a[label_i] += 1.0

    module.args.drop = 1
    return sum(acc) / sum(num), sum(acc_a) / sum(num), sum(acc_v) / sum(num)


def main():
    args = get_arguments()
    args.p = [0, 0]
    print(args)

    if args.grad_strategy == 'sdgl':
        if args.selector_type != 'hard':
            raise NotImplementedError('SDGL currently only supports hard selector.')
        if args.selector_level != 'feature':
            raise NotImplementedError('SDGL currently only supports feature-level selector.')
        if args.modality != 'full':
            raise NotImplementedError('SDGL currently only supports full modality.')

    setup_seed(args.random_seed)
    gpu_ids = list(range(torch.cuda.device_count()))
    device = torch.device('cuda:0')

    if args.backbone == 'resnet':
        if args.grad_strategy == 'sdgl':
            model = AVClassifier_SDGL(args)
        else:
            model = AVClassifier_DGL(args)
        model.apply(weight_init)
    else:
        raise EOFError

    model.to(device)
    model = torch.nn.DataParallel(model, device_ids=gpu_ids)
    model.cuda()

    if args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, eval(args.lr_decay_step), args.lr_decay_ratio)
    elif args.optimizer == 'AdaGrad':
        optimizer = optim.Adagrad(model.parameters(), lr=args.learning_rate)
        scheduler = None
    elif args.optimizer == 'Adam':
        optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999))
        scheduler = None
    else:
        raise ValueError

    if args.dataset == 'VGGSound':
        train_dataset = VGGSound(args, mode='train')
        test_dataset = VGGSound(args, mode='test')
    elif args.dataset == 'KineticSound':
        train_dataset = KSDataset(args, mode='train')
        test_dataset = KSDataset(args, mode='test')
    elif args.dataset == 'kinect400':
        train_dataset = Kinect400(args, mode='train')
        test_dataset = Kinect400(args, mode='test')
    elif args.dataset == 'CREMAD':
        if args.backbone == 'swin':
            train_dataset = CramedDataset_swin(args, mode='train')
            test_dataset = CramedDataset_swin(args, mode='test')
        else:
            train_dataset = CramedDataset(args, mode='train')
            test_dataset = CramedDataset(args, mode='test')
    elif args.dataset == 'AVE':
        train_dataset = AVEDataset(args, mode='train')
        test_dataset = AVEDataset(args, mode='test')
    else:
        raise NotImplementedError('Incorrect dataset name {}! '
                                  'Only support VGGSound, KineticSound and CREMA-D for now!'.format(args.dataset))

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size,
                                  shuffle=True, num_workers=32, pin_memory=True, drop_last=True)

    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size,
                                 shuffle=False, num_workers=32, pin_memory=True, drop_last=True)

    if not os.path.exists(args.ckpt_path):
        os.makedirs(args.ckpt_path)

    log_path = os.path.join(args.ckpt_path, args.dataset + '_' + args.modality + '.csv')
    ensure_parent_dir(log_path)
    with open(log_path, 'a+', newline='') as csvfile:
        log_writer = csv.writer(csvfile, delimiter=",")
        log_writer.writerow([1000, 1000, 1000])

    if args.grad_strategy == 'sdgl':
        ensure_csv_header(
            get_selector_log_path(args),
            [
                'phase', 'epoch', 'step',
                'loss_f', 'loss_a', 'loss_v',
                'sim_audio', 'sim_visual',
                'pos_ratio_audio', 'pos_ratio_visual',
                'sample_useful_ratio_audio', 'sample_useful_ratio_visual',
                'tau_audio', 'tau_visual',
                'lambda_audio', 'lambda_visual',
                'beta_audio', 'beta_visual',
                'actual_keep_audio', 'actual_keep_visual',
                'grad_norm_audio_uni', 'grad_norm_audio_multi', 'grad_norm_audio_final',
                'grad_norm_visual_uni', 'grad_norm_visual_multi', 'grad_norm_visual_final',
                'grad_norm_fusion', 'total_grad_norm'
            ]
        )

    train_fn = train_epoch_sdgl if args.grad_strategy == 'sdgl' else train_epoch

    if args.train:
        best_acc = 0.0
        acc, acc_a, acc_v = 0, 0, 0

        for epoch in range(args.epochs):
            print('Epoch: {}: '.format(epoch))
            args.epoch_now = epoch

            if args.use_tensorboard:
                writer_path = os.path.join(args.tensorboard_path, args.dataset)
                if not os.path.exists(writer_path):
                    os.mkdir(writer_path)
                log_name = '{}_{}_{}'.format(args.fusion_method, args.modulation, args.grad_strategy)
                writer = SummaryWriter(os.path.join(writer_path, log_name))

                batch_loss, batch_loss_a, batch_loss_v, a_diveristy, v_diveristy, a_re, v_re = train_fn(
                    args, epoch, model, device, train_dataloader, optimizer, scheduler, writer
                )
                acc, acc_a, acc_v = valid(args, model, device, test_dataloader)

                writer.add_scalars('Loss', {'Total Loss': batch_loss,
                                            'Audio Loss': batch_loss_a,
                                            'Visual Loss': batch_loss_v}, epoch)
                writer.add_scalars('Evaluation', {'Total Accuracy': acc,
                                                  'Audio Accuracy': acc_a,
                                                  'Visual Accuracy': acc_v}, epoch)
                writer.close()
            else:
                batch_loss, batch_loss_a, batch_loss_v, a_diveristy, v_diveristy, a_re, v_re = train_fn(
                    args, epoch, model, device, train_dataloader, optimizer, scheduler
                )
                acc, acc_a, acc_v = valid(args, model, device, test_dataloader)

                with open(log_path, 'a+', newline='') as csvfile:
                    log_writer = csv.writer(csvfile, delimiter=",")
                    log_writer.writerow([acc, acc_a, acc_v])

            if acc > best_acc and epoch:
                best_acc = float(acc)

                if not os.path.exists(args.ckpt_path):
                    os.makedirs(args.ckpt_path)

                model_name = 'best_model_of_dataset_{}_{}_alpha_{}' \
                             'optimizer_{}_modulate_starts_{}_ends_{}_' \
                             'epoch_{}_acc_{}.pth'.format(args.dataset,
                                                          args.modulation,
                                                          args.alpha,
                                                          args.optimizer,
                                                          args.modulation_starts,
                                                          args.modulation_ends,
                                                          epoch, acc)

                if scheduler is None:
                    saved_dict = {
                        'saved_epoch': epoch,
                        'modulation': args.modulation,
                        'alpha': args.alpha,
                        'fusion': args.fusion_method,
                        'grad_strategy': args.grad_strategy,
                        'selector_type': args.selector_type,
                        'selector_level': args.selector_level,
                        'selector_tau': args.selector_tau,
                        'selector_lambda': args.selector_lambda,
                        'selector_tau_audio': get_selector_tau(args, 'audio'),
                        'selector_tau_visual': get_selector_tau(args, 'visual'),
                        'selector_lambda_audio': get_selector_lambda(args, 'audio'),
                        'selector_lambda_visual': get_selector_lambda(args, 'visual'),
                        'selector_start_epoch': args.selector_start_epoch,
                        'acc': acc,
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                    }
                else:
                    saved_dict = {
                        'saved_epoch': epoch,
                        'modulation': args.modulation,
                        'alpha': args.alpha,
                        'fusion': args.fusion_method,
                        'grad_strategy': args.grad_strategy,
                        'selector_type': args.selector_type,
                        'selector_level': args.selector_level,
                        'selector_tau': args.selector_tau,
                        'selector_lambda': args.selector_lambda,
                        'selector_tau_audio': get_selector_tau(args, 'audio'),
                        'selector_tau_visual': get_selector_tau(args, 'visual'),
                        'selector_lambda_audio': get_selector_lambda(args, 'audio'),
                        'selector_lambda_visual': get_selector_lambda(args, 'visual'),
                        'selector_start_epoch': args.selector_start_epoch,
                        'acc': acc,
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict()
                    }

                save_dir = os.path.join(args.ckpt_path, model_name)
                torch.save(saved_dict, save_dir)
                print('The best model has been saved at {}.'.format(save_dir))
                print("Loss: {:.3f}, Acc: {:.3f}".format(batch_loss, acc))
                print("Audio Acc: {:.3f}, Visual Acc: {:.3f}".format(acc_a, acc_v))
                if args.grad_strategy == 'sdgl':
                    print("Audio sim: {:.3f}, Visual sim: {:.3f}".format(a_diveristy, v_diveristy))
                    print("Audio sample-useful ratio: {:.3f}, Visual sample-useful ratio: {:.3f}".format(a_re, v_re))
                else:
                    print("Audio similar: {:.3f}, Visual similar: {:.3f}".format(a_diveristy, v_diveristy))
                    print("Audio regurize: {:.3f}, Visual regurize: {:.3f}".format(a_re, v_re))
            else:
                print_epoch_stats(args, batch_loss, acc, best_acc, acc_a, acc_v, a_diveristy, v_diveristy, a_re, v_re)

        write_training_summary(args, log_path)

    else:
        loaded_dict = torch.load(args.ckpt_path)
        modulation = loaded_dict['modulation']
        fusion = loaded_dict['fusion']
        saved_grad_strategy = loaded_dict.get('grad_strategy', 'dgl')
        state_dict = loaded_dict['model']

        assert modulation == args.modulation, 'inconsistency between modulation method of loaded model and args !'
        assert fusion == args.fusion_method, 'inconsistency between fusion method of loaded model and args !'
        assert saved_grad_strategy == args.grad_strategy, 'inconsistency between grad strategy of loaded model and args !'

        model.load_state_dict(state_dict)
        print('Trained model loaded!')

        acc, acc_a, acc_v = valid(args, model, device, test_dataloader)
        print('Accuracy: {}, accuracy_a: {}, accuracy_v: {}'.format(acc, acc_a, acc_v))


if __name__ == "__main__":
    main()
