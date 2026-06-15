import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbone import resnet18
from .fusion_modules import SumFusion, ConcatFusion, FiLM, GatedFusion, ConcatFusion_Swin,ConcatFusion_DGL,GatedFusion_DGL,SumFusion_DGL,FiLM_DGL,ConcatFusion_DGL_unimodal,SumFusion_SDGL,ConcatFusion_SDGL,FiLM_SDGL,GatedFusion_SDGL
import numpy as np
from models.swin_transformer import SwinTransformer


class AVClassifier_DGL(nn.Module):
    def __init__(self, args):
        super(AVClassifier_DGL, self).__init__()
        # 根据不同数据集进行不同的初始化
        fusion = args.fusion_method
        if args.dataset == 'VGGSound':
            n_classes = 309
        elif args.dataset == 'KineticSound':
            n_classes = 34
        elif args.dataset == 'kinect400':
            n_classes = 400
        elif args.dataset == 'CREMAD':
            n_classes = 6
        elif args.dataset == 'AVE':
            n_classes = 28
        else:
            raise NotImplementedError('Incorrect dataset name {}'.format(args.dataset))
        # 融合方式不同选择不同的融合函数
        if fusion == 'sum':
            self.fusion_module = SumFusion_DGL(output_dim=n_classes)
        elif fusion == 'concat':
            if args.dataset == 'kinect400':
                self.fusion_module = ConcatFusion_DGL(output_dim=n_classes,input_dim=1024)
            else:
                self.fusion_module = ConcatFusion_DGL(output_dim=n_classes)
        elif fusion == 'film':
            self.fusion_module = FiLM_DGL(output_dim=n_classes, x_film=True)
        elif fusion == 'gated':
            self.fusion_module = GatedFusion_DGL(output_dim=n_classes, x_gate=True)
        else:
            raise NotImplementedError('Incorrect fusion method: {}!'.format(fusion))
        # 根据不同模态以及不同数据集进入不同分支
        if args.modality == 'full':
            self.audio_net = resnet18(modality='audio', args=args)
            self.visual_net = resnet18(modality='visual', args=args)

        if args.modality == 'visual':
            if args.dataset == 'kinect400':
                self.visual_net = resnet18(modality='visual', args=args)
                self.visual_classifier = nn.Linear(512, n_classes)
            else:
                self.visual_net = resnet18(modality='visual', args=args)
                self.visual_classifier = nn.Linear(512, n_classes)
        if args.modality == 'audio':
            if args.dataset == 'kinect400':
                self.audio_net = resnet18(modality='audio', args=args)
                self.audio_classifier = nn.Linear(512, n_classes)
            else:
                self.audio_net = resnet18(modality='audio', args=args)
                self.audio_classifier = nn.Linear(512, n_classes)
        self.modality = args.modality
        self.args = args



    def forward(self, audio, visual):

        if self.modality == 'full':

            # 音频和视觉分别进入各自的resnet得到a和v，也就是编码器提取特征
            a = self.audio_net(audio)  # only feature
            v = self.visual_net(visual)
            # 获取尺寸
            (_, C, H, W) = v.size()
            B = a.size()[0]
            v = v.view(B, -1, C, H, W)
            v = v.permute(0, 2, 1, 3, 4)

            a = F.adaptive_avg_pool2d(a, 1)
            v = F.adaptive_avg_pool3d(v, 1)
            # 展平
            a = torch.flatten(a, 1)
            v = torch.flatten(v, 1)
            # out是多模态融合预测，a_out是只保留音频把视觉置零后的单模态预测，v_out是只保留视觉把音频置零后的单模态预测
            # 进入fusion_module进行梯度截断
            a_out, v_out, out = self.fusion_module(a, v)  # av 是原来的，out是融合结果

            return  out,a_out,v_out
        
        elif self.modality == 'visual':
            # 只有视觉
            v = self.visual_net(visual)


            (_, C, H, W) = v.size()
            B = self.args.batch_size
            v = v.view(B, -1, C, H, W)

            v = v.permute(0, 2, 1, 3, 4)

            v = F.adaptive_avg_pool3d(v, 1)

            v = torch.flatten(v, 1)

            out = self.visual_classifier(v)

            a = torch.zeros_like(v)


            return out, out, out

        elif self.modality == 'audio':
            # 只有音频
            a = self.audio_net(audio)  # only feature
            a_feature = a

            a = F.adaptive_avg_pool2d(a, 1)

            a = torch.flatten(a, 1)

            out = self.audio_classifier(a)
            v = torch.zeros_like(a)

            return out, out,out
        else:
            # 什么都没有
            return 0, 0, 0



class AVClassifier(nn.Module):
    def __init__(self, args):
        super(AVClassifier, self).__init__()

        fusion = args.fusion_method
        if args.dataset == 'VGGSound':
            n_classes = 309
        elif args.dataset == 'KineticSound':
            n_classes = 34
        elif args.dataset == 'CREMAD':
            n_classes = 6
        elif args.dataset == 'AVE':
            n_classes = 28
        else:
            raise NotImplementedError('Incorrect dataset name {}'.format(args.dataset))

        if fusion == 'sum':
            self.fusion_module = SumFusion(output_dim=n_classes)
        elif fusion == 'concat':
            self.fusion_module = ConcatFusion(output_dim=n_classes)
        elif fusion == 'film':
            self.fusion_module = FiLM(output_dim=n_classes, x_film=True)
        elif fusion == 'gated':
            self.fusion_module = GatedFusion(output_dim=n_classes, x_gate=True)
        else:
            raise NotImplementedError('Incorrect fusion method: {}!'.format(fusion))

        if args.modality == 'full':
            self.audio_net = resnet18(modality='audio',args=args)
            self.visual_net = resnet18(modality='visual',args=args)

        if args.modality == 'visual':
            self.visual_net = resnet18(modality='visual',args=args)
            self.visual_classifier = nn.Linear(512, n_classes)
        if args.modality == 'audio':
            self.audio_net = resnet18(modality='audio',args=args)
            self.audio_classifier = nn.Linear(512, n_classes)

        self.args = args

    def forward(self, audio, visual):

        if self.args.modality == 'full':
            a = self.audio_net(audio)  # only feature
            v = self.visual_net(visual)


            (_, C, H, W) = v.size()
            B = a.size()[0]
            # print(B)
            # print(B*C*H*W)
            v = v.view(B, -1, C, H, W)
            v = v.permute(0, 2, 1, 3, 4)

            a = F.adaptive_avg_pool2d(a, 1)
            v = F.adaptive_avg_pool3d(v, 1)

            a = torch.flatten(a, 1)
            v = torch.flatten(v, 1)

            a, v, out = self.fusion_module(a, v)  # av 是原来的，out是融合结果

            return a, v, out

        elif self.args.modality == 'visual':
            v = self.visual_net(visual)

            (_, C, H, W) = v.size()
            B = self.args.batch_size
            v = v.view(B, -1, C, H, W)
            # print(B)
            # print(B * C * H * W)
            v = v.permute(0, 2, 1, 3, 4)

            v = F.adaptive_avg_pool3d(v, 1)

            v = torch.flatten(v, 1)

            out = self.visual_classifier(v)

            a = torch.zeros_like(v)

            return a, v, out

        elif self.args.modality == 'audio':
            a = self.audio_net(audio)  # only feature

            a = F.adaptive_avg_pool2d(a, 1)

            a = torch.flatten(a, 1)

            out = self.audio_classifier(a)
            v = torch.zeros_like(a)

            return a, v, out
        else:
            return 0, 0 ,0


def _get_num_classes(dataset):
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


class AVClassifier_SDGL(nn.Module):
    def __init__(self, args):
        super(AVClassifier_SDGL, self).__init__()

        if args.modality != 'full':
            raise NotImplementedError('AVClassifier_SDGL only supports full modality.')

        fusion = args.fusion_method
        n_classes = _get_num_classes(args.dataset)

        if fusion == 'sum':
            self.fusion_module = SumFusion_SDGL(output_dim=n_classes)
        elif fusion == 'concat':
            if args.dataset == 'kinect400':
                self.fusion_module = ConcatFusion_SDGL(output_dim=n_classes, input_dim=1024)
            else:
                self.fusion_module = ConcatFusion_SDGL(output_dim=n_classes)
        elif fusion == 'film':
            self.fusion_module = FiLM_SDGL(output_dim=n_classes, x_film=True)
        elif fusion == 'gated':
            self.fusion_module = GatedFusion_SDGL(output_dim=n_classes, x_gate=True)
        else:
            raise NotImplementedError('Incorrect fusion method: {}!'.format(fusion))

        self.audio_net = resnet18(modality='audio', args=args)
        self.visual_net = resnet18(modality='visual', args=args)
        self.modality = args.modality
        self.args = args

    def forward(self, audio, visual):
        # a和v是特征
        a = self.audio_net(audio)
        v = self.visual_net(visual)
        # 处理视觉特征的时间维度
        (_, C, H, W) = v.size()
        B = a.size()[0]
        v = v.view(B, -1, C, H, W)
        v = v.permute(0, 2, 1, 3, 4)
        # 特征聚合
        a = F.adaptive_avg_pool2d(a, 1)
        v = F.adaptive_avg_pool3d(v, 1)
        # 展平
        a = torch.flatten(a, 1)
        v = torch.flatten(v, 1)
        # 输出结果调用融合模块进入fusion_modules
        a_out, v_out, out = self.fusion_module(a, v)
        return {
            'features': {
                'audio': a,
                'visual': v,
            },
            'logits': {
                'multi': out,
                'audio': a_out,
                'visual': v_out,
            }
        }

