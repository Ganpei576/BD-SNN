# YOLOv3 🚀 by Ultralytics, GPL-3.0 license
"""
YOLO-specific modules

Usage:
    $ python path/to/models/yolo.py --cfg yolov3.yaml
"""

import argparse
import sys
from copy import deepcopy
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
# ROOT = ROOT.relative_to(Path.cwd())  # relative

from models.common import *

from models.experimental import *
from utils.autoanchor import check_anchor_order
from utils.general import LOGGER, check_version, check_yaml, make_divisible, print_args
from utils.plots import feature_visualization
from utils.torch_utils import (copy_attr, fuse_conv_and_bn, initialize_weights, model_info, scale_img, select_device,
                               time_sync)

try:
    import thop  # for FLOPs computation
except ImportError:
    thop = None
time_window = 3


class Detect(nn.Module):
    stride = None  # strides computed during build
    onnx_dynamic = False  # ONNX export parameter

    def __init__(self, nc=80, anchors=(), ch=(), inplace=True):  # detection layer
        super().__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl  # init grid     [tensor([0.]), tensor([0.])]    存放预先计算好的网格偏移量
        self.anchor_grid = [torch.zeros(1)] * self.nl  # init anchor grid   [tensor([0.]), tensor([0.])]
        self.register_buffer('anchors', torch.tensor(anchors).float().view(self.nl, -1, 2))  # shape(nl,na,2)   nn.Module 类中定义的方法 它用于将一个张量注册为模块的一部分，但不会被视为模型参数（即不会通过优化器更新）。这在保存和加载模型状态时非常有用
        self.m = nn.ModuleList(Snn_Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv
        self.inplace = inplace  # use in-place ops (e.g. slice assignment)

    def forward(self, x):

        z = []  # inference output
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            times, bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(times, bs, self.na, self.no, ny, nx).permute(0, 1, 2, 4, 5,
                                                                          3).contiguous()  # 调整输入的形状，使其适合后续处理       这里用的是维度变换，相当于将中间的通道数拆成3*85，而保持特征图不变。

            x[i] = x[i].sum(dim=0) / x[i].size()[0]  # 在时间维度上求和并归一化   会丢失一个维度

            if not self.training:  # inference
                if self.onnx_dynamic or self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny,
                                                                        i)  # 最初的self.grid和anchor_grid应该是在热身阶段计算出并保存的
                    # grid：表示特征图上的坐标网格，用于确定每个锚框的中心位置。     anchor_grid：表示特征图上的锚框网格，用于在特征图的每个位置生成特定大小的锚框。
                y = x[i].sigmoid()
                if self.inplace:  # 执行这里
                    y[..., 0:2] = (y[..., 0:2] * 2 - 0.5 + self.grid[i]) * self.stride[
                        i]  # 这行代码调整预测框的中心坐标 (x, y)。首先，将 (x, y) 从 [0, 1] 映射到 [-0.5, 1.5]，然后加上对应的网格偏移，最后乘以 stride 以恢复到原始图像的尺度
                    # y[..., 0:2] 选择的是最后一维（即预测输出的特征）的前两个元素，这通常对应于预测框的中心坐标 (x, y)
                    y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[
                        i]  # 这行代码调整预测框的宽高 (w, h)。将 (w, h) 从 [0, 1] 映射到 [0, 4]，然后平方以确保宽高为正，再乘以对应的锚点网格，恢复到原始图像的尺度
                else:  # for  on AWS Inferentia https://github.com/ultralytics/yolov5/pull/2953
                    xy = (y[..., 0:2] * 2 - 0.5 + self.grid[i]) * self.stride[i]  # xy
                    wh = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                    y = torch.cat((xy, wh, y[..., 4:]), -1)
                z.append(y.view(bs, -1, self.no))

        return x if self.training else (torch.cat(z, 1), x)  # 返回训练模式下的输出 x 或推理模式下的拼接输出和原始输出

    def _make_grid(self, nx=20, ny=20, i=0):
        d = self.anchors[i].device
        if check_version(torch.__version__, '1.10.0'):  # torch>=1.10.0 meshgrid workaround for torch>=0.7 compatibility
            yv, xv = torch.meshgrid([torch.arange(ny).to(d), torch.arange(nx).to(d)], indexing='ij')
        else:
            yv, xv = torch.meshgrid([torch.arange(ny).to(d), torch.arange(nx).to(d)])
        grid = torch.stack((xv, yv), 2).expand((1, self.na, ny, nx, 2)).float()
        anchor_grid = (self.anchors[i].clone() * self.stride[i]) \
            .view((1, self.na, 1, 1, 2)).expand((1, self.na, ny, nx, 2)).float()
        return grid, anchor_grid


class Model(nn.Module):  # 加载我们传入的配置文件
    def __init__(self, cfg='yolov3.yaml', ch=3, nc=None, anchors=None):  # model, input channels, number of classes
        super().__init__()
        if isinstance(cfg, dict):
            self.yaml = cfg  # model dict
        else:  # is *.yaml执行这里
            import yaml  # for torch hub
            self.yaml_file = Path(cfg).name
            with open(cfg, encoding='ascii', errors='ignore') as f:
                self.yaml = yaml.safe_load(f)  # model dict

        # Define model      利用传入的配置文件搭建网络
        ch = self.yaml['ch'] = self.yaml.get('ch', ch)  # input channels 尝试从yaml中读取ch的值，如果没有就设置为ch并赋值给self.yaml['ch']
        if nc and nc != self.yaml['nc']:
            LOGGER.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml['nc'] = nc  # override yaml value
        if anchors:
            LOGGER.info(f'Overriding model.yaml anchors with anchors={anchors}')
            self.yaml['anchors'] = round(anchors)  # override yaml value    round 是一个 Python 内置函数，用于将一个数字四舍五入到最近的整数
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=[ch])  # model, savelist
        self.names = [str(i) for i in range(self.yaml['nc'])]  # default names 创建装类别名的列表
        self.inplace = self.yaml.get('inplace', True)

        # Build strides, anchors    求网络的步长（根据输入图片的尺寸以及最后要预测的尺寸做除法得出） 并对网络的anchor进行了一些处理
        m = self.model[-1]  # Detect()  取出网络的最后一层
        if isinstance(m, Detect):
            s = 256  # 2x min stride
            m.inplace = self.inplace
            m.stride = torch.tensor([s / x.shape[-2] for x in self.forward(
                torch.zeros(1, ch, s, s))])  # forward   用一张空白图片进行一次前向传播 采用列表推导式得到输出张量的形状，并取出高度(x.shape(-2))来计算步长
            m.anchors /= m.stride.view(-1, 1,
                                       1)  # 将anchor大小从原始图片映射到特征图   self.forward会返回两个两个特征图形状 所以m.stride是一个列表[16.,32.] 此处是将m.stride进行变形以方便进行广播
            check_anchor_order(m)  # 检测anchor的顺序是否正确
            self.stride = m.stride
            self._initialize_biases()  # only run once

        # Init weights, biases      初始化 打印信息
        initialize_weights(self)
        self.info()
        LOGGER.info('')

    def forward(self, x, augment=False, profile=False,
                visualize=False):  # augment表示是否要做数据增强 visualize表示是否保存中间特征图 profile 参数用于控制是否对模型的前向传播进行性能分析
        input = torch.zeros(time_window, x.size()[0], x.size()[1], x.size()[2], x.size()[3], device=x.device)
        for i in range(time_window):
            input[i] = x
        # 将输入复制3份，变为三个时间步，张量的维度从4维变为5维
        if augment:
            return self._forward_augment(x)  # augmented inference, None
        return self._forward_once(input, profile, visualize)  # single-scale inference, train

    def _forward_augment(self, x):
        img_size = x.shape[-2:]  # height, width
        s = [1, 0.83, 0.67]  # scales
        f = [None, 3, None]  # flips (2-ud, 3-lr)
        y = []  # outputs
        for si, fi in zip(s, f):
            xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
            yi = self._forward_once(xi)[0]  # forward
            # cv2.imwrite(f'img_{si}.jpg', 255 * xi[0].cpu().numpy().transpose((1, 2, 0))[:, :, ::-1])  # save
            yi = self._descale_pred(yi, fi, si, img_size)
            y.append(yi)
        y = self._clip_augmented(y)  # clip augmented tails
        return torch.cat(y, 1), None  # augmented inference, train

    def _forward_once(self, x, profile=False, visualize=False):  # 执行这里【1,256,16,16】
        y, dt = [], []  # outputs
        for m in self.model:
            if m.f != -1:  # 如果不是从上一层的输出中获得数据，那么如果m.f是个int，就从y列表中保存的值里取，else，构成一个列表，里面保存要用的数据
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            if profile:
                self._profile_one_layer(m, x, dt)
            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output m.i是第几层，self
            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)
        # print('=======')
        # print(x[0].shape)#torch.Size([32, 3, 40, 40, 85])
        # # #torch.Size([32, 3, 20, 20, 85])
        # print(x[1].shape)
        return x

    def _descale_pred(self, p, flips, scale, img_size):
        # de-scale predictions following augmented inference (inverse operation)
        if self.inplace:
            p[..., :4] /= scale  # de-scale
            if flips == 2:
                p[..., 1] = img_size[0] - p[..., 1]  # de-flip ud
            elif flips == 3:
                p[..., 0] = img_size[1] - p[..., 0]  # de-flip lr
        else:
            x, y, wh = p[..., 0:1] / scale, p[..., 1:2] / scale, p[..., 2:4] / scale  # de-scale
            if flips == 2:
                y = img_size[0] - y  # de-flip ud
            elif flips == 3:
                x = img_size[1] - x  # de-flip lr
            p = torch.cat((x, y, wh, p[..., 4:]), -1)
        return p

    def _clip_augmented(self, y):
        # Clip  augmented inference tails
        nl = self.model[-1].nl  # number of detection layers (P3-P5)
        g = sum(4 ** x for x in range(nl))  # grid points
        e = 1  # exclude layer count
        i = (y[0].shape[1] // g) * sum(4 ** x for x in range(e))  # indices
        y[0] = y[0][:, :-i]  # large
        i = (y[-1].shape[1] // g) * sum(4 ** (nl - 1 - x) for x in range(e))  # indices
        y[-1] = y[-1][:, i:]  # small
        return y

    def _profile_one_layer(self, m, x, dt):
        c = isinstance(m, Detect)  # is final layer, copy input as inplace fix
        o = thop.profile(m, inputs=(x.copy() if c else x,), verbose=False)[0] / 1E9 * 2 if thop else 0  # FLOPs
        t = time_sync()
        for _ in range(10):
            m(x.copy() if c else x)
        dt.append((time_sync() - t) * 100)
        if m == self.model[0]:
            LOGGER.info(f"{'time (ms)':>10s} {'GFLOPs':>10s} {'params':>10s}  {'module'}")
        LOGGER.info(f'{dt[-1]:10.2f} {o:10.2f} {m.np:10.0f}  {m.type}')
        if c:
            LOGGER.info(f"{sum(dt):10.2f} {'-':>10s} {'-':>10s}  Total")

    def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        m = self.model[-1]  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b.data[:, 5:] += math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def _print_biases(self):
        m = self.model[-1]  # Detect() module
        for mi in m.m:  # from
            b = mi.bias.detach().view(m.na, -1).T  # conv.bias(255) to (3,85)
            LOGGER.info(
                ('%6g Conv2d.bias:' + '%10.3g' * 6) % (mi.weight.shape[1], *b[:5].mean(1).tolist(), b[5:].mean()))

    # def _print_weights(self):
    #     for m in self.model.modules():
    #         if type(m) is Bottleneck:
    #             LOGGER.info('%10.3g' % (m.w.detach().sigmoid() * 2))  # shortcut weights

    def fuse(self):  # fuse model Conv2d() + BatchNorm2d() layers
        LOGGER.info('Fusing layers... ')
        for m in self.model.modules():
            if isinstance(m, (Conv, DWConv)) and hasattr(m, 'bn'):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                delattr(m, 'bn')  # remove batchnorm
                m.forward = m.forward_fuse  # update forward
        self.info()
        return self

    def autoshape(self):  # add AutoShape module
        LOGGER.info('Adding AutoShape... ')
        m = AutoShape(self)  # wrap model
        copy_attr(m, self, include=('yaml', 'nc', 'hyp', 'names', 'stride'), exclude=())  # copy attributes
        return m

    def info(self, verbose=False, img_size=640):  # print model information
        model_info(self, verbose, img_size)

    def _apply(self, fn):
        # Apply to(), cpu(), cuda(), half() to model tensors that are not parameters or registered buffers
        self = super()._apply(fn)
        m = self.model[-1]  # Detect()
        if isinstance(m, Detect):
            m.stride = fn(m.stride)
            m.grid = list(map(fn, m.grid))
            if isinstance(m.anchor_grid, list):
                m.anchor_grid = list(map(fn, m.anchor_grid))
        return self


def parse_model(d, ch):  # model_dict, input_channels(3)
    LOGGER.info(f"\n{'':>3}{'from':>18}{'n':>3}{'params':>10}  {'module':<40}{'arguments':<30}")
    anchors, nc, gd, gw = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple']
    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors
    no = na * (nc + 5)  # number of outputs = anchors * (classes + 5)

    layers, save, c2 = [], [], ch[-1]  # layers：负责存储创建网络的每一层, save：标签，统计哪一层的数据是需要保存的, c2：输出通道数
    for i, (f, n, m, args) in enumerate(
            d['backbone'] + d['head']):  # from, number, module, args   ****************遍历******************
        m = eval(m) if isinstance(m, str) else m  # eval strings       eval函数用于解析字符串，将字符串解析为对应地类名
        for j, a in enumerate(args):  # 将args中的参数从str转变为可用类型，（如果args是str类型的话）
            try:
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings
            except NameError:
                pass

        n = n_ = max(round(n * gd), 1) if n > 1 else n  # depth gain    根据深度因子决定模块数量
        if m in [Conv, GhostConv, Bottleneck, GhostBottleneck, SPP, SPPF, MixConv2d, Focus, CrossConv,
                 BottleneckCSP, C3, C3TR, C3SPP, C3Ghost, Conv_2,
                 BasicBlock, BD_Block, BasicBlock_2, BD_Block1, Conv_A,
                 ConcatBlock_ms, BasicBlock_ms, Conv_1, BD_Block2, Encode]:
            c1, c2 = ch[f], args[0]  # 输入通道，输出通道
            if c2 != no:  # if not output
                c2 = make_divisible(c2 * gw, 8)  # 输出通道乘上宽度缩放因子   make_divisible表示将c2 * gw调整为最接近能被8整除的数

            args = [c1, c2, *args[1:]]  # 拼接以得到完整的arg信息，原始的不完整
            if m in [BottleneckCSP, C3, C3TR, C3Ghost]:
                args.insert(2, n)  # number of repeats
                n = 1
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum(ch[x] for x in f)
        elif m is Detect:
            args.append([ch[x] for x in f])  # 此时f中存放的是索引，ch[x] for x in f表示将第14层和第10层的通道数加进args中
            if isinstance(args[1], int):  # number of anchors   没有执行
                args[1] = [list(range(args[1] * 2))] * len(f)
        elif m is Contract:
            c2 = ch[f] * args[0] ** 2
        elif m is Expand:
            c2 = ch[f] // args[0] ** 2
        else:
            c2 = ch[f]

        m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(
            *args)  # module 根据n的数量来决定搭建几个模块的层  如果n>1就以列表创建式创建一个存储对应模块的列表，然后再用*解列表后输入m_   否则直接创建对应模块
        t = str(m)[8:-2].replace('__main__.', '')  # module type        判断模块字符串里有没有__main__.，若有则将其替换
        np = sum(x.numel() for x in m_.parameters())  # number params               统计参数量
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
        LOGGER.info(f'{i:>3}{str(f):>18}{n_:>3}{np:10.0f}  {t:<40}{str(args):<30}')  # print    打印
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if
                    x != -1)  # append to savelist     [9, 6, 14, 10] yaml中第一个值不是-1的就会进行计算
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)  # 使下一层可以取出上一层的输出通道数
    return nn.Sequential(*layers), sorted(save)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='resnet18.yaml', help='model.yaml')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--profile', action='store_true', help='profile model speed')
    parser.add_argument('--test', action='store_true', help='test all yolo*.yaml')
    opt = parser.parse_args()
    opt.cfg = check_yaml(opt.cfg)  # check YAML
    print_args(FILE.stem, opt)
    device = select_device(opt.device)

    # Create model
    model = Model(opt.cfg).to(device)
    # model.train()

    # Profile
    if opt.profile:
        img = torch.rand(8 if torch.cuda.is_available() else 1, 3, 640, 640).to(device)
        y = model(img, profile=True)

    # Test all models
    if opt.test:
        for cfg in Path(ROOT / 'models').rglob('yolo*.yaml'):
            try:
                _ = Model(cfg)
            except Exception as e:
                print(f'Error in {cfg}: {e}')
