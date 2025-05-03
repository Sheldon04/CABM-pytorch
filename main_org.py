
import json
from decimal import Decimal

import cv2
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

import data
import utility

from model.carn import CARN
from model.edsr import EDSR
from model.idn import IDN
from model.srresnet import SRResNet

import torch.nn as nn
from option import args
from utils import common as util
from utils.common import AverageMeter
import torch.nn.parallel as P
import numpy as np

import time
from torchvision.utils import save_image

torch.manual_seed(args.seed)
checkpoint = utility.checkpoint(args)
device = torch.device('cpu' if args.cpu else f'cuda:{args.gpu_id}')


class Trainer():
    def __init__(self, args, loader, t_model, s_model, ckp):
        self.args = args
        self.scale = args.scale

        self.epoch = 0
        self.ckp = ckp
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test
        self.t_model = t_model
        self.s_model = s_model


        if args.model == 'EDSR' or args.model == 'SRResNet':
            arch_param = [v for k, v in self.t_model.named_parameters(
            ) if 'alpha' not in k and 'net' not in k]
            alpha_param = [
                v for k, v in self.t_model.named_parameters() if 'alpha' in k]

        else:
            alpha_param = [
                v for k, v in self.t_model.named_parameters() if 'alpha' in k]
            arch_param = [v for k, v in self.t_model.named_parameters(
            ) if 'alpha' not in k and 'net' not in k]

        params = [{'params': arch_param}, {'params': alpha_param, 'lr': 1e-2}]
        self.optimizer = torch.optim.Adam(
            params, lr=args.lr, betas=args.betas, eps=args.epsilon)
        self.scheduler = StepLR(
            self.optimizer, step_size=int(args.decay), gamma=args.gamma)

        self.resume_epoch = 0
        if args.resume is not None:
            ckpt = torch.load(args.resume)
            self.epoch = ckpt['epoch']
            print(f"Continue from {self.epoch}")
            self.t_model.load_state_dict(ckpt['state_dict'])
            self.optimizer.load_state_dict(ckpt['optimizer'])
            self.scheduler.load_state_dict(ckpt['scheduler'])
            self.resume_epoch = ckpt['epoch']
            # self.epoch -= self.resume_epoch

        # --------------- Print Model ---------------------
        if args.test_only:
            self.ckp.write_log('Test on {}'.format(args.teacher_weights))

        # --------------- Print # Params ---------------------
        n_params = 0
        for p in list(s_model.parameters()):
            n_p = 1
            for s in list(p.size()):
                n_p = n_p*s
            n_params += n_p
        self.ckp.write_log('Parameters: {:.1f}K'.format(n_params/(1e+3)))

        self.losses = AverageMeter()
        self.att_losses = AverageMeter()
        self.nor_losses = AverageMeter()
        self.bit_losses = AverageMeter()
        self.avg_bit = AverageMeter()

        self.test_patch_size = args.patch_size
        self.step_size = args.step_size

        self.mse_loss = nn.MSELoss()

        self.losses_list = []
        self.bit_list = []
        self.valpsnr_list = []
        self.valbit_list = []

    def train(self):
        self.scheduler.step()
        # self.cadyq_scheduler.step()

        self.epoch = self.epoch + 1
        lr = self.optimizer.state_dict()['param_groups'][0]['lr']
        # bitsel_lr = self.cadyq_optimizer.state_dict()['param_groups'][0]['lr']

        self.w_bit = self.epoch*self.args.w_bit_decay + \
            self.args.w_bit if self.args.cadyq else self.args.w_bit

        self.ckp.write_log(
            '[Epoch {}]\tLearning rate: {:.2e}'.format(
                self.epoch, Decimal(lr))
        )

        self.t_model.train()
        # self.s_model.train()

        self.t_model.apply(lambda m: setattr(m, 'epoch', self.epoch))

        num_iterations = len(self.loader_train)
        timer_data, timer_model = utility.timer(), utility.timer()


        for batch, (lr, hr, idx_scale, ) in enumerate(self.loader_train):
            num_iters = num_iterations * (self.epoch-1) + batch

            lr, hr = self.prepare(lr, hr)

            data_size = lr.size(0)

            timer_data.hold()
            timer_model.tic()
            
            self.optimizer.zero_grad()

            if hasattr(self.t_model, 'set_scale'):
                self.t_model.set_scale(idx_scale)
            if hasattr(self.s_model, 'set_scale'):
                self.s_model.set_scale(idx_scale)

            # Teacher
            if self.args.model == 'CARN':
                t_sr  = self.t_model(lr/255., self.scale[0])
                t_sr *=255.
            else:
                # t_sr, t_res, _, t_feat, _ = self.t_model(lr)
                t_sr = self.t_model(lr)

            # 1. Pixel-wise L1 loss
            if self.args.model=='FSRCNN':
                nor_loss = self.mse_loss(t_sr, hr)
            else:
                nor_loss = args.w_l1 * F.l1_loss(t_sr, hr)            
            loss = nor_loss
    
            loss.backward()
            self.optimizer.step()

            timer_model.hold()
            
            self.losses.update(loss.item(), data_size)
            self.nor_losses.update(nor_loss.item(), data_size)

            display_loss = f'Loss: {self.losses.avg: .3f}'
            display_loss_nor = f'L_1: {self.nor_losses.avg: .3f}'

            if (batch + 1) % self.args.print_every == 0:
                self.ckp.write_log('[{}/{}] \t{:.1f}+{:.1f}s+ \t{}'.format(
                    (batch + 1) * self.args.batch_size,
                    len(self.loader_train.dataset),
                    timer_model.release(),
                    timer_data.release(),
                    display_loss_nor
                ))
                self.losses_list.append(self.losses.avg)

            timer_data.tic()

    def test(self, is_teacher=True):
        torch.set_grad_enabled(False)
        epoch = self.epoch
        self.ckp.write_log('\nEvaluation:')
        self.ckp.add_log(
            torch.zeros(1, len(self.loader_test), len(self.scale))
        )
        if is_teacher:
            model = self.t_model
        else:
            model = self.s_model
        model.eval()
        timer_test = utility.timer()

        if self.args.save_results:
            self.ckp.begin_background()

        for idx_data, d in enumerate(self.loader_test):
            for idx_scale, scale in enumerate(self.scale):
                if self.args.test_patch:
                    # ------------------------Test patch-wise------------------------------
                    # Check options : --test_patch --patch_size 128 --step_size 16 --student_weights STUDENT_MODEL_DIRECTORY
                    d.dataset.set_scale(idx_scale)
                    i = 0
                    tot_bits = 0
                    for lr, hr, filename in tqdm(d, ncols=80):
                        i += 1
                        lr, hr = self.prepare(lr, hr)

                        print(lr.size())
                        print(lr[0].size())

                        lr_list, num_h, num_w, h, w = self.crop(
                            lr[0], self.test_patch_size, self.step_size)
                        hr_list = self.crop(
                            hr[0], self.test_patch_size*self.args.scale[0], self.step_size*self.args.scale[0])[0]
                        sr_list = []

                        p = 0
                        tot_bits_image = 0
                        psnrs = []

                        for lr_sub_img, hr_sub_img in zip(lr_list, hr_list):
                            time_start = time.time()
                            # --------------------select which quantization to pass through---------------------
                            if self.args.model == 'CARN':
                                sr_sub = model(
                                    lr_sub_img.unsqueeze(0)/255., scale)
                                sr_sub *= 255.
                            else:
                                sr_sub = model(
                                    lr_sub_img.unsqueeze(0))
                            time_end = time.time()
                            # print('\n')
                            # print(get_bit_config(model))

                            avg_bit = 32.00
                            tot_bits_image += avg_bit

                            patch_psnr = utility.calc_psnr(
                                sr_sub, hr_sub_img, scale, self.args.rgb_range, dataset=d)
                            psnrs.append(patch_psnr)
                            self.ckp.write_log(
                                '{}-{:3d}: {:.2f} dB, {:.2f} avg bits'.format(filename[0], p, patch_psnr, avg_bit))

                            if self.args.save_patch:
                                save_image(sr_sub[0]/255, './experiment/'+self.args.save+'/results-'+self.args.data_test[0] +
                                           '/{}_{}_{:.2f}_{:.2f}.png'.format(filename[0], p, patch_psnr, avg_bit))

                            sr_sub = utility.quantize(
                                sr_sub, self.args.rgb_range)
                            sr_list.append(sr_sub)
                            p += 1

                        sr = self.combine(
                            sr_list, num_h, num_w, h, w, self.test_patch_size, self.step_size)
                        sr = sr.unsqueeze(0)

                        save_list = [sr]
                        if self.args.add_mask:
                            sr_mask = util.add_mask_psnr(sr.cpu(), scale, num_h, num_w, h*scale, w*scale, self.test_patch_size, self.step_size, psnrs)
                            save_list.append(sr_mask)
                        cur_psnr = utility.calc_psnr(
                            sr, hr, scale, self.args.rgb_range, dataset=d)
                        cur_ssim = utility.calc_ssim(
                            sr, hr, scale, benchmark=d.dataset.benchmark)

                        self.ckp.log[-1, idx_data, idx_scale] += cur_psnr
                        self.ckp.bit_log[-1, idx_data,
                                         idx_scale] += tot_bits_image/p
                        self.ckp.ssim_log[-1, idx_data, idx_scale] += cur_ssim

                        tot_bits += tot_bits_image/p
                        # per image
                        self.ckp.write_log(
                            '\n[{}] PSNR: {:.3f} dB; SSIM: {:.3f}; Avg_bit: {:.2f}; Num_patch: {}'.format(
                                filename[0],
                                cur_psnr,
                                cur_ssim,
                                tot_bits_image/p,
                                p
                            )
                        )

                        if self.args.save_gt:
                            save_list.extend([lr, hr])

                        if self.args.save_results:
                            save_name = '{}_{:.2f}'.format(
                                filename[0], cur_psnr)
                            self.ckp.save_results(
                                d, save_name, save_list, scale)

                    self.ckp.log[-1, idx_data, idx_scale] /= len(d)
                    self.ckp.ssim_log[-1, idx_data, idx_scale] /= len(d)

                    best_psnr = self.ckp.log.max(0)

                    self.ckp.write_log(
                            '[{} x{}] PSNR: {:.3f} SSIM:{:.3f} (Best: {:.3f} @epoch {})'.format(
                                d.dataset.name,
                                scale,
                                self.ckp.log[-1, idx_data, idx_scale],
                                self.ckp.ssim_log[-1, idx_data, idx_scale],
                                best_psnr[0][idx_data, idx_scale],
                                best_psnr[1][idx_data, idx_scale] + 1,
                            )
                        )

                else:
                    # ------------------------Test image-wise------------------------------
                    d.dataset.set_scale(idx_scale)
                    i = 0

                    tot_bits = 0
                    pbar = tqdm(d, ncols=80)
                    for lr, hr, filename in pbar:
                        i += 1
                        lr, hr = self.prepare(lr, hr)

                        if self.args.precision == 'half':
                            model = model.half()
                        if self.args.chop:
                            sr, s_res = self.forward_chop(lr)
                        else:
                            if self.args.model.lower() == 'fsrcnn':
                                sr = model(lr)
                            elif self.args.model == 'IDN':
                                sr = model(lr)
                            elif self.args.model == 'CARN':
                                sr = model(
                                    lr/255., scale)  # for CARN
                                sr *= 255.  # for CARN
                            else:
                                # EDSR, SRResNet
                                sr = model(lr)


                        sr = utility.quantize(sr, self.args.rgb_range)
                        save_list = [sr]

                        cur_psnr = utility.calc_psnr(
                            sr, hr, scale, self.args.rgb_range, dataset=d)
                        if self.args.test_only:
                            cur_ssim = utility.calc_ssim(
                                sr, hr, scale, benchmark=d.dataset.benchmark)
                        else:
                            cur_ssim = 0

                        self.ckp.log[-1, idx_data, idx_scale] += cur_psnr
                        self.ckp.ssim_log[-1, idx_data, idx_scale] += cur_ssim

                        if self.args.save_gt:
                            save_list.extend([lr, hr])

                        if self.args.save_results:
                            save_name = f'{filename[0]}_{args.k_bits}bit' + \
                                '_{:.2f}'.format(cur_psnr)
                            self.ckp.save_results(
                                d, save_name, save_list, scale)

                    self.ckp.log[-1, idx_data, idx_scale] /= len(d)
                    self.ckp.ssim_log[-1, idx_data, idx_scale] /= len(d)

                    best_psnr = self.ckp.log.max(0)
                    self.ckp.write_log(
                            '[{} x{}] PSNR: {:.3f} (Best: {:.3f} @epoch {})'.format(
                                d.dataset.name,
                                scale,
                                self.ckp.log[-1, idx_data, idx_scale],
                                best_psnr[0][idx_data, idx_scale],
                                best_psnr[1][idx_data, idx_scale] + 1,
                            )
                        )

        if self.args.save_results:
            self.ckp.end_background()

        if not self.args.test_only:
            is_best_psnr = (best_psnr[1][0, 0] + 1 == epoch)

            state = {
                'epoch': epoch,
                'state_dict': self.s_model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'scheduler': self.scheduler.state_dict(),
            }
            util.save_checkpoint(state, is_best_psnr,
                                 checkpoint=self.ckp.dir + '/model')
            util.plot_psnr(self.args, self.ckp.dir, self.epoch -
                           self.resume_epoch, self.ckp.log)
            util.plot_bit(self.args, self.ckp.dir, self.epoch -
                          self.resume_epoch, self.ckp.bit_log)  # in utils/common.py

        self.ckp.write_log(
            'Total: {:.2f}s\n'.format(timer_test.toc()), refresh=True
        )

        torch.set_grad_enabled(True)

    def prepare(self, *args):
        # device = torch.device('cpu' if self.args.cpu else 'cuda')
        def _prepare(tensor):
            if self.args.precision == 'half':
                tensor = tensor.half()
            return tensor.to(device)

        return [_prepare(a) for a in args]

    def terminate(self):
        if self.args.test_only:
            self.test()
            return True
        else:
            return self.epoch >= self.args.epochs

    def forward_chop(self, *args, shave=10, min_size=160000):
        # min_size : 400 x 400
        scale = self.scale[0]
        n_GPUs = min(self.args.n_GPUs, 4)
        # height, width
        h, w = args[0].size()[-2:]

        top = slice(0, h//2 + shave)
        bottom = slice(h - h//2 - shave, h)
        left = slice(0, w//2 + shave)
        right = slice(w - w//2 - shave, w)
        x_chops = [torch.cat([
            a[..., top, left],
            a[..., top, right],
            a[..., bottom, left],
            a[..., bottom, right]
        ]) for a in args]

        y_chops = []
        if h * w < 4 * min_size:
            for i in range(0, 4, n_GPUs):

                x = [x_chop[i:(i + n_GPUs)] for x_chop in x_chops]
                y, y_res = P.data_parallel(self.s_model, *x, range(n_GPUs))

                if not isinstance(y, list):
                    y = [y]
                if not y_chops:
                    y_chops = [[c for c in _y.chunk(
                        n_GPUs, dim=0)] for _y in y]
                else:
                    for y_chop, _y in zip(y_chops, y):
                        y_chop.extend(_y.chunk(n_GPUs, dim=0))

        else:
            for p in zip(*x_chops):

                y, y_res = self.forward_chop(
                    p[0].unsqueeze(0), shave=shave, min_size=min_size)

                if not isinstance(y, list):
                    y = [y]
                if not y_chops:
                    y_chops = [[_y] for _y in y]
                else:
                    for y_chop, _y in zip(y_chops, y):
                        y_chop.append(_y)

        h *= scale
        w *= scale
        top = slice(0, h//2)
        bottom = slice(h - h//2, h)
        bottom_r = slice(h//2 - h, None)
        left = slice(0, w//2)
        right = slice(w - w//2, w)
        right_r = slice(w//2 - w, None)

        b, c = y_chops[0][0].size()[:-2]
        y = [y_chop[0].new(b, c, h, w) for y_chop in y_chops]
        for y_chop, _y in zip(y_chops, y):
            _y[..., top, left] = y_chop[0][..., top, left]
            _y[..., top, right] = y_chop[1][..., top, right_r]
            _y[..., bottom, left] = y_chop[2][..., bottom_r, left]
            _y[..., bottom, right] = y_chop[3][..., bottom_r, right_r]

        if len(y) == 1:
            y = y[0]

        return y, y_res

    def crop(self, img, crop_sz, step):
        n_channels = len(img.shape)
        if n_channels == 2:
            h, w = img.shape
        elif n_channels == 3:
            c, h, w = img.shape
        else:
            raise ValueError('Wrong image shape - {}'.format(n_channels))

        h_space = np.arange(0, max(h - crop_sz, 0) + 1, step)
        w_space = np.arange(0, max(w - crop_sz, 0) + 1, step)
        index = 0
        num_h = 0
        lr_list = []
        for x in h_space:
            num_h += 1
            num_w = 0
            for y in w_space:
                num_w += 1
                index += 1
                if n_channels == 2:
                    crop_img = img[x:x + crop_sz, y:y + crop_sz]
                else:
                    if x == h_space[-1]:
                        if y == w_space[-1]:
                            crop_img = img[:, x:h, y:w]
                        else:
                            crop_img = img[:, x:h, y:y + crop_sz]
                    elif y == w_space[-1]:
                        crop_img = img[:, x:x + crop_sz, y:w]
                    else:
                        crop_img = img[:, x:x + crop_sz, y:y + crop_sz]
                lr_list.append(crop_img)
        return lr_list, num_h, num_w, h, w

    def combine(self, sr_list, num_h, num_w, h, w, patch_size, step):
        index = 0

        sr_img = torch.zeros((3, h*self.scale[0], w*self.scale[0])).to(device)
        s = int(((patch_size - step) / 2)*self.scale[0])
        index1 = 0
        index2 = 0
        if num_h == 1:
            if num_w == 1:
                sr_img[:, :h*self.scale[0], :w *
                       self.scale[0]] += sr_list[index][0]
            else:
                for j in range(num_w):
                    y0 = j*step*self.scale[0]
                    if j == 0:
                        sr_img[:, :, y0:y0+s+step*self.scale[0]
                               ] += sr_list[index1][0][:, :, :s+step*self.scale[0]]
                    elif j == num_w-1:
                        sr_img[:, :, y0+s:w*self.scale[0]
                               ] += sr_list[index1][0][:, :, s:]
                    else:
                        sr_img[:, :, y0+s:y0+s+step*self.scale[0]
                               ] += sr_list[index1][0][:, :, s:s+step*self.scale[0]]
                    index1 += 1

        elif num_w == 1:
            for i in range(num_h):
                x0 = i*step*self.scale[0]
                if i == 0:
                    sr_img[:, x0:x0+s+step*self.scale[0],
                           :] += sr_list[index2][0][:, :s+step*self.scale[0], :]
                elif i == num_h-1:
                    sr_img[:, x0+s:h*self.scale[0],
                           :] += sr_list[index2][0][:, s:, :]
                else:
                    sr_img[:, x0+s:x0+s+step*self.scale[0],
                           :] += sr_list[index2][0][:, s:s+step*self.scale[0], :]
                index2 += 1

        else:
            for i in range(num_h):
                for j in range(num_w):
                    x0 = i*step*self.scale[0]
                    y0 = j*step*self.scale[0]

                    if i == 0:
                        if j == 0:
                            sr_img[:, x0:x0+s+step*self.scale[0], y0:y0+s+step*self.scale[0]
                                   ] += sr_list[index][0][:, :s+step*self.scale[0], :s+step*self.scale[0]]
                        elif j == num_w-1:
                            sr_img[:, x0:x0+s+step*self.scale[0], y0+s:w*self.scale[0]
                                   ] += sr_list[index][0][:, :s+step*self.scale[0], s:]
                        else:
                            sr_img[:, x0:x0+s+step*self.scale[0], y0+s:y0+s+step*self.scale[0]
                                   ] += sr_list[index][0][:, :s+step*self.scale[0], s:s+step*self.scale[0]]
                    elif j == 0:
                        if i == num_h-1:
                            sr_img[:, x0+s:h*self.scale[0], y0:y0+s+step*self.scale[0]
                                   ] += sr_list[index][0][:, s:, :s+step*self.scale[0]]
                        else:
                            sr_img[:, x0+s:x0+s+step*self.scale[0], y0:y0+s+step*self.scale[0]
                                   ] += sr_list[index][0][:, s:s+step*self.scale[0], :s+step*self.scale[0]]
                    elif i == num_h-1:
                        if j == num_w-1:
                            sr_img[:, x0+s:h*self.scale[0], y0+s:w *
                                   self.scale[0]] += sr_list[index][0][:, s:, s:]
                        else:
                            sr_img[:, x0+s:h*self.scale[0], y0+s:y0+s+step*self.scale[0]
                                   ] += sr_list[index][0][:, s:, s:s+step*self.scale[0]]
                    elif j == num_w-1:
                        sr_img[:, x0+s:x0+s+step*self.scale[0], y0+s:w*self.scale[0]
                               ] += sr_list[index][0][:, s:s+step*self.scale[0], s:]
                    else:
                        sr_img[:, x0+s:x0+s+step*self.scale[0], y0+s:y0+s+step*self.scale[0]
                               ] += sr_list[index][0][:, s:s+step*self.scale[0], s:s+step*self.scale[0]]

                    index += 1

        return sr_img


def main():
    if checkpoint.ok:
        loader = data.Data(args)
        if args.model == 'CARN':
            t_model = CARN(args, multi_scale=True, is_teacher=False).to(device)
        elif args.model == 'EDSR':
            t_model = EDSR(args,is_teacher=False).to(device)
        elif args.model == 'IDN':
            t_model = IDN(args, is_teacher=False).to(device)

        elif args.model == 'SRResNet':
            t_model = SRResNet(
                args, is_teacher=False).to(device)

        else:
            raise ValueError('not expected model = {}'.format(args.model))

        if args.teacher_weights is not None:
            if args.test_only:
                ckpt = torch.load(f'{args.teacher_weights}')
                t_checkpoint = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
                t_model.load_state_dict(t_checkpoint)

        t = Trainer(args, loader, t_model, t_model, checkpoint)

        print(f'{args.save} start!')
        while not t.terminate():
            t.train()
            t.test()
        checkpoint.done()
        print(f'{args.save} done!')


if __name__ == '__main__':
    main()
