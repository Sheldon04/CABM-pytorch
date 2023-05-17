import torch
import torch.nn as nn
import torch.nn.functional as F
from model.quant_ops import quant_act_pams

from option import args
device = torch.device('cpu' if args.cpu else f'cuda:{args.gpu_id}')
class BitSelector(nn.Module):
    def __init__(self, n_feats, bias=False, ema_epoch=1, search_space=[4,6,8], linq=False):
        super(BitSelector, self).__init__()
 
        self.quant_bit1 = quant_act_pams(k_bits=search_space[0], ema_epoch=ema_epoch)
        self.quant_bit2 = quant_act_pams(k_bits=search_space[1], ema_epoch=ema_epoch)
        self.quant_bit3 = quant_act_pams(k_bits=search_space[2], ema_epoch=ema_epoch)

        self.search_space =search_space
        self.take_index = None

        self.flag = None

        self.net_small = nn.Sequential(
            nn.Linear(n_feats+2, len(search_space)) 
        )
        nn.init.ones_(self.net_small[0].weight)
        nn.init.zeros_(self.net_small[0].bias)
        nn.init.ones_(self.net_small[0].bias[-1])

    def forward(self, x):
        if len(x) >= 4 and x[3] is not None:
            self.take_index = x[3]
        else:
            self.take_index = torch.arange(x[1].shape[0], device=x[1].device)
        bits = x[2]
        grad = x[0] 
        x = x[1]

        self.flag = self.flag.to(x.device)

        if len(self.search_space)== 4:
            bits_hard = (self.flag[self.take_index]==0)*self.search_space[0] + (self.flag[self.take_index]==1)*self.search_space[1] + (self.flag[self.take_index]==2)*self.search_space[2] + (self.flag[self.take_index]==3)*self.search_space[3]
            bits_out = bits_hard.detach()
            if not isinstance(bits, int):
                bits[self.take_index] += bits_out
            else:
                bits += bits_out

            q_bit1 = self.quant_bit1(x)
            q_bit2 = self.quant_bit2(x)
            q_bit3 = self.quant_bit3(x)
            q_bit4 = self.quant_bit4(x)
            out_hard = (self.flag[self.take_index]==0).view(self.flag[self.take_index].size(0),1,1,1)*q_bit1 + (self.flag[self.take_index]==1).view(self.flag[self.take_index].size(0),1,1,1)*q_bit2 + (self.flag[self.take_index]==2).view(self.flag[self.take_index].size(0),1,1,1)*q_bit3 + (self.flag[self.take_index]==3).view(self.flag[self.take_index].size(0),1,1,1)*q_bit4

            if args.test_only:
                residual = out_hard.detach()
            else: 
                residual = out_hard

        elif len(self.search_space)== 3:
            bits_hard = (self.flag[self.take_index]==0)*self.search_space[0] + (self.flag[self.take_index]==1)*self.search_space[1] + (self.flag[self.take_index]==2)*self.search_space[2]
            bits_out = bits_hard.detach()
            if not isinstance(bits, int):
                bits[self.take_index] += bits_out
            else:
                bits += bits_out

            q_bit1 = self.quant_bit1(x)
            q_bit2 = self.quant_bit2(x)
            q_bit3 = self.quant_bit3(x)
            out_hard = (self.flag[self.take_index]==0).view(self.flag[self.take_index].size(0),1,1,1)*q_bit1 + (self.flag[self.take_index]==1).view(self.flag[self.take_index].size(0),1,1,1)*q_bit2 + (self.flag[self.take_index]==2).view(self.flag[self.take_index].size(0),1,1,1)*q_bit3

            if args.test_only:
                residual = out_hard.detach()
            else: 
                residual = out_hard

        elif len(self.search_space)== 2:
            bits_hard = (self.flag[self.take_index]==0)*self.search_space[0] + (self.flag[self.take_index]==1)*self.search_space[1]
            bits_out = bits_hard.detach()
            if not isinstance(bits, int):
                bits[self.take_index] += bits_out
            else:
                bits += bits_out
            q_bit1 =self.quant_bit1(x)
            q_bit2 = self.quant_bit2(x)
            out_hard = (self.flag[self.take_index]==0).view(self.flag[self.take_index].size(0),1,1,1)*q_bit1 + (self.flag[self.take_index]==1).view(self.flag[self.take_index].size(0),1,1,1)*q_bit2
            if args.test_only:
                residual = out_hard.detach()
            else: 
                residual = out_hard
        
        return [grad, residual, bits]