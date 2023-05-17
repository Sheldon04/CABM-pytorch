import torch
import torch.nn as nn
import torch.nn.functional as F
from model.quant_ops import quant_act_pams


class BitSelector(nn.Module):
    def __init__(self, n_feats, bias=False, ema_epoch=1, search_space=[2,4,6,8], linq=False):
        super(BitSelector, self).__init__()
 
        self.quant_bit1 = quant_act_pams(k_bits=search_space[0], ema_epoch=ema_epoch)
        self.quant_bit2 = quant_act_pams(k_bits=search_space[1], ema_epoch=ema_epoch)
        self.quant_bit3 = quant_act_pams(k_bits=search_space[2], ema_epoch=ema_epoch)
        # self.quant_bit4 = quant_act_pams(k_bits=search_space[3], ema_epoch=ema_epoch)

        self.search_space =search_space

        self.bits_out = 8
        
        self.net_small = nn.Sequential(
            nn.Linear(n_feats+2, len(search_space)) 
        )
        nn.init.ones_(self.net_small[0].weight)
        nn.init.zeros_(self.net_small[0].bias)
        nn.init.ones_(self.net_small[0].bias[-1])

    def forward(self, x):
        weighted_bits = x[3]
        bits = x[2]
        grad = x[0] 
        x = x[1]
        # print('x: ' + str(x.size()))
        layer_std_s = torch.std(x, (2,3)).detach() # remove dim 2,3
        # print('layer_std: ' + str(layer_std_s.size()))
        # print('grad: ' + str(grad.size()))
        x_embed = torch.cat([grad, layer_std_s], dim=1) #[B, C+2] 
        # print('x_embed: ' + str(x_embed.size()))

        bit_type = self.net_small(x_embed) 
        # print('bit_type: ' + str(bit_type))
        flag = torch.argmax(bit_type, dim=1)
        #[1,0]
        p = F.softmax(bit_type, dim=1)
        # print('flag: ' + str(flag.size()))
        if len(self.search_space)== 4:
            p1 = p[:,0]
            # print("p1: " + str(p1))
            # print("p1.view: " + str(p1.view(p1.size(0),1,1,1)))
            p2 = p[:,1]
            p3 = p[:,2]
            p4 = p[:,3]
            bits_hard = (flag==0)*self.search_space[0] + (flag==1)*self.search_space[1] + (flag==2)*self.search_space[2] + (flag==3)*self.search_space[3]
            bits_soft = p1*self.search_space[0]+p2*self.search_space[1]+ p3*self.search_space[2] + p4*self.search_space[3]
            bits_out = bits_hard.detach() - bits_soft.detach() + bits_soft
            self.bits_out = bits_out
            bits += bits_out
            weighted_bits += bits_out / (self.search_space[0]*p1.detach()+self.search_space[1]*p2.detach()+self.search_space[2]*p3.detach()+self.search_space[3]*p4.detach())

            q_bit1 = self.quant_bit1(x)
            q_bit2 = self.quant_bit2(x)
            q_bit3 = self.quant_bit3(x)
            q_bit4 = self.quant_bit4(x)
            out_soft = p1.view(p1.size(0),1,1,1)*q_bit1 + p2.view(p2.size(0),1,1,1)*q_bit2 + p3.view(p3.size(0),1,1,1)*q_bit3 + p4.view(p4.size(0),1,1,1)*q_bit4
            out_hard = (flag==0).view(flag.size(0),1,1,1)*q_bit1 + (flag==1).view(flag.size(0),1,1,1)*q_bit2 + (flag==2).view(flag.size(0),1,1,1)*q_bit3 + (flag==3).view(flag.size(0),1,1,1)*q_bit4
            residual = out_hard.detach() - out_soft.detach() + out_soft
        elif len(self.search_space)== 3:
            p1 = p[:,0]
            # print("p1: " + str(p1))
            # print("p1.view: " + str(p1.view(p1.size(0),1,1,1)))
            p2 = p[:,1]
            p3 = p[:,2]
            bits_hard = (flag==0)*self.search_space[0] + (flag==1)*self.search_space[1] + (flag==2)*self.search_space[2]
            # print(bits_hard)
            bits_soft = p1*self.search_space[0]+p2*self.search_space[1]+ p3*self.search_space[2]
            # print(bits_soft)
            bits_out = bits_hard.detach() - bits_soft.detach() + bits_soft
            self.bits_out = bits_out
            bits += bits_out
            weighted_bits += bits_out / (self.search_space[0]*p1.detach()+self.search_space[1]*p2.detach()+self.search_space[2]*p3.detach())
            # print(weighted_bits)
            q_bit1 = self.quant_bit1(x)
            q_bit2 = self.quant_bit2(x)
            q_bit3 = self.quant_bit3(x)
            out_soft = p1.view(p1.size(0),1,1,1)*q_bit1 + p2.view(p2.size(0),1,1,1)*q_bit2 + p3.view(p3.size(0),1,1,1)*q_bit3
            out_hard = (flag==0).view(flag.size(0),1,1,1)*q_bit1 + (flag==1).view(flag.size(0),1,1,1)*q_bit2 + (flag==2).view(flag.size(0),1,1,1)*q_bit3
            residual = out_hard.detach() - out_soft.detach() + out_soft

        elif len(self.search_space)== 2:
            p1 = p[:,0]
            p2 = p[:,1]
            bits_hard = (flag==0)*self.search_space[0] + (flag==1)*self.search_space[1]
            bits_soft = p1*self.search_space[0]+p2*self.search_space[1]
            bits_out = bits_hard.detach() - bits_soft.detach() + bits_soft
            self.bits_out = bits_out
            bits += bits_out
            weighted_bits += bits_out / (self.search_space[0]*p1.detach()+self.search_space[1]*p2.detach())
            q_bit1 =self.quant_bit1(x)
            q_bit2 = self.quant_bit2(x)
            out_soft = p1.view(p1.size(0),1,1,1)*q_bit1 + p2.view(p2.size(0),1,1,1)*q_bit2 
            out_hard = (flag==0).view(flag.size(0),1,1,1)*q_bit1 + (flag==1).view(flag.size(0),1,1,1)*q_bit2
            residual = out_hard.detach() - out_soft.detach() + out_soft
        
        # return [grad, residual, bits, weighted_bits]
        return [grad, residual, bits, weighted_bits]