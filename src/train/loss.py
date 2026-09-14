from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch import nn
from torchvision.transforms import Resize
from merlinth.layers.fft import fft2c
from utils import warp_non_rigid_flow, warp_rigid_flow


class CriterionBase(nn.Module):
    def __init__(self, config):
        super(CriterionBase, self).__init__()
        self.loss_names = config.which
        self.loss_weights = config.loss_weights
        self.loss_list = []
        for loss_name in config.which:
            loss_args = eval(f'config.{loss_name}').__dict__
            loss_item = self.get_loss(loss_name=loss_name, args_dict=loss_args)
            self.loss_list.append(loss_item)

    def get_loss(self, loss_name, args_dict):
        if loss_name == 'photometric_weighted':
            return PhotometricLossWeighted(**args_dict)
        elif loss_name == 'smooth':
            return SmoothLoss(**args_dict)
        else:
            raise NotImplementedError


class LAPANetLoss2D(CriterionBase, nn.Module):
    def __init__(self, config):
        super().__init__(config=config)

    def forward(self, flow_pred=None, ref=None, mov=None, shift=None, box=None):
        loss_dict = {}
        total_loss = 0
        mse = nn.MSELoss()
        loss_weights=[0.05,0.15,0.2,0.6]

        for loss_name, loss_weight, loss_term in zip(self.loss_names, self.loss_weights, self.loss_list):
            if loss_name == 'photometric_weighted':
                if torch.is_tensor(flow_pred):
                    img_warped = warp_non_rigid_flow(mov, flow_pred)
                    loss_dict['photometric_multi_coil'] = loss_term(ref, img_warped, box)

                    img_warped2 = warp_non_rigid_flow(torch.sum(mov,1,keepdim=True), flow_pred)
                    loss_dict['photometric_single_coil'] = loss_term(torch.sum(ref,1,keepdim=True), img_warped2, box)

                    if torch.is_tensor(shift):
                        trans_warped = warp_rigid_flow(img_warped2, shift)
                        loss_dict['translation_shift_loss'] = loss_term(torch.sum(ref,1,keepdim=True), trans_warped, box)
                        ###
                        trans_flow = torch.ones_like(flow_pred) * shift.unsqueeze(2).unsqueeze(3)
                        trans_flow += flow_pred
                        img_warped_trans = warp_non_rigid_flow(mov, trans_flow)
                        loss_dict['photometric_shift_loss'] = loss_term(ref, img_warped_trans, box)


                    k_ref = fft2c(ref)
                    k_warped = fft2c(img_warped)

                    k_ref =   torch.abs(k_ref)
                    k_warped = torch.abs(k_warped)
                    loss_dict['k_photometric_loss'] =  mse(k_warped, k_ref)
                    i_loss = 0.01 * loss_dict['k_photometric_loss']  \
                             + loss_dict['photometric_shift_loss'] \
                             + 0.5 * loss_dict['translation_shift_loss'] \
                             + loss_dict['photometric_single_coil'] \
                             + loss_dict['photometric_multi_coil']

                else:
                    i_loss = 0.0
                    for i, flo in enumerate(flow_pred):
                        # print(flo.shape)
                        if flo[-1].shape != ref[-1].shape:
                            mov_i = Resize(flo.shape[-2:])(mov)
                            ref_i = Resize(flo.shape[-2:])(ref)
                            box_i = Resize(flo.shape[-2:])(box)
                        else:
                            mov_i = mov
                            ref_i = ref

                        img_warped = warp_non_rigid_flow(mov_i, flo)
                        i_loss += loss_weights[i] * loss_term(ref_i, img_warped, box_i)


            elif loss_name == 'smooth':
                if torch.is_tensor(flow_pred):
                    i_loss = loss_term(flow_pred, ref)
                else:
                    i_loss = 0.0
                    for i, flo in enumerate(flow_pred):
                        ref_i = Resize(flo.shape[-2:])(ref)
                        i_loss += loss_weights[i] * loss_term(flo, ref_i)

            else:
                raise KeyError('loss_name not registered')
            loss_dict[loss_name] = i_loss
            total_loss += loss_weight * i_loss
        loss_dict['total_loss'] = total_loss
        return loss_dict


class PhotometricLossWeighted(nn.Module):
    def __init__(self, alpha=0.45, eps = 1e-6):
        super(PhotometricLossWeighted, self).__init__()
        self.eps,  self.alpha = eps, alpha

    def forward(self, inputs, outputs, weight):
        diff = inputs - outputs
        square = torch.conj(diff) * diff
        loss = torch.pow(square + self.eps, exponent=self.alpha)
        loss = torch.multiply(weight, loss) / weight.mean()
        loss = torch.mean(loss)
        return loss


def gradient_torch(data):
    D_dx = data[:, :, 1:] - data[:, :, :-1]
    D_dy = data[:, :, :, 1:] - data[:, :, :, :-1]
    return D_dx, D_dy


def gradient(data):
    D_dy = data[:, :, 1:] - data[:, :, :-1]
    D_dx = data[:, :, :, 1:] - data[:, :, :, :-1]
    return D_dx, D_dy


class SmoothLoss(nn.Module):
    def __init__(self, boundary_awareness=True, alpha=10):
        super(SmoothLoss, self).__init__()
        self.boundary_awareness = boundary_awareness
        self.alpha = alpha
        self.func_smooth = self.smooth_grad_1st
    def smooth_grad_1st(self, flow, image):
        img_dx, img_dy = gradient(image)
        dx, dy = gradient(flow)
        eps = 1e-6
        dx, dy = torch.sqrt(dx ** 2 + eps), torch.sqrt(dy ** 2 + eps)
        dx, dy = dx.abs(), dy.abs()
        if self.boundary_awareness:
            weights_x = torch.exp(-torch.mean(torch.abs(img_dx), 1, keepdim=True) * self.alpha)
            weights_y = torch.exp(-torch.mean(torch.abs(img_dy), 1, keepdim=True) * self.alpha)
            loss_x = weights_x * dx / 2.
            loss_y = weights_y * dy / 2.
        else:
            loss_x = dx / 2.
            loss_y = dy / 2.

        return loss_x.mean() / 2. + loss_y.mean() / 2.




    def forward(self, flow_vec, image, box=None):
        if box is None:
            return self.func_smooth(flow_vec, image).mean()
        else:
            return self.func_smooth(flow_vec, image, box).mean()
