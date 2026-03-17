import importlib
import torch
import torch.nn.functional as F
from collections import OrderedDict
from copy import deepcopy
import logging
import os

from basicsr.models.archs import define_network
from basicsr.models.base_model import BaseModel
from basicsr.utils import tensor2img, imwrite

loss_module = importlib.import_module('basicsr.models.losses.losses')
logger = logging.getLogger('basicsr')


class ImageEventRestorationModel(BaseModel):

    def __init__(self, opt):
        super(ImageEventRestorationModel, self).__init__(opt)

        # build network
        self.net_g = define_network(deepcopy(opt['network_g']))
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)

        if self.is_train:
            self.init_training_settings()

    # ---------------- TRAIN SETTINGS ----------------

    def init_training_settings(self):

        self.net_g.train()
        train_opt = self.opt['train']

        pixel_type = train_opt['pixel_opt']['type']
        train_opt['pixel_opt'].pop('type')

        cri_pix_cls = getattr(loss_module, pixel_type)
        self.cri_pix = cri_pix_cls(**train_opt['pixel_opt']).to(self.device)

        self.setup_optimizers()
        self.setup_schedulers()

    # ---------------- OPTIMIZER ----------------

    def setup_optimizers(self):

        train_opt = self.opt['train']

        optim_params = []
        for _, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params.append(v)

        optim_type = train_opt['optim_g']['type']
        train_opt['optim_g'].pop('type')

        if optim_type == 'AdamW':
            self.optimizer_g = torch.optim.AdamW(optim_params, **train_opt['optim_g'])
        else:
            self.optimizer_g = torch.optim.Adam(optim_params, **train_opt['optim_g'])

        self.optimizers.append(self.optimizer_g)

    # ---------------- DATA LOADING ----------------

    def feed_data(self, data):

        self.lq = data['frame'].to(self.device)
        self.gt = data['frame_gt'].to(self.device)

        event = data.get('event', None)

        if event is None:
            b, c, h, w = self.lq.shape
            event = torch.zeros(b, 6, h, w).to(self.device)
        else:
            event = event.to(self.device)

        self.event = event

    # ---------------- EDGE LOSS ----------------

    def edge_loss(self, pred, gt):

        pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        gt_dx = gt[:, :, :, 1:] - gt[:, :, :, :-1]

        pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        gt_dy = gt[:, :, 1:, :] - gt[:, :, :-1, :]

        return F.l1_loss(pred_dx, gt_dx) + F.l1_loss(pred_dy, gt_dy)

    # ---------------- SSIM LOSS ----------------

    def ssim_loss(self, pred, gt):

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        mu_x = F.avg_pool2d(pred, 3, 1, 1)
        mu_y = F.avg_pool2d(gt, 3, 1, 1)

        sigma_x = F.avg_pool2d(pred * pred, 3, 1, 1) - mu_x ** 2
        sigma_y = F.avg_pool2d(gt * gt, 3, 1, 1) - mu_y ** 2
        sigma_xy = F.avg_pool2d(pred * gt, 3, 1, 1) - mu_x * mu_y

        ssim_map = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / \
                   ((mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2))

        return torch.clamp((1 - ssim_map) / 2, 0, 1).mean()

    # ---------------- TRAIN STEP ----------------

    def optimize_parameters(self, current_iter):

        self.optimizer_g.zero_grad()

        preds = self.net_g(self.lq)

        if not isinstance(preds, list):
            preds = [preds]

        self.output = preds[-1]

        loss_dict = OrderedDict()

        l_pix = 0.
        for pred in preds:
            l_pix += self.cri_pix(pred, self.gt)

        l_edge = self.edge_loss(self.output, self.gt)
        l_ssim = self.ssim_loss(self.output, self.gt)

        l_total = l_pix + 0.1 * l_edge + 0.1 * l_ssim

        loss_dict['l_pix'] = l_pix
        loss_dict['l_edge'] = l_edge
        loss_dict['l_ssim'] = l_ssim

        l_total.backward()

        if self.opt['train'].get('use_grad_clip', True):
            torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), 0.01)

        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

       # ---------------- TEST ----------------

    def test(self):

        self.net_g.eval()

        tile = self.opt.get('val', {}).get('tile', 256)
        tile_overlap = self.opt.get('val', {}).get('tile_overlap', 32)

        with torch.no_grad():

            b, c, h, w = self.lq.size()

            if tile is None or tile == 0:
                self.output = self.net_g(self.lq)

            else:

                tile = min(tile, h, w)
                stride = tile - tile_overlap

                h_idx_list = list(range(0, h - tile, stride)) + [h - tile]
                w_idx_list = list(range(0, w - tile, stride)) + [w - tile]

                output = torch.zeros_like(self.lq)
                weight = torch.zeros_like(self.lq)

                for h_idx in h_idx_list:
                    for w_idx in w_idx_list:

                        in_patch = self.lq[:, :, h_idx:h_idx + tile, w_idx:w_idx + tile]

                        e_patch = self.event[:, :, h_idx:h_idx + tile, w_idx:w_idx + tile]
                        out_patch = self.net_g(in_patch)

                        output[:, :, h_idx:h_idx + tile, w_idx:w_idx + tile] += out_patch
                        weight[:, :, h_idx:h_idx + tile, w_idx:w_idx + tile] += 1

                self.output = output / weight

        self.net_g.train()
    # ---------------- SAVE ----------------

    def save(self, epoch, current_iter):

        self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter)

    # ---------------- VALIDATION ----------------

    def nondist_validation(self, dataloader, current_iter, tb_logger,
                           save_img=False, rgb2bgr=True, use_image=True):

        self.net_g.eval()

        for idx, data in enumerate(dataloader):

            self.feed_data(data)
            self.test()

            if save_img:

                img_name = data['image_name'][0]

                save_path = os.path.join(
                    self.opt['path']['visualization'],
                    img_name
                )

                output = tensor2img([self.output])

                imwrite(output, save_path)

        self.net_g.train()
    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        return out_dict
