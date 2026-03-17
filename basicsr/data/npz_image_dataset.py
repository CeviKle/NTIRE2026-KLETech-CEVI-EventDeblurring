from torch.utils import data as data
import os
from pathlib import Path
import numpy as np
import torch

from basicsr.data.data_util import recursive_glob
from basicsr.data.transforms import augment, paired_random_crop
from basicsr.utils import FileClient, imfrombytes, img2tensor, get_root_logger


class NpzPngSingleDeblurDataset(data.Dataset):

    def __init__(self, opt):
        super(NpzPngSingleDeblurDataset, self).__init__()

        self.opt = opt
        self.dataroot = Path(opt['dataroot'])

        self.max_read_attempts = int(
            opt.get('max_read_attempts', opt.get('max_retries', 20))
        )

        # storage for dataset entries
        self.dataPath = []

        blur_frames = sorted(
            recursive_glob(os.path.join(self.dataroot, 'blur'), '.png')
        )
        blur_frames = [
            os.path.join(self.dataroot, 'blur', f) for f in blur_frames
        ]

        sharp_frames = sorted(
            recursive_glob(os.path.join(self.dataroot, 'sharp'), '.png')
        )
        sharp_frames = [
            os.path.join(self.dataroot, 'sharp', f) for f in sharp_frames
        ]

        # build dataset list
        for i in range(len(blur_frames)):
            sharp = sharp_frames[i] if i < len(sharp_frames) else None

            self.dataPath.append({
                'blur_path': blur_frames[i],
                'sharp_path': sharp
            })

        self.logger = get_root_logger()
        self.logger.info(
            f"Dataset initialized with {len(self.dataPath)} samples."
        )

        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.bad_indices = set()

    def __getitem__(self, index):

        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt['type'],
                **{k: v for k, v in self.io_backend_opt.items() if k != 'type'}
            )

        scale = self.opt['scale']
        gt_size = self.opt['gt_size']

        if len(self.bad_indices) >= len(self.dataPath):
            raise RuntimeError('All dataset samples are marked unreadable.')

        cur_index = index

        if cur_index in self.bad_indices:
            cur_index = np.random.randint(0, len(self.dataPath))

        last_error = None

        for retry_idx in range(self.max_read_attempts):

            if cur_index in self.bad_indices:
                cur_index = np.random.randint(0, len(self.dataPath))
                continue

            blur_path = self.dataPath[cur_index]['blur_path']
            sharp_path = self.dataPath[cur_index]['sharp_path']

            try:
                img_lq = imfrombytes(
                    self.file_client.get(blur_path), float32=True
                )

                # handle test set without GT
                if sharp_path is not None:
                    img_gt = imfrombytes(
                        self.file_client.get(sharp_path), float32=True
                    )
                else:
                    img_gt = img_lq.copy()

                if img_lq is None or img_gt is None:
                    raise ValueError('cv2.imdecode returned None')

                break

            except Exception as err:
                last_error = err
                self.bad_indices.add(cur_index)

                self.logger.warning(
                    f'Failed to load sample index={cur_index} '
                    f'(blur={blur_path}, sharp={sharp_path}) '
                    f'[retry {retry_idx + 1}/{self.max_read_attempts}]: {err}'
                )

                cur_index = np.random.randint(0, len(self.dataPath))

        else:
            raise RuntimeError(
                f'Failed to load a valid sample after '
                f'{self.max_read_attempts} retries. '
                f'Last error: {last_error}'
            )

        if gt_size is not None and gt_size > 0:
            img_gt, img_lq = paired_random_crop(
                img_gt, img_lq, gt_size, scale, sharp_path
            )

        img_lq, img_gt = img2tensor(
            augment(
                [img_lq, img_gt],
                self.opt['use_hflip'],
                self.opt['use_rot']
            )
        )

        # dummy event voxel (since test set has no events)
        voxel = torch.zeros(3, img_lq.shape[1], img_lq.shape[2])

        image_name = os.path.basename(blur_path)

        return {
            'frame': img_lq,
            'frame_gt': img_gt,
            'voxel': voxel,
            'image_name': image_name
        }

    def __len__(self):
        return len(self.dataPath)