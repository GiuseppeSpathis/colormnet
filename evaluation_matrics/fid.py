import os
import numpy as np
from PIL import Image
from scipy import linalg
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import models
from torchvision.transforms import transforms


class INCEPTION_V3_FID(nn.Module):
    """pretrained InceptionV3 network returning feature maps"""
    # Index of default block of inception to return,
    # corresponds to output of final average pooling
    DEFAULT_BLOCK_INDEX = 3

    # Maps feature dimensionality to their output blocks indices
    BLOCK_INDEX_BY_DIM = {
        64: 0,  # First max pooling features
        192: 1,  # Second max pooling featurs
        768: 2,  # Pre-aux classifier features
        2048: 3  # Final average pooling features
    }

    def __init__(self,
                 incep_state_dict,
                 output_blocks=[DEFAULT_BLOCK_INDEX],
                 resize_input=True):
        """Build pretrained InceptionV3
        Parameters
        ----------
        output_blocks : list of int
            Indices of blocks to return features of. Possible values are:
                - 0: corresponds to output of first max pooling
                - 1: corresponds to output of second max pooling
                - 2: corresponds to output which is fed to aux classifier
                - 3: corresponds to output of final average pooling
        resize_input : bool
            If true, bilinearly resizes input to width and height 299 before
            feeding input to model. As the network without fully connected
            layers is fully convolutional, it should be able to handle inputs
            of arbitrary size, so resizing might not be strictly needed
        normalize_input : bool
            If true, normalizes the input to the statistics the pretrained
            Inception network expects
        """
        super(INCEPTION_V3_FID, self).__init__()

        self.resize_input = resize_input
        self.output_blocks = sorted(output_blocks)
        self.last_needed_block = max(output_blocks)

        assert self.last_needed_block <= 3, \
            'Last possible output block index is 3'

        self.blocks = nn.ModuleList()

        inception = models.inception_v3(init_weights=False)
        inception.load_state_dict(incep_state_dict)
        for param in inception.parameters():
            param.requires_grad = False

        # Block 0: input to maxpool1
        block0 = [
            inception.Conv2d_1a_3x3,
            inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2)
        ]
        self.blocks.append(nn.Sequential(*block0))

        # Block 1: maxpool1 to maxpool2
        if self.last_needed_block >= 1:
            block1 = [
                inception.Conv2d_3b_1x1,
                inception.Conv2d_4a_3x3,
                nn.MaxPool2d(kernel_size=3, stride=2)
            ]
            self.blocks.append(nn.Sequential(*block1))

        # Block 2: maxpool2 to aux classifier
        if self.last_needed_block >= 2:
            block2 = [
                inception.Mixed_5b,
                inception.Mixed_5c,
                inception.Mixed_5d,
                inception.Mixed_6a,
                inception.Mixed_6b,
                inception.Mixed_6c,
                inception.Mixed_6d,
                inception.Mixed_6e,
            ]
            self.blocks.append(nn.Sequential(*block2))

        # Block 3: aux classifier to final avgpool
        if self.last_needed_block >= 3:
            block3 = [
                inception.Mixed_7a,
                inception.Mixed_7b,
                inception.Mixed_7c,
                nn.AdaptiveAvgPool2d(output_size=(1, 1))
            ]
            self.blocks.append(nn.Sequential(*block3))

    def forward(self, inp):
        """Get Inception feature maps
        Parameters
        ----------
        inp : torch.autograd.Variable
            Input tensor of shape Bx3xHxW. Values are expected to be in
            range (0, 1)
        Returns
        -------
        List of torch.autograd.Variable, corresponding to the selected output
        block, sorted ascending by index
        """
        outp = []
        x = inp

        if self.resize_input:
            x = F.interpolate(x, size=(299, 299), mode='bilinear')

        x = x.clone()
        # [-1.0, 1.0] --> [0, 1.0]
        x = x * 0.5 + 0.5
        x[:, 0] = x[:, 0] * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
        x[:, 1] = x[:, 1] * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
        x[:, 2] = x[:, 2] * (0.225 / 0.5) + (0.406 - 0.5) / 0.5

        for idx, block in enumerate(self.blocks):
            x = block(x)
            if idx in self.output_blocks:
                outp.append(x)

            if idx == self.last_needed_block:
                break

        return outp


def get_activations(images, model, batch_size, verbose=False):
    """Calculates the activations of the pool_3 layer for all images.
    Params:
    -- images      : Numpy array of dimension (n_images, 3, hi, wi). The values
                     must lie between 0 and 1.
    -- model       : Instance of inception model
    -- batch_size  : the images numpy array is split into batches with
                     batch size batch_size. A reasonable batch size depends
                     on the hardware.
    -- verbose     : If set to True and parameter out_step is given, the number
                     of calculated batches is reported.
    Returns:
    -- A numpy array of dimension (num images, dims) that contains the
       activations of the given tensor when feeding inception with the
       query tensor.
    """
    model.eval()

    #d0 = images.shape[0]
    d0 = int(images.size(0))
    if batch_size > d0:
        # print(('Warning: batch size is bigger than the data size. '
        #        'Setting batch size to data size'))
        batch_size = d0

    n_batches = d0 // batch_size
    n_used_imgs = n_batches * batch_size

    pred_arr = np.empty((n_used_imgs, 2048))
    for i in range(n_batches):
        if verbose:
            print('\rPropagating batch %d/%d' % (i + 1, n_batches), end='', flush=True)
        start = i * batch_size
        end = start + batch_size

        '''batch = torch.from_numpy(images[start:end]).type(torch.FloatTensor)
        batch = Variable(batch, volatile=True)
        if cfg.CUDA:
            batch = batch.cuda()'''
        batch = images[start:end]

        pred = model(batch)[0]

        # If model output is not scalar, apply global spatial average pooling.
        # This happens if you choose a dimensionality not equal 2048.
        if pred.shape[2] != 1 or pred.shape[3] != 1:
            pred = F.adaptive_avg_pool2d(pred, output_size=(1, 1))

        pred_arr[start:end] = pred.cpu().data.numpy().reshape(batch_size, -1)

    if verbose:
        print(' done')

    return pred_arr


def calculate_activation_statistics(act):
    """Calculation of the statistics used by the FID.
    Params:
    -- act      : Numpy array of dimension (n_images, dim (e.g. 2048)).
    Returns:
    -- mu    : The mean over samples of the activations of the pool_3 layer of
               the inception model.
    -- sigma : The covariance matrix of the activations of the pool_3 layer of
               the inception model.
    """
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representive data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representive data set.
    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)


class pair_dataset(Dataset):
    def __init__(self, data_dir, gt_dir):
        self.gt_dir = gt_dir
        self.data_dir = data_dir

        filenames = []
        input_folder_list = os.listdir(gt_dir)
        for folder in input_folder_list:
            frame_names = os.listdir(os.path.join(gt_dir, folder))
            filenames += [os.path.join(folder, frame_name) for frame_name in frame_names]

        self.filenames = filenames
        self.transform_list = transforms.Compose([transforms.ToTensor(),
                                                  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    def __getitem__(self, index):
        img_pth = self.filenames[index]
        img_dir = os.path.join(self.gt_dir, img_pth)
        img = Image.open(img_dir).convert('RGB')
        gt_img = self.transform_list(img).float()

        fake_dir = os.path.join(self.data_dir, img_pth)
        fake_img = Image.open(fake_dir).convert("RGB")
        fake_img = self.transform_list(fake_img).float()

        return fake_img, gt_img

    def __len__(self):
        return len(self.filenames)

class pair_dataset_Imagelist(Dataset):
    def __init__(self, data_dir, gt_dir):
        self.gt_dir = gt_dir
        self.data_dir = data_dir

        input_folder_list = sorted(os.listdir(gt_dir))

        self.filenames = input_folder_list
        self.transform_list = transforms.Compose([transforms.ToTensor(),
                                                  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    def __getitem__(self, index):
        img_pth = self.filenames[index]
        img_dir = os.path.join(self.gt_dir, img_pth)

        img = Image.open(img_dir).convert('RGB')
        gt_img = self.transform_list(img).float()

        fake_dir = os.path.join(self.data_dir, img_pth)
        fake_img = Image.open(fake_dir).convert("RGB")
        fake_img = self.transform_list(fake_img).float()

        return fake_img, gt_img

    def __len__(self):
        return len(self.filenames)
    

def calculate_fid_yyx(pred_dir, gt_dir, bs=None):
    batch_size = bs
    new_batch_size = bs
    incep_pth = os.path.join('pretrain', "inception_v3_google-1a9a5a14.pth")
    if not os.path.exists(incep_pth):
        import wget
        incep_url = 'https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth'
        os.makedirs('pretrain', exist_ok=True)
        wget.download(incep_url, incep_pth)

    incep_state_dict = torch.load(incep_pth, map_location='cpu')
    block_idx = INCEPTION_V3_FID.BLOCK_INDEX_BY_DIM[2048]
    inception_model_fid = INCEPTION_V3_FID(incep_state_dict, [block_idx])
    inception_model_fid.cuda()
    inception_model_fid.eval()

    clips = sorted(os.listdir(pred_dir))
    avg_fid_score = 0
    for clip in clips:
        pred_dir_clip = os.path.join(pred_dir, clip) 
        gt_dir_clip = os.path.join(gt_dir, clip) 

        dataset = pair_dataset_Imagelist(pred_dir_clip, gt_dir_clip)

        # print(len(dataset))
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
        fake_acts_set, acts_set = [], []
        with torch.no_grad():
            # for i, batch in enumerate(tqdm(dataloader)):
            for i, batch in enumerate(dataloader):
                # pred, gt = batch[0], batch[1]  # cpu
                pred, gt = batch[0].cuda(), batch[1].cuda()
                fake_act = get_activations(pred, inception_model_fid, new_batch_size)
                real_act = get_activations(gt, inception_model_fid, new_batch_size)
                fake_acts_set.append(fake_act)
                acts_set.append(real_act)
                # break
            acts_set = np.concatenate(acts_set, 0)
            fake_acts_set = np.concatenate(fake_acts_set, 0)

            real_mu, real_sigma = calculate_activation_statistics(acts_set)
            fake_mu, fake_sigma = calculate_activation_statistics(fake_acts_set)
            fid_score = calculate_frechet_distance(real_mu, real_sigma, fake_mu, fake_sigma)

            avg_fid_score += fid_score
            # print(fid_score)

    avg_fid_score = avg_fid_score/len(clips)
    return avg_fid_score


def calculate_fid(pred_dir, gt_dir, bs=None):
    batch_size = bs
    new_batch_size = bs
    incep_pth = os.path.join('pretrain', "inception_v3_google-1a9a5a14.pth")
    if not os.path.exists(incep_pth):
        import wget
        incep_url = 'https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth'
        os.makedirs('pretrain', exist_ok=True)
        wget.download(incep_url, incep_pth)

    incep_state_dict = torch.load(incep_pth, map_location='cpu')
    block_idx = INCEPTION_V3_FID.BLOCK_INDEX_BY_DIM[2048]
    inception_model_fid = INCEPTION_V3_FID(incep_state_dict, [block_idx])
    inception_model_fid.cuda()
    inception_model_fid.eval()

    dataset = pair_dataset(pred_dir, gt_dir)
    # dataset = pair_dataset_Imagelist(pred_dir, gt_dir)

    # print(len(dataset))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    fake_acts_set, acts_set = [], []
    with torch.no_grad():
        # for i, batch in enumerate(tqdm(dataloader)):
        for i, batch in enumerate(dataloader):
            # pred, gt = batch[0], batch[1]  # cpu
            pred, gt = batch[0].cuda(), batch[1].cuda()
            fake_act = get_activations(pred, inception_model_fid, new_batch_size)
            real_act = get_activations(gt, inception_model_fid, new_batch_size)
            fake_acts_set.append(fake_act)
            acts_set.append(real_act)
            # break
        acts_set = np.concatenate(acts_set, 0)
        fake_acts_set = np.concatenate(fake_acts_set, 0)

        real_mu, real_sigma = calculate_activation_statistics(acts_set)
        fake_mu, fake_sigma = calculate_activation_statistics(fake_acts_set)
        fid_score = calculate_frechet_distance(real_mu, real_sigma, fake_mu, fake_sigma)
    return fid_score



class pair_dataset_fixres(Dataset):
    def __init__(self, data_dir, gt_dir):
        self.gt_dir = gt_dir
        self.data_dir = data_dir

        filenames = []
        input_folder_list = os.listdir(gt_dir)
        for folder in input_folder_list:
            frame_names = os.listdir(os.path.join(gt_dir, folder))
            filenames += [os.path.join(folder, frame_name) for frame_name in frame_names]

        self.filenames = filenames
        self.transform_list = transforms.Compose([transforms.ToTensor(),
                                                  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    def __getitem__(self, index):
        img_pth = self.filenames[index]
        img_dir = os.path.join(self.gt_dir, img_pth)
        # img = Image.open(img_dir).convert('RGB')
        img = Image.open(img_dir).convert('RGB').resize((832, 480))
        gt_img = self.transform_list(img).float()

        fake_dir = os.path.join(self.data_dir, img_pth)
        fake_img = Image.open(fake_dir).convert("RGB").resize((832, 480))
        fake_img = self.transform_list(fake_img).float()

        return fake_img, gt_img

    def __len__(self):
        return len(self.filenames)


def calculate_fid_fixres(pred_dir, gt_dir):
    batch_size = 128
    new_batch_size = 128
    incep_pth = os.path.join('pretrain', "inception_v3_google-1a9a5a14.pth")
    if not os.path.exists(incep_pth):
        import wget
        incep_url = 'https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth'
        os.makedirs('pretrain', exist_ok=True)
        wget.download(incep_url, incep_pth)

    incep_state_dict = torch.load(incep_pth, map_location='cpu')
    block_idx = INCEPTION_V3_FID.BLOCK_INDEX_BY_DIM[2048]
    inception_model_fid = INCEPTION_V3_FID(incep_state_dict, [block_idx])
    inception_model_fid.cuda()
    inception_model_fid.eval()

    dataset = pair_dataset_fixres(pred_dir, gt_dir)
    # print(len(dataset))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    fake_acts_set, acts_set = [], []
    with torch.no_grad():
        # for i, batch in enumerate(tqdm(dataloader)):
        for i, batch in enumerate(dataloader):
            # pred, gt = batch[0], batch[1]  # cpu
            pred, gt = batch[0].cuda(), batch[1].cuda()
            # print(pred.shape, gt.shape);assert 1==0
            fake_act = get_activations(pred, inception_model_fid, new_batch_size)
            real_act = get_activations(gt, inception_model_fid, new_batch_size)
            fake_acts_set.append(fake_act)
            acts_set.append(real_act)
            # break
        acts_set = np.concatenate(acts_set, 0)
        fake_acts_set = np.concatenate(fake_acts_set, 0)

        real_mu, real_sigma = calculate_activation_statistics(acts_set)
        fake_mu, fake_sigma = calculate_activation_statistics(fake_acts_set)
        fid_score = calculate_frechet_distance(real_mu, real_sigma, fake_mu, fake_sigma)
    return fid_score