import h5py
from torch.utils.data import Dataset
from .utils import get_logger
import pandas as pd
import numpy as np
from scipy.ndimage.filters import gaussian_filter
import torchio as tio
import yaml # pyyaml

logger = get_logger('ConfigLoader')

def getBoundingBoxMultichannel(Image, MRvox_middle, xthr, ythr, zthr):
    # zero padding
    if (Image.shape[1] < xthr) or (Image.shape[2] < ythr) or (Image.shape[3] < zthr):
        pad = np.array([xthr-Image.shape[1], ythr-Image.shape[2], zthr-Image.shape[3]])/2
        pad[pad<0] = 0
        Image = np.pad(Image, ((0,0),
                               (np.floor(pad[0]).astype('int'),np.ceil(pad[0]).astype('int')),
                               (np.floor(pad[1]).astype('int'), np.ceil(pad[1]).astype('int')),
                               (np.floor(pad[2]).astype('int'), np.ceil(pad[2]).astype('int'))), 'constant')

    # Image: 4D
    bbx_start, bby_start, bbz_start = np.floor(MRvox_middle - np.array([xthr / 2, ythr / 2, zthr / 2])).astype(
        'int')
    if bbx_start < 0: bbx_start = 0
    if bby_start < 0: bby_start = 0
    if bbz_start < 0: bbz_start = 0
    # first Image dimension is CHANNEL
    if bbx_start + xthr > Image.shape[1]: bbx_start = Image.shape[1]-xthr
    if bby_start + ythr > Image.shape[2]: bby_start = Image.shape[2]-ythr
    if bbz_start + zthr > Image.shape[3]: bbz_start = Image.shape[3]-zthr
    return Image[:, bbx_start: (bbx_start + xthr), bby_start: (bby_start + ythr), bbz_start:(bbz_start + zthr)]

def _load_config_yaml(config_file):
    return yaml.safe_load(open(config_file, 'r'))

class prostatex_dataloader_cnn3d(Dataset):
    def __init__(self, csvfname='/home/hk672/data-preprocessed/PROSTATEx-new/prostateX-demo-csPCa-prediction-multimodal.csv',
                mode='train', config_mode=None, **kwargs ):
        # mode = 'train' 'test' 'val'
        self.mode = mode
        self.csvfname = csvfname
        self.csvf = pd.read_csv(self.csvfname, index_col=None)
        self.csvfmode = self.csvf[self.csvf.trainvaltest == self.mode].reset_index(drop=False)

        assert config_mode is not None, f"config_mode info should be assigned"
        partialdata = config_mode['partialdata']
        if (partialdata > 0) or (partialdata != False):
            self.partialdata = np.round(partialdata * len(self.csvfmode)).astype('int')
            self.csvfmode = self.csvfmode[:self.partialdata]

        self.znorm = config_mode['znorm']
        self.minmax = config_mode['minmax']
        self.targetname =  config_mode['target']
        self.lr_flip = config_mode['augment_lr_flip']
        self.affine = config_mode['augment_affine']
        self.intensity = config_mode['augment_intensity']
        self.translate = config_mode['augment_translate']
        self.patchsize = config_mode['patchsize']

        if 'clip' in config_mode:
            self.clip = True
        else:
            self.clip = False

        if 'lesion_probability' in config_mode['channels']:
            self.channels = config_mode['channels'][:-1]
            self.add_lesion_probability = True
        else:
            self.channels = config_mode['channels']
            self.add_lesion_probability = False

        if 'onoffchannels' in config_mode:
            self.onoffchannels = config_mode['onoffchannels']
        else:
            self.onoffchannels = None


        # # get target from Gleason Group (bc of the label weight)
        if not (self.targetname in ['csPCa', 'Gleason Group', 'Clinically Significant',
                                    'Gleason Grade Group','ClinSig', 'ggg']):
            raise NameError('check "target" name in config')
        else:
            self.target = np.array(self.csvfmode[self.targetname]).astype('int')


        transforms = []

        if self.znorm:
            znorm = tio.ZNormalization()
            transforms.append(znorm)

        if self.minmax:
            minmax = tio.RescaleIntensity(out_min_max=(0, 1))
            transforms.append(minmax)

        if self.lr_flip:
            lrflip = tio.RandomFlip(axes=(0,), flip_probability=1) # always flip
            transforms.append(lrflip)

        if self.affine:
            affine = tio.RandomAffine(scales=0.1,
                                         degrees=10,
                                         image_interpolation='linear',
                                         default_pad_value='otsu')  # bspline
            transforms.append(affine)


        if self.intensity: # .5 percentage .5 percentage
            # intensity = tio.RandomNoise()
            intensity = tio.OneOf({
                # tio.RandomBiasField(): 0.5,
                tio.RandomNoise(): 0.5,
                tio.RandomBlur(): 0.5
            },
                p=0.5,
            )
            transforms.append(intensity)

        self.transform = tio.Compose(transforms)

    def __len__(self):
        return len(self.csvfmode)
    #
    def __getitem__(self, idx):

        hf = h5py.File(self.csvfmode['h5-fname'].iloc[idx], 'r')
        patch_size = self.patchsize # (np.array(hf['patch-size'])/2).astype('int')
        # diagnosis = np.array(hf['csPCa'])
        diagnosis = self.target[idx]

        masknames = ['lesion', 'prostate', 'tz', 'pz', 'cz']

        if not self.onoffchannels == None:
            # randomly mask the selected channels
            onoff_idx = np.array([cname in self.onoffchannels for cname in self.channels])
            onoff_random = np.random.randint(2, size = onoff_idx.sum())
            onoff_idx_update = onoff_idx
            onoff_idx_update[onoff_idx] = onoff_random
            self.onoff_chan_bool = list(onoff_idx_update)
        else:
            self.onoff_chan_bool = [False] * len(self.channels)


        def image_clip(image):
            if self.clip:
                lb, up = np.percentile(image, 0.1),np.percentile(image, 99.9)
                return np.clip(image, lb, up)
            else:
                return image

        subject_dict = dict()
        for c_idx in range(len(self.channels)):
            # if channel is mask
            if self.channels[c_idx] in masknames:
                # if this channel will be on and off
                if self.onoff_chan_bool[c_idx]:
                    subject_dict[self.channels[c_idx]] = tio.LabelMap(
                        tensor=np.zeros(hf[self.channels[c_idx]].shape)[None, :])
                else:
                    subject_dict[self.channels[c_idx]] = tio.LabelMap(tensor=np.array(hf[self.channels[c_idx]])[None, :])
            # if channel is image not mask
            else:
                # if this channel will be on and off
                if self.onoff_chan_bool[c_idx]:
                    subject_dict[self.channels[c_idx]] = tio.ScalarImage(
                        tensor=np.zeros(hf[self.channels[c_idx]].shape)[None, :])
                else:
                    subject_dict[self.channels[c_idx]] = tio.ScalarImage(
                        tensor=image_clip(np.array(hf[self.channels[c_idx]])[None, :])) # if self.clip is true clip applied

        # lesion always present
        assert 'lesion' in hf.keys(), 'lesion always present!!! check h5 file'
        subject_dict['segmentation'] = tio.LabelMap(tensor=np.array(hf['lesion'])[None,:])
        subject = tio.Subject(subject_dict)

        transformed = self.transform(subject)

        # get patches ## add randomness 0.5 percentage:
        if self.translate:
            translation_binary = np.random.randint(2, size=1)
        else:
            translation_binary = 0

        lesion_middle_point = np.median(np.array(np.where(transformed['segmentation'])),
                                        1).astype('int')[1:] + translation_binary*np.random.randint(-4, 5, 3)

        # concatenate images
        img_concat = np.concatenate(tuple([transformed[self.channels[c_idx]] for c_idx in range(len(self.channels))]))

        if self.add_lesion_probability:
            lesionprob = gaussian_filter(np.array(transformed['segmentation'])[0]*np.prod(self.patchsize), (10, 10, 10)).astype('float')
            lesionprobmax = lesionprob.max().astype('float')
            lesionprob = np.divide(lesionprob, lesionprobmax)
            # print(np.sum(np.isnan(lesionprob)))
            img_concat = np.concatenate((img_concat,lesionprob[None, :]))

        input = getBoundingBoxMultichannel(img_concat, \
                        lesion_middle_point, patch_size[0], patch_size[1], patch_size[2])

        hf.close()
        #
        return input , np.array([diagnosis]) # target for learning prediction
