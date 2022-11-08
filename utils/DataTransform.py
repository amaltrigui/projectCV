import transforms

import pathlib
import random

import numpy as np
import torch

class DataTransform:
  
  def __init__(self, resolution, challengetyp, mask_func=None, use_seed=True):
    """
    Args :
      resolution : resolution of image, int
      challengetyp: singlecoil (for now), or multicoil, str
      mask_func : function that creates a mask (from MaskFunc)
      use_seed (bool): If true, this class computes a pseudo random number generator seed
      from the filename. This ensures that the same mask is used for all the slices of
      a given volume every time.
      
    """
    # we ignore this for the moment
    #if challengetyp not in ('singlecoil', 'multicoil'):
     # raise ValueError(f'Challenge should either be "singlecoil" or "multicoil"')
    
    self.mask_func = mask_func
    self.resolution = resolution
    self.challengetyp = challengetyp
    self.use_seed = use_seed
 



  def __call__(self, kspace, target, attrs, fname, slice):
    # transforms from numpay.array kspace data to torch.tensor image
    """
     Args:
            kspace (numpy.array): Input k-space of shape (num_coils, rows, cols, 2) for multi-coil
                data or (rows, cols, 2) for single coil data.
            target (numpy.array): Target image
            attrs (dict): Acquisition related information stored in the HDF5 object.
            fname (str): File name
            slice (int): Serial number of the slice.
     Returns:
          (tuple): tuple containing:
              image (torch.Tensor): Zero-filled input image.
              target (torch.Tensor): Target image converted to a torch Tensor.
              mean (float): Mean value used for normalization.
              std (float): Standard deviation value used for normalization.
    """
    
    # 1. transform kspace numpyarray to kspace torch.tensor
    kspace = transforms.to_tensor(kspace)
    
    # 2. apply mask on the kspace to create undersampled images
    if self.mask_func:
      seed = None if not self.use_seed else tuple(map(ord, fname))
      masked_kspace, mask = transforms.apply_mask(kspace, self.mask_func, seed)
    else:
      masked_kspace = kspace
      
    # 3. apply inverse fourier transformation to get the zero filled solution
    image = transforms.ifft2(masked_kspace)
    
    # 4. adjust to the given resolution -> crop input to correct size
    crop_width = min(self.resolution, image.shape[-2])
    crop_height = min(self.resolution, image.shape[-3])
    if target is not None:
        crop_width = min(crop_width, target.shape[-1])
        crop_height = min(crop_height, target.shape[-2])
    crop_size = (crop_height, crop_width)
    image = transforms.complex_center_crop(image, crop_size) # image is still complex
    
    # 5. transform to absolute value
    image = transforms.complex_abs(image)
    
    # 6. normalize input
    image, mean, std = transforms.normalize_instance(image, eps=1e-11)
    image = image.clamp(-6, 6)
    
    # 7. transform target to tensor torch, crop, normalize it
    if target is not None:
      target = transforms.to_tensor(target)
      target = transforms.center_crop(target, crop_size)
      target = transforms.normalize(temp, mean, std, eps=1e-11)
      target = target.clamp(-6, 6)
    else:
      target = torch.Tensor([0])
      
      
    return image, target, mean, std, fname, slice
