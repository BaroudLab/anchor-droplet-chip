from typing import Tuple
from tifffile import imread, imwrite
import matplotlib.pyplot as plt
import imreg_dft as reg
import numpy as np
import scipy.ndimage as ndi
import os

GREY = np.array([np.arange(256)] * 3, dtype='uint8')
RED = np.array([np.arange(256), np.zeros((256,)), np.zeros((256,))], dtype='uint8')
GREEN = np.array([np.zeros((256,)), np.arange(256), np.zeros((256,))], dtype='uint8')
BLUE = np.array([np.zeros((256,)), np.zeros((256,)), np.arange(256)], dtype='uint8')

META_ALIGNED = {'ImageJ': '1.53c',
 'images': 3,
 'channels': 3,
 'hyperstack': True,
 'mode': 'composite',
 'unit': '',
 'loop': False,
 'min': 1878.0,
 'max': 30728.0,
 'Ranges': (1878.0, 30728.0, 430.0, 600.0, 0.0, 501.0),
 'LUTs': [GREY, GREEN, BLUE]
 }




def align_stack(bf_fluo_data:np.ndarray, template16:np.ndarray, mask2:np.ndarray, plot:bool=False, path_to_save:str=None, binnings:Tuple=(2,16,2), metadata:dict=META_ALIGNED):
    '''
    stack should contain two channels: bright field and fluorescence.
    BF will be binned 8 times and registered with template8 (aligned BF).
    When the transformation verctor will be applied to the original data and stacked with the mask.
    The output stack is of the same size as mask.
    The resulting 3-layer stack will be returned and also saved with suffix ".aligned.tif"

    Parameters:
    ===========
    bf_fluo_data : np.ndarray
        Bright-field + fluorescence stack with the shape (2, Y, X)
    template16 : np.ndarray
        binned template of aligned bright-filed image of the chip
    mask2 : np.ndarray
        Labelled mask which you try to align with the data
    plot : bool, optional
        Plot results
    path_to_save :str
        Path to the .tif file to save aligned bright-filed + fluo + mask
    binnings : tuple(data, template, mask)
        Bright-field channel will be binned to match the scale of the template. 
        The transformation vector will then be upscaled back to transform the original data.
        The aligned data will be binned to match the scale of the mask
    metadata: dict, optional
        ImageJ tif metadata.
        Default:
            META_ALIGNED = {'ImageJ': '1.53c',
                'images': 3,
                'channels': 3,
                'hyperstack': True,
                'mode': 'composite',
                'unit': '',
                'loop': False,
                'min': 1878.0,
                'max': 30728.0,
                'Ranges': (1878.0, 30728.0, 430.0, 600.0, 0.0, 501.0),
                'LUTs': [grey, green, blue]
            }

    Returns
    -------
    aligned_stack : np.ndarray, 
    tvec : dict
        aligned_stack: bf+fluo+mask,
        tvec: transform dictionary
    
    '''
    if isinstance(data_or_path, str):
        path = data_or_path
        stack = imread(path)
        print(path, stack.shape)
    else:
        assert data_or_path.ndim == 3 and data_or_path.shape[0] == 2
        stack = data_or_path

    print(f'Aligned stack will be saved to {path_to_save}')

    bf, tritc = stack[:2]
    stack_temp_scale = binnings[1] // binnings[0]
    mask_temp_scale = binnings[1] // binnings[2]
    stack_mask_scale = binnings[2] // binnings[0]
    
    f_bf = bf[::stack_temp_scale, ::stack_temp_scale]
    f_bf_sm = ndi.gaussian_filter(f_bf, 2)
    # f_bf = filter_by_fft(
    #     bf[::stack_temp_scale, ::stack_temp_scale], 
    #     sigma=40,
    #     fix_horizontal_stripes=True, 
    #     fix_vertical_stripes=True,
    #     highpass=True
    # )
    tvec8 = get_transform(f_bf_sm, template16, plot=plot)
    plt.show()
    tvec = scale_tvec(tvec8, mask_temp_scale)
    print(tvec)
    try:
        aligned_tritc = unpad(transform(tritc[::stack_mask_scale, ::stack_mask_scale], tvec), mask2.shape)
        aligned_bf = unpad(transform(bf[::stack_mask_scale, ::stack_mask_scale], tvec), mask2.shape)
    except ValueError as e:
        print("stack_mask_scale: ", stack_mask_scale)
        print(e.args)
        raise e
    
    if plot:
        plt.figure(dpi=300)
        plt.imshow(aligned_tritc, cmap='gray',)# vmax=aligned_tritc.max()/5)
        plt.colorbar()
        plt.show()

        saturated_tritc = aligned_tritc.copy()
        saturated_tritc[saturated_tritc > 500] = 500
        plt.figure(dpi=300)
        plt.imshow(mic.segment.label2rgb(mic.segment.label(mask2), to_8bits(saturated_tritc), bg_label=0))
        plt.show()

    aligned_stack = np.stack((aligned_bf, aligned_tritc, mask2)).astype('uint16')

    if path_to_save is not None:
        imwrite(path_to_save, aligned_stack, imagej=True, metadata=META_ALIGNED)
    print(f'Saved aligned stack {path_to_save}')
    return aligned_stack, tvec


def get_transform(image, template, plot=False, pad_ratio=1.2, figsize=(10,5), dpi=300):
    '''
    Pads image and template, registers and returns tvec
    '''
    padded_template = pad(template, (s := increase(image.shape, pad_ratio)))
    padded_image = pad(image, s)
    tvec = register(padded_image, padded_template)
    if plot:
        aligned_bf = unpad(tvec['timg'], template.shape)
        plt.figure(figsize=figsize, dpi=dpi)
        plt.imshow(aligned_bf, cmap='gray')
    return tvec


def register(image, template):
    '''
    Register image towards template
    Return:
    tvec:dict
    '''
    assert np.array_equal(image.shape, template.shape), \
        f'unequal shapes {(image.shape, template.shape)}'
    return reg.similarity(template, image, constraints={'scale': [1,0.2], 'tx': [0, 500], 'ty': [0, 500], 'angle': [0, 30]})


def pad(image:np.ndarray, to_shape:tuple=None, padding:tuple=None):
    '''
    Pad the data to desired shape
    '''
    if padding is None:
        padding = calculate_padding(image.shape, to_shape)
    try:
        padded = np.pad(image, padding, 'edge')
    except TypeError as e:
        print(e.args, padding)
        raise e
    return padded


def unpad(image:np.ndarray, to_shape:tuple=None, padding:tuple=None):
    '''
    Remove padding to get desired shape
    '''
    if any(np.array(image.shape) - np.array(to_shape) < 0):
        print(f'unpad:warning: image.shape {image.shape} is within to_shape {to_shape}')
        image = pad(image, np.array((image.shape, to_shape)).max(axis=0))
        print(f'new image shape after padding {image.shape}')
    if padding is None:
        padding = calculate_padding(to_shape, image.shape)
    
    y = [padding[0][0], -padding[0][1]]
    if y[1] == 0:
        y[1] = None
    x = [padding[1][0], -padding[1][1]]
    if x[1] == 0:
        x[1] = None
    return image[y[0]:y[1], x[0]:x[1]]


def calculate_padding(shape1:tuple, shape2:tuple):
    '''
    Calculates padding to get shape2 from shape1
    Return:
    2D tuple of indices
    '''
    dif = np.array(shape2) - np.array(shape1)
    assert all(dif >= 0), f'Shape2 must be bigger than shape1, got {shape2}, {shape1}'
    mid = dif // 2
    rest = dif - mid
    y = mid[0], rest[0]
    x = mid[1], rest[1]
    return y, x


def scale_tvec(tvec, scale=8):
    '''
    Scale up transform vector from imreg_dft
    '''
    tvec_8x = tvec.copy()
    tvec_8x['tvec'] = tvec['tvec'] * scale
    try:
        tvec_8x['timg'] = None
    except KeyError:
        pass
    finally:
        return tvec_8x
    
    
def transform(image, tvec):
    '''
    apply transform
    '''
    print(f'transform {image.shape}')
    fluo = reg.transform_img_dict(image, tvec)
    return fluo.astype('uint')

def main(data_path:str, template_path:str, mask_path:str, binnings:tuple=(2,16,2), path_to_save:str=''):
    '''
    reads the data from disk and runs alignment
    all paths should be .tif

    
    '''
    stack = imread(data_path)
    template = imread(template_path)
    mask = imread(mask_path)
    aligned, tvec = align_stack(stack, template, mask, path_to_save=path_to_save, binnings=binnings)
    return tvec

if __name__ == "__main__":
    fire.Fire(main)