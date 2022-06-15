from functools import partial
import matplotlib.pyplot as plt
import numpy as np
from skimage.feature import peak_local_max
from skimage.feature import peak
from scipy import ndimage as ndi
from tifffile import imread, imwrite
import fire



def get_cell_numbers(
    multiwell_image:np.ndarray, 
    labels:np.ndarray, 
    plot=False, 
    threshold_abs:float=2, 
    min_distance:float=5,
    meta:dict={},
    bf:np.ndarray=None
    ) -> pd.DataFrame:
    
    props = segment.regionprops(labels)

    def get_n_peaks(i):
        if bf is None:
            return get_peak_number(
                multiwell_image[props[i].slice], 
                plot=plot, 
                dif_gauss_sigma=(3, 5), 
                threshold_abs=threshold_abs, 
                min_distance=min_distance,
                title=props[i].label
                )
        else:
            return get_peak_number(
                multiwell_image[props[i].slice], 
                plot=plot, 
                dif_gauss_sigma=(3, 5), 
                threshold_abs=threshold_abs, 
                min_distance=min_distance,
                title=props[i].label,
                bf_crop=bf[props[i].slice],
                return_std=True
                )

    n_cells = list(map(get_n_peaks, range(labels.max())))
    return pd.DataFrame([{
        'label': prop.label, 
        'x': prop.centroid[0], 
        'y': prop.centroid[1], 
        'n_cells': n_cell[0],
        # 'std': n_cell[1],
        **meta
        } for prop, n_cell in zip(props, n_cells)])



def crop(stack:np.ndarray, center:tuple, size:int):
    im = stack[:, int(center[0]) - size//2:int(center[0]) + size//2, int(center[1]) - size//2:int(center[1]) + size//2]
    return im


def gdif(array2d, dif_gauss_sigma=(1, 3)):
    array2d = array2d.astype('f')
    return ndi.gaussian_filter(array2d, sigma=dif_gauss_sigma[0]) - ndi.gaussian_filter(array2d, sigma=dif_gauss_sigma[1])


def get_peak_number(crop2d, dif_gauss_sigma=(1, 3), min_distance=3, threshold_abs=5, plot=False, 
                    title='', bf_crop=None, return_std=False):
    image_max = gdif(crop2d, dif_gauss_sigma)
    peaks = peak_local_max(image_max, min_distance=min_distance, threshold_abs = threshold_abs)
    
    if plot:
        if bf_crop is None:
            fig, ax = plt.subplots(1, 2, sharey=True)
            ax[0].imshow(crop2d)
            ax[0].set_title(f'raw image {title}')
            ax[1].imshow(image_max)
            ax[1].set_title('Filtered + peak detection')
            ax[1].plot(peaks[:,1], peaks[:,0], 'r.')
            plt.show()
        else:
            fig, ax = plt.subplots(1, 3, sharey=True)

            ax[0].imshow(bf_crop, cmap='gray')
            ax[0].set_title(f'BF {title}')
            
            ax[1].imshow(crop2d, vmax=crop2d.mean() + 2 * crop2d.std())
            ax[1].set_title(f'raw image {title}')

            ax[2].imshow(image_max)
            ax[2].set_title(f'Filtered + {len(peaks)} peaks (std {image_max.std():.2f})')
            ax[2].plot(peaks[:,1], peaks[:,0], 'r.')
            plt.show()

    if return_std:
        return len(peaks), crop2d.std()
    else:
        return (len(peaks), )


def get_peaks_per_frame(stack3d, dif_gauss_sigma=(1, 3), **kwargs):
    image_ref = gdif(stack3d[0], dif_gauss_sigma)
    thr = 5 * image_ref.std()
    return list(map(partial(get_peak_number, threshold_abs=thr, **kwargs), stack3d))


def get_peaks_all_wells(stack, centers, size, plot=0):
    n_peaks = []
    for c in centers:
        print('.', end='')
        well = crop(stack, c['center'], size)
        n_peaks.append(get_peaks_per_frame(well, plot=plot))
    return n_peaks

def main(
    aligned_path:str, 
    gaussian_filter:tuple=(3,5), 
    threshold:float=2, 
    min_distance:float=5, 
    save_path_csv:str="", 
    **kwargs
):
def get_cell_numbers(
    multiwell_image:np.ndarray, 
    labels:np.ndarray, 
    plot=False, 
    threshold_abs:float=2, 
    min_distance:float=5,
    meta:dict={},
    bf:np.ndarray=None
    ) -> pd.DataFrame:
    
    props = segment.regionprops(labels)

    def get_n_peaks(i):
        if bf is None:
            return count.get_peak_number(
                multiwell_image[props[i].slice], 
                plot=plot, 
                dif_gauss_sigma=(3, 5), 
                threshold_abs=threshold_abs, 
                min_distance=min_distance,
                title=props[i].label
                )
        else:
            return count.get_peak_number(
                multiwell_image[props[i].slice], 
                plot=plot, 
                dif_gauss_sigma=(3, 5), 
                threshold_abs=threshold_abs, 
                min_distance=min_distance,
                title=props[i].label,
                bf_crop=bf[props[i].slice],
                return_std=True
                )

    n_cells = list(map(get_n_peaks, range(labels.max())))
    return pd.DataFrame([{
        'label': prop.label, 
        'x': prop.centroid[0], 
        'y': prop.centroid[1], 
        'n_cells': n_cell[0],
        # 'std': n_cell[1],
        **meta
        } for prop, n_cell in zip(props, n_cells)])


    '''
    Reads the data and saves the counting table
    '''
    bf, fluo, mask = imread(aligned_path)
    table = get_cell_numbers(
        multiwell_image=fluo, 
        labels=mask, 
        threshold_abs=threshold, 
        min_distance=min_distance, 
        bf=bf, 
        meta=kwargs
    )
    if save_path_csv.endswith('.csv'):
        table.to_csv(save_path_csv, index=None)
    return 0

if __name__ == "__main__":
    fire.Fire(main)
