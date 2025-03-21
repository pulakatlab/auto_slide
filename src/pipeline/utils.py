"""
Utilities to support the main script.
"""

import matplotlib as mpl
from matplotlib import patches
from tqdm import tqdm
from numpy.random import choice
import numpy as np
import os
import cv2 as cv
import slideio
import pylab as plt
import pandas as pd
from scipy.stats import mode
from tqdm import tqdm, trange


class slide_handler():
    def __init__(
            self,
            slide_path):
        self.slide_path = slide_path
        self.slide = slideio.open_slide(slide_path, 'SVS')
        self.scene = self.slide.get_scene(0)
        self.metadata_str = self.slide.raw_metadata
        self.metadata = {}
        for item in self.metadata_str.split('|'):
            key, value = item.split('=')
            self.metadata[key.strip()] = value.strip()
        self.og_width = int(self.metadata['OriginalWidth'])
        self.og_height = int(self.metadata['OriginalHeight'])
        self.magnification = int(self.metadata['AppMag'])

def gen_step_windows(
        step_shape = None, 
        window_shape = None, 
        image_shape = None,
        overlap = 0.8
        ):
    """
    Generate steps for sliding windows in image.

    Inputs:
        step_shape: step shape
        window_shape: window shape
        image_shape: image shape
        overlap: overlap between windows

    Outputs:
        step_list: list of steps

    """
    # Make sure window_shape is not larger than image_shape
    assert (window_shape[0] <= image_shape[0]) and (window_shape[1] <= image_shape[1]), \
            f'Window shape {window_shape} must be smaller than image shape {image_shape}'

    if (step_shape is None) and (window_shape is not None) and (overlap is not None):
        step_shape = [int(i*(1-overlap)) for i in window_shape]

    step_list = []
    for i in np.arange(0, image_shape[0]-window_shape[0], step_shape[0]):
        for j in np.arange(0, image_shape[1]-window_shape[1], step_shape[1]):
            step_list.append((i, j, i+window_shape[0], j+window_shape[1]))
    return step_list

def visualize_sections(
        scene,
        step_list,
        plot_n = 10,
        down_sample = 100,
        edgecolor = 'y',
        return_image = False,
        crop_to_sections = False,
        linewidth = 1,
        ):
    """
    Visualize step window sizes.

    Inputs:
        scene: slideio scene object
        step_list: list of steps
        plot_n: number of steps to plot
        down_sample: down sample factor
        edgecolor: edge color
        return_image: return image or show it
        crop_to_sections: crop image to sections

    Outputs:
        fig: figure
        ax: axis
    """

    # If edgecolor is string, keep it the same for all windows
    # Else, if it is a vector of strings, use it to color the windows
    if isinstance(edgecolor, str):
        edgecolor = [edgecolor]*len(step_list)
        label_colormap = {edgecolor[0]: edgecolor[0]}
    elif all(isinstance(i, str) for i in edgecolor):
        assert len(edgecolor) == len(step_list)
        cmap = mpl.colormaps.get_cmap('tab10') 
        edgecolor_labels = [str(i) for i in edgecolor]
        unique_labels = np.unique(edgecolor_labels)
        label_colormap = {label: cmap(i) for i, label in enumerate(unique_labels)}
    else:
        cmap = mpl.colormaps.get_cmap('tab10')
        label_colormap = {i: cmap(i) for i in edgecolor} 

    if plot_n == -1:
        plot_n = len(step_list)

    if not crop_to_sections:
        image_shape = scene.rect[2:]
    
        # Check that windows are not larger than image
        for i, j, i_end, j_end in step_list:
            assert i_end <= image_shape[0]
            assert j_end <= image_shape[1]

        down_shape = [int(i/down_sample) for i in image_shape]

        small_image = scene.read_block(size=down_shape)
    else:
        step_list_array = np.array(step_list)
        # Get min and max x and y coordinates
        min_x = np.min(step_list_array[:, 0])
        min_y = np.min(step_list_array[:, 1])
        max_x = np.max(step_list_array[:, 2])
        max_y = np.max(step_list_array[:, 3])
        width = max_x - min_x
        height = max_y - min_y

        # Correct step_list to be relative to the new image
        step_list = step_list_array - np.array([min_x, min_y, min_x, min_y])

        image_shape = (max_x - min_x, max_y - min_y)
        down_shape = [int(i/down_sample) for i in image_shape]

        small_image = scene.read_block(
                rect=(min_x, min_y, width, height), 
                size=down_shape,
                )

    if small_image.shape[:2] != down_shape and \
            small_image.shape[:2][::-1] == image_shape:
        small_image = np.swapaxes(small_image, 0, 1) 

    fig, ax = plt.subplots(1,1)
    ax.imshow(small_image)

    mag = np.mean(
            [
            image_shape[0] / small_image.shape[0],
            image_shape[1] / small_image.shape[1],
            ]
            )

    vec_int = np.vectorize(int)
    adj_step_list = [vec_int(i/mag) for i in step_list] 

    for counter, (i, j, i_end, j_end) in enumerate(tqdm(adj_step_list)):
        if counter <= plot_n or counter == len(step_list)-1:
            # Adjust for magnification
            rect = patches.Rectangle(
                    (j, i),
                    j_end-j,
                    i_end-i,
                    linewidth=linewidth,
                    edgecolor= label_colormap[edgecolor[counter]],
                    facecolor='none')
            ax.add_patch(rect)
    # Add legend
    legend_elements = [
            patches.Patch(
                facecolor='none',
                edgecolor=label_colormap[label],
                label=label,
                )
            for label in label_colormap.keys()
            ]
    # Put legend outside of plot
    ax.legend(
            handles=legend_elements,
            bbox_to_anchor=(1.05, 1),
            loc='upper left',
            borderaxespad=0.,
            )
    if return_image:
        return fig, ax
    else:
        plt.show()
        return None, None

def get_threshold_mask(
        scene,
        down_sample = 100,
        ):
    """
    Get threshold mask for scene.
    """
    image_shape = scene.rect[2:]
    down_shape = [int(i/down_sample) for i in image_shape]
    small_image = scene.read_block(size=down_shape)
    if small_image.shape[:2] != down_shape:
        small_image = np.swapaxes(small_image, 0, 1) 

    # Get image
    gray_image = cv.cvtColor(small_image, cv.COLOR_BGR2GRAY)
    log_gray_image = np.log(gray_image)
    # Rescale to 0-255
    log_gray_image = log_gray_image - np.min(log_gray_image) 
    log_gray_image = log_gray_image / np.max(log_gray_image)
    log_gray_image = log_gray_image * 255
    log_gray_image = log_gray_image.astype(np.uint8)

    ret, thresh = cv.threshold(
            log_gray_image, 0, 255, 
            cv.THRESH_BINARY+cv.THRESH_OTSU)

    bin_image = thresh == 0

    # plt.imshow(bin_image)
    # plt.colorbar()
    # plt.show()
    return bin_image

def get_wanted_sections(
        scene,
        threshold_mask,
        step_list,
        min_fraction = 0.75,
        ):
    """
    Iterate over all sections and check if they are in the threshold mask.

    Inputs:
        scene: slideio scene object
        threshold_mask: threshold mask
        step_list: list of steps
        min_fraction: minimum fraction of pixels that must be in threshold mask

    Outputs:
        wanted_sections: list of wanted sections
    """
    mag = np.mean(
            [
            scene.rect[2] / threshold_mask.shape[0],
            scene.rect[3] / threshold_mask.shape[1],
            ]
            )

    # Adjust step list for magnification
    vec_int = np.vectorize(int) 
    adj_step_list = [vec_int(step / mag) for step in step_list]

    wanted_section_inds = []
    for counter, (i, j, i_end, j_end) in enumerate(adj_step_list):
        section = threshold_mask[i:i_end, j:j_end]
        if np.mean(section) >= min_fraction:
            wanted_section_inds.append(counter)
    wanted_sections = [step_list[ind] for ind in wanted_section_inds]

    return wanted_section_inds, wanted_sections

def get_section(
        scene,
        section,
        down_sample,
        ):
    """
    Get section from scene.

    Inputs:
        scene: slideio scene object
        section: section to single out
        down_sample: down sample factor

    Outputs:
        image: image
    """
    section_size = (section[2]-section[0], section[3]-section[1])
    down_size = (
            int(section_size[0]/down_sample), 
            int(section_size[1]/down_sample)
            )

    x = section[0]
    y = section[1]
    w = section[2]-section[0]
    h = section[3]-section[1]
    block_tuple = (x, y, w, h)
    image = scene.read_block(
            block_tuple,
            size=down_size,
            )
    return image

def single_out(
        scene, 
        section, 
        down_sample, 
        output_dir,
        ): 
    """
    Single out section.

    Inputs:
        scene: slideio scene object
        section: section to single out
        down_sample: down sample factor
        output_dir: output directory

    Outputs:
        None
    """

    image = get_section(scene, section, down_sample)

    # Save image
    image_name = (
            f'{section[0]}_{section[1]}_'
            f'{section[2]}_{section[3]}.png'
            )
    image_path = os.path.join(output_dir, image_name)
    plt.imsave(image_path, image) 

def output_sections(
        scene,
        section_list,
        output_dir,
        create_output_dir = True,
        down_sample = 100,
        random_output_n = None,
        output_type = 'write',
        ):
    """
    Output sections to output_dir.

    Inputs:
        scene: slideio scene object
        section_list: list of sections
        output_dir: output directory
        section_labels: list of section labels
        down_sample: down sample factor
        random_output_n: number of random sections to output
        output_type: 'write' or 'return'

    Outputs:
        if output_type == 'write':
            Writes images to output_dir
        if output_type == 'return':
            section_list: list of sections
            return_list: list of images
    """
    assert output_type in ['write', 'return'], 'output_type must be write or return'

    if create_output_dir:
        os.makedirs(output_dir, exist_ok=True)

    temp_single_out = lambda section: single_out(
            scene, section, down_sample, output_dir) 

    if random_output_n is not None:
        assert isinstance(random_output_n, int) 
        assert random_output_n <= len(section_list)
        wanted_inds = np.random.choice(
                np.arange(len(section_list)),
                size=random_output_n,
                replace=False,
                )
        section_list = [section_list[ind] for ind in wanted_inds] 

    if output_type == 'write':
        for section in tqdm(section_list):
            temp_single_out(section)
    elif output_type == 'return':
        return_list = []
        for section in tqdm(section_list):
            return_list.append(get_section(scene, section, down_sample))
        return section_list, return_list

def annotate_sections(
        scene,
        label_mask,
        metadata,
        step_list,
        ):
    """
    Iterate over all sections and check what tissue they belong to

    Inputs:
        scene: slideio scene object
        label_mask: label mask
        metadata: metadata
        step_list: list of steps

    Outputs:
        out_frame: dataframe with section information
    """
    mag = np.mean(
            [
            scene.rect[2] / label_mask.shape[0],
            scene.rect[3] / label_mask.shape[1],
            ]
            )

    # Adjust step list for magnification
    vec_int = np.vectorize(int) 
    adj_step_list = [vec_int(step / mag) for step in step_list]

    # Use area of mask occupuied by section to determine label value
    label_values = []
    for section in adj_step_list:
        section_mask = label_mask[
                section[0]:section[2],
                section[1]:section[3],
                ]
        all_labels = section_mask[section_mask > 0]
        if len(all_labels) > 0:
            # label = int(np.mean(all_labels))
            label = int(mode(all_labels)[0])
        else:
            label = np.nan
        label_values.append(label)

    section_frame = pd.DataFrame(
            dict(
                section_bounds = step_list,
                label_values = label_values,
                )
            )

    fin_frame = section_frame.merge( 
            metadata[['tissue_num', 'tissue_type']],
            left_on='label_values',
            right_on='tissue_num',
            how='left',
            ).drop(columns=['tissue_num'])

    fin_frame.dropna(inplace=True)

    return fin_frame

def write_out_images(
        img_list,
        section_frame,
        output_dir,
        ):
    """
    Write out images to output_dir, labelled by section_labels.
    If match_pattern is not None, only images with section_labels

    Inputs:
        img_list: list of images
        section_frame: dataframe with section information
        output_dir: output directory
        match_pattern: pattern to match section_labels

    Outputs:
        None
    """
    assert len(img_list) == len(section_frame), \
            'img_list and section_frame must be same length'

    section_labels = section_frame['section_labels'].values
    section_hashes = section_frame['section_hash'].values

    section_names = [f'{label}_{hash}' for label, hash in zip(
        section_labels, section_hashes)]

    for img, name in tqdm(zip(img_list, section_names)):
        img_path = os.path.join(output_dir, f'{name}.png')
        plt.imsave(img_path, img)


def generate_negative_samples(img, mask):
    """
    Generate negative samples by creating regions without vessels.
    
    Inputs:
        img: Original image
        mask: Original mask
        
    Outputs:
        neg_img: Image with negative samples
        neg_mask: Mask for negative samples
    """

    # Create a binary mask where vessels are present
    vessel_present = mask > 0
    
    # Create a copy of the original image
    neg_img = img.copy()

    # Make label area white
    neg_img[vessel_present, :] = 255
    
    # Create an empty mask (all zeros)
    neg_mask = np.zeros_like(mask)
    
    return neg_img, neg_mask 

def generate_artificial_vessels(img, mask): 
    """
    Generate artificial vessel samples by warping and transforming existing vessels.
    
    Inputs:
        img: Original image
        mask: Original mask
        
    Outputs:
        art_img: Image with artificial vessels 
        art_mask: Mask for artificial vessels
    """
    
    # Check if there are any vessels in the mask
    if np.max(mask) == 0:
        return None, None
    
    # Create copies of the original image and mask
    art_img = img.copy()

    img_len = img.shape[0]
    
    # Find vessel regions
    vessel_regions = mask > 0
    
    if np.sum(vessel_regions) > 0:
        # Extract vessel pixels
        vessel_img = np.zeros_like(img)
        vessel_img[vessel_regions] = img[vessel_regions]
        
        # Apply random transformations
        # 1. Random rotation
        angle = np.random.uniform(-180, 180)
        rows, cols = vessel_img.shape[:2]
        M = cv.getRotationMatrix2D((cols/2, rows/2), angle, 1)
        
        # 2. Random scaling
        scale = np.random.uniform(0.25, 3)
        M[:, 0:2] = M[:, 0:2] * scale
        # x_scale = np.random.uniform(0.8, 1.2)
        # y_scale = np.random.uniform(0.8, 1.2)
        # M[:, 0] = M[:, 0] * x_scale
        # M[:, 1] = M[:, 1] * y_scale
        
        # 3. Random translation
        translation_factor = 0.5
        tx = np.random.randint(-img_len * translation_factor, img_len * translation_factor)
        ty = np.random.randint(-img_len * translation_factor, img_len * translation_factor)
        # tx = np.random.randint(-20, 20)
        # ty = np.random.randint(-20, 20)
        M[0, 2] += tx
        M[1, 2] += ty
        
        # Apply transformations
        warped_vessel = cv.warpAffine(vessel_img, M, (cols, rows), borderMode=cv.BORDER_REFLECT)
        warped_mask = cv.warpAffine(mask.astype(np.uint8), M, (cols, rows), borderMode=cv.BORDER_REFLECT)
        
        # Warping introduces variability, so assign value to closest in original
        orig_mask_values = np.unique(mask)
        warped_mask_list = []
        for val in orig_mask_values:
            this_mask = warped_mask.copy()
            this_mask[warped_mask != val] = 0
            this_mask = this_mask.astype(float)
            val = val.astype(float)
            max_orig = np.max(orig_mask_values).astype(float)
            this_mask = (this_mask / val) * (val + max_orig + 1)
            warped_mask_list.append(this_mask)
        warped_mask = np.nansum(warped_mask_list, axis=0)
        warped_mask = warped_mask.astype(np.uint16)

        fin_mask = mask.copy()
        fin_mask = fin_mask.astype(np.uint16)
        fin_mask[warped_mask > 0] = warped_mask[warped_mask > 0]

        # Renormalize
        fin_mask = fin_mask / np.max(fin_mask) * 255    
        art_mask = fin_mask.astype(np.uint8)

        fig, ax = plt.subplots(1, 3)
        ax[0].imshow(mask)
        ax[1].imshow(warped_mask)
        ax[2].imshow(fin_mask)
        plt.show()

        # # Renormalize mask
        # warped_mask = warped_mask / np.max(warped_mask) * 255
        #
        # warped_mask = (warped_mask > 0).astype(np.uint8) * 255
        # 
        # # Place the warped vessel in a different location
        # # Find non-vessel regions in the original image
        # non_vessel = ~vessel_regions
        # 
        # # Create a new mask for the artificial vessel
        # art_mask = warped_mask
        # 
        # Combine the original image with the warped vessel
        art_img = img.copy()
        art_img[warped_mask > 0] = warped_vessel[warped_mask > 0]
        
    return art_img, art_mask 

def augment_dataset(images, masks, neg_ratio=0.2, art_ratio=0.5):
    """
    Augment a dataset with negative samples and artificial vessels.
    
    Inputs:
        images: List of original images
        masks: List of original masks
        neg_ratio: Ratio of negative samples to add
        art_ratio: Ratio of artificial vessel samples to add
        
    Outputs:
        aug_images: List of augmented images
        aug_masks: List of augmented masks
    """
    aug_images = images.copy()
    aug_masks = masks.copy()
    
    num_orig = len(images)
    num_neg = int(num_orig * neg_ratio)
    num_art = int(num_orig * art_ratio)
    
    # Generate negative samples
    print('Generating negative samples...')
    for i in trange(min(num_neg, num_orig)):
        neg_imgs, neg_msks = generate_negative_samples(images[i], masks[i])
        aug_images.append(neg_imgs)
        aug_masks.append(neg_msks)
    
    # Generate artificial vessel samples
    print('Generating artificial vessel samples...')
    for i in trange(min(num_art, num_orig)):
        art_imgs, art_msks = generate_artificial_vessels(images[i], masks[i])
        aug_images.append(art_imgs)
        aug_masks.append(art_msks)
    
    return aug_images, aug_masks

def gen_negative_masks(neg_dir, output_dir):
    neg_image_names = os.listdir(neg_dir)
    out_mask_names = [f'{name.split(".")[0]}_mask.png' for name in neg_image_names]
    for img_name, mask_name in zip(neg_image_names, out_mask_names):
        img_path = os.path.join(neg_dir, img_name)
        mask_path = os.path.join(output_dir, mask_name)
        img = plt.imread(img_path)
        mask = np.zeros_like(img)
        plt.imsave(mask_path, mask)


# neg_dir = '/home/abuzarmahmood/projects/auto_slide/data/labelled_images/negative_images'
# out_dir = '/home/abuzarmahmood/projects/auto_slide/data/labelled_images/negative_masks'
# gen_negative_masks(neg_dir, out_dir)
