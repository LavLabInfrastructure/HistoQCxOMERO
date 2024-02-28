import logging
import os
import sys
import numpy as np
from skimage import io, color, img_as_ubyte
from skimage.exposure import rescale_intensity
from skimage.color import separate_stains
from skimage.color import hed_from_rgb, hdx_from_rgb, fgx_from_rgb, bex_from_rgb, rbd_from_rgb
from skimage.color import gdx_from_rgb, hax_from_rgb, bro_from_rgb, bpx_from_rgb, ahx_from_rgb, \
    hpx_from_rgb  # need to load all of these in case the user selects them
from histoqc.BaseImage import strtobool, desync
from sklearn.decomposition import PCA
from scipy.spatial.transform import Rotation as R
from scipy.optimize import minimize
import time
import math
print("DeconvolutionModule.py loaded")

def _rotz(M, deg):
    """
    Helper Func: Rotate a matrix around the Z axis

    :param M: The matrix to rotate, must be 3D
    :param deg: Degrees to rotate
    """

    r = R.from_euler("z", deg, degrees=True)
    return r.apply(M)


def _angle_cost(x, U, desired):
    """
    Helper Func: Average angle of points away from the desired degrees

    :param x: The 3D matrix rotation degs around z
    :param U: The 3D matrix
    :param desired: Desired average rotation angle around the z axis
    """

    Q = _rotz(U, x)
    adjusted = np.mean(np.arctan(Q[:, 1].flatten() / Q[:, 0].flatten()) / math.pi * 180)
    return (adjusted - desired) ** 2


def _color_conv_selector(f, background_mask):
    """
    Build the M matrix required for color deconvolution. More can be found in the following papers:
    https://www.researchgate.net/publication/319879820_Quantification_of_histochemical_staining_by_color_deconvolution
    http://wwwx.cs.unc.edu/~mn/sites/default/files/macenko2009.pdf
    Notes:
        OD = optical density (the intensity of this is linear with the concentration of absorbing material)
        In a 3D OD space, the length of a vector is proportional to amount of stain
        In a 3D OD space, the relative values of the 3 channels represent the type of stain

    :param f: The input image
    :returns: The M matrix for the color deconvolution
    """

    start=time.time()
    keep_loc = background_mask.flatten() == 0
    background_mask_3d = np.repeat(background_mask[:, :, np.newaxis], 3, axis=2)
    logging.error(f"Background mask: {time.time()-start}")

    f = f * (
        np.invert(background_mask_3d.astype(bool)).astype(np.uint16)
    )  # Set all background intensity to 0
    height, width, channels = f.shape

    sample_rgb_od = np.log10((f.astype(np.float32) + 1) / 256) * -1
    odv = sample_rgb_od.reshape(-1, 3)  # flatten each layer
    print(f"ODV: {time.time()-start}")
    start=time.time()
    # Drop the background and the bottom .1% of all remaining pixel values (based on intensity)
    odv_keep = odv[keep_loc, :]
    drop_thresh = np.percentile(np.mean(odv_keep, axis=1), 0.1)
    odv_keep = odv_keep[np.mean(odv_keep, axis=1) > drop_thresh, :]

    # PCA rotated around the z index at ~15 degrees
    pca = PCA(n_components=3)
    pca_features = pca.fit_transform(odv_keep)
    avg_rotation_current = np.mean(
        np.arctan(pca_features[:, 1].flatten() / pca_features[:, 0].flatten())
        / math.pi
        * 180
    )
    guess = min(90, max(-90, 15 - avg_rotation_current))  # force between 90 and - 90
    desired_rot = minimize(
        lambda x: _angle_cost(x[0], pca_features, 15),
        [guess],
        method="SLSQP",
        bounds=[(-90, 90)],
    ).x[0]
    pca_features = _rotz(pca_features, desired_rot)

    # Normalize the rotated PCA features
    mins = np.min(pca_features, axis=0)
    maxs = np.max(pca_features, axis=0)
    pca_features = (pca_features - mins) / (maxs - mins)
    pca_features = _rotz(pca_features, 45)
    print(f"PCA: {time.time()-start}")
    start=time.time()
    # Define a function that calculates the degrees of rotation around axis b
    def cvt(a, b):
        d = a / np.sqrt((b**2) + (a**2))
        return np.arcsin(d) / math.pi * 180

    # Calculate the degrees of rotation around the first pca feature
    candidate = cvt(pca_features[:, 1], pca_features[:, 0])

    # Get threshs
    deg_v_low = np.percentile(candidate, 5)
    deg_v_high = np.percentile(candidate, 95)

    # Get loc max and min [indices of the values corresponding to the top and bottom 5% angle](changed from matlab; check)
    loc_low = np.where(candidate == max(candidate[candidate <= deg_v_low]))[0][0]
    loc_high = np.where(candidate == min(candidate[candidate >= deg_v_high]))[0][0]

    # Values to use for reference in color deconv, each represents a different stain
    stain_1_vec = odv_keep[loc_low, :]  # Adjusted pixel values low
    stain_2_vec = odv_keep[loc_high, :]  # AdBojusted pixel values high

    # Normalize
    s1 = stain_1_vec / np.linalg.norm(stain_1_vec)  # Stain One
    s2 = stain_2_vec / np.linalg.norm(stain_2_vec)  # Stain Two
    s3 = np.cross(stain_1_vec, stain_2_vec)  # Noise ratios
    s3 = s3 / np.linalg.norm(s3)
    print(f"Stain: {time.time()-start}")
    # 2d matrix
    # R G B
    # + + + Stain One Ratios
    # + + + Stain Two Ratios
    # + + + Noise
    return np.column_stack((s1, s2, s3))


async def separateStains(s, params):
    logging.info(f"{s['filename']} - \tseparateStains")
    stain = params.get("stain", "")
    logging.error(f"{s['filename']} - \tstain: {stain}")
    use_mask = strtobool(params.get("use_mask", "True"))
    
    if stain == "":
        # No stain provided, we will generate a matrix using _color_conv_selector
        logging.error(f"{s['filename']} - No stain matrix provided, generating using _color_conv_selector.")
        img = s.getImgThumb(s["image_work_size"])
        stain_matrix = _color_conv_selector(img)
    else:
        stain_matrix = getattr(sys.modules[__name__], stain, "")
        if stain_matrix == "":
            logging.error(f"{s['filename']} - Unknown stain matrix specified in DeconvolutionModule.separateStains")
            sys.exit(1)
            return

    mask = s["img_mask_use"]

    if use_mask and len(mask.nonzero()[0]) == 0: #-- lets just error check at the top if mask is empty and abort early
        for c in range(3):
            s.addToPrintList(f"deconv_c{c}_std", str(-100))
            s.addToPrintList(f"deconv_c{c}_mean", str(-100))
            io.imsave(s["outdir"] + os.sep + s["filename"] + f"_deconv_c{c}.png", img_as_ubyte(np.zeros(mask.shape)))

        logging.warning(f"{s['filename']} - DeconvolutionModule.separateStains: NO tissue "
                             f"remains detectable! Saving Black images")
        s["warnings"].append(f"DeconvolutionModule.separateStains: NO tissue "
                             f"remains detectable! Saving Black images")

        return

    tiled = strtobool(params.get("tilewise", "True"))
    dim=s.parseDim(s["image_work_size"])

    map = np.empty((dim[1],dim[0],s["image_channel_count"]), dtype=bool)
    imgs = s.tileGenerator(dim) if tiled is True else desync([s.getImgThumb(s["image_work_size"])])

    async for img, tile in imgs:
        if stain_matrix is None:
            _color_conv_selector(img, mask)
        dimg = separate_stains(img, stain_matrix)

        for c in range(0, 3):
            dc = dimg[:, :, c]

            clip_max_val = np.quantile(dc.flatten(), .99)
            dc = np.clip(dc, a_min=0, a_max=clip_max_val)


            if use_mask:
                dc_sub = dc[mask]
                dc_min = dc_sub.min()
                dc_max = dc_sub.max()

                s.addToPrintList(f"deconv_c{c}_mean", str(dc_sub.mean()))
                s.addToPrintList(f"deconv_c{c}_std", str(dc_sub.std()))
            else:
                mask = 1.0
                dc_min = dc.min()
                dc_max = dc.max()

                s.addToPrintList(f"deconv_c{c}_mean", str(dc.mean()))
                s.addToPrintList(f"deconv_c{c}_std", str(dc.std()))

            map[tile[1]:(tile[1]+tile[3]),tile[0]:(tile[0]+tile[2]), c] = (dc - dc_min) / float(dc_max - dc_min) * mask
    for c in range(0,3):
        io.imsave(s["outdir"] + os.sep + s["filename"] + f"_deconv_c{c}.png", img_as_ubyte(map[..., c]))

    return
