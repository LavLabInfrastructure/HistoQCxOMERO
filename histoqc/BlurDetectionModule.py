import logging
import os

import skimage
from histoqc.BaseImage import printMaskHelper, strtobool, desync
from histoqc.OmeroModule import uploadAsPolygons
from skimage import io, img_as_ubyte, morphology, measure
from skimage.color import rgb2gray
import numpy as np

import matplotlib.pyplot as plt


# Analysis of focus measure operators for shape-from-focus
# Said Pertuza,, Domenec Puiga, Miguel Angel Garciab, 2012
# https://pdfs.semanticscholar.org/8c67/5bf5b542b98bf81dcf70bd869ab52ab8aae9.pdf


async def identifyBlurryRegions(s, params):
    logging.info(f"{s['filename']} - \tidentifyBlurryRegions")

    blur_radius = int(params.get("blur_radius", 7))
    blur_threshold = float(params.get("blur_threshold", .1))

    tiled = strtobool(params.get("tilewise", "True"))
    work_size=params.get("image_work_size", "2.5x")
    work_dim=s.parseDim(work_size)
    save_dim=s.parseDim(s["image_work_size"])
    scale = (round(save_dim[0]/work_dim[0], 1), round(save_dim[1]/work_dim[1], 1))

    map = np.empty((save_dim[1],save_dim[0]), dtype=bool)
    imgs = s.tileGenerator(s.parseDim(work_size)) if tiled else desync([s.getImgThumb(work_size)])
    async for img, tile in imgs:
        img = rgb2gray(img)
        
        img_laplace = np.abs(skimage.filters.laplace(img))
        mask_large = skimage.filters.gaussian(img_laplace, sigma=blur_radius) <= blur_threshold

        tile = (int(tile[0]*scale[0]), int(tile[1]*scale[1]), int(tile[2]*scale[0]), int(tile[3]*scale[1]))
        mask = skimage.transform.resize(mask_large,(tile[3],tile[2]))
        map[tile[1]:(tile[1]+tile[3]),tile[0]:(tile[0]+tile[2])] = s["img_mask_use"][tile[1]:(tile[1]+tile[3]),tile[0]:(tile[0]+tile[2])] & (mask > 0)
    if strtobool(params.get("upload", "False")):
        uploadAsPolygons(s, map, "blurry")
    else:
        io.imsave(s["outdir"] + os.sep + s["filename"] + "_blurry.png", img_as_ubyte(map))
    s["img_mask_blurry"] = (map * 255) > 0

    prev_mask = s["img_mask_use"]
    s["img_mask_use"] = s["img_mask_use"] & ~s["img_mask_blurry"]

    rps = measure.regionprops(morphology.label(map))
    if rps:
        areas = np.asarray([rp.area for rp in rps])
        nobj = len(rps)
        area_max = areas.max()
        area_mean = areas.mean()
    else:
        nobj = area_max = area_mean = 0


    s.addToPrintList("blurry_removed_num_regions", str(nobj))
    s.addToPrintList("blurry_removed_mean_area", str(area_mean))
    s.addToPrintList("blurry_removed_max_area", str(area_max))


    s.addToPrintList("blurry_removed_percent",
                     printMaskHelper(params.get("mask_statistics", s["mask_statistics"]), prev_mask, s["img_mask_use"]))

    if len(s["img_mask_use"].nonzero()[0]) == 0:  # add warning in case the final tissue is empty
        logging.warning(
            f"{s['filename']} - After BlurDetectionModule.identifyBlurryRegions NO tissue remains detectable! Downstream modules likely to be incorrect/fail")
        s["warnings"].append(
            f"After BlurDetectionModule.identifyBlurryRegions NO tissue remains detectable! Downstream modules likely to be incorrect/fail")


    return
