import logging
import os
import asyncio
import numpy as np
from histoqc.BaseImage import printMaskHelper, desync, strtobool
from histoqc.OmeroModule import uploadAsPolygons
from skimage import io, color, img_as_ubyte
from skimage.filters import threshold_otsu, rank
from skimage.morphology import disk
from sklearn.cluster import KMeans
from skimage import exposure



async def getIntensityThresholdOtsu(s, params):
    logging.info(f"{s['filename']} - \tLightDarkModule.getIntensityThresholdOtsu")
    name = params.get("name", "classTask")    
    local = strtobool(params.get("local", "False"))
    radius = float(params.get("radius", 15))
    selem = disk(radius)

    tiled = strtobool(params.get("tilewise", "True"))
    dim=s.parseDim(s["image_work_size"])

    map = np.empty((dim[1],dim[0]), dtype=bool)
    imgs = s.tileGenerator(dim) if tiled is True else desync([s.getImgThumb(s["image_work_size"])])

    async for img, tile in imgs:
        img = color.rgb2gray(img)
    
        if local:
            thresh = rank.otsu(img, selem)
        else:
            thresh = threshold_otsu(img)

        map[tile[1]:(tile[1]+tile[3]),tile[0]:(tile[0]+tile[2])] = img < thresh

    s["img_mask_" + name] = map > 0
    if strtobool(params.get("invert", "False")):
        s["img_mask_" + name] = ~s["img_mask_" + name]

    io.imsave(s["outdir"] + os.sep + s["filename"] + "_" + name + ".png", img_as_ubyte(s["img_mask_" + name]))

    prev_mask = s["img_mask_use"]
    s["img_mask_use"] = s["img_mask_use"] & s["img_mask_" + name]

    s.addToPrintList(name,
                     printMaskHelper(params.get("mask_statistics", s["mask_statistics"]), prev_mask, s["img_mask_use"]))

    if len(s["img_mask_use"].nonzero()[0]) == 0:  # add warning in case the final tissue is empty
        logging.warning(f"{s['filename']} - After LightDarkModule.getIntensityThresholdOtsu:{name} NO tissue remains "
                        f"detectable! Downstream modules likely to be incorrect/fail")
        s["warnings"].append(f"After LightDarkModule.getIntensityThresholdOtsu:{name} NO tissue remains detectable! "
                             f"Downstream modules likely to be incorrect/fail")

    return
    

async def getIntensityThresholdPercent(s, params):
    name = params.get("name", "classTask")
    logging.info(f"{s['filename']} - \tLightDarkModule.getIntensityThresholdPercent:\t {name}")

    lower_thresh = float(params.get("lower_threshold", -float("inf")))
    upper_thresh = float(params.get("upper_threshold", float("inf")))

    lower_var = float(params.get("lower_variance", -float("inf")))
    upper_var = float(params.get("upper_variance", float("inf")))

    tiled = strtobool(params.get("tilewise", "True"))
    dim=s.parseDim(s["image_work_size"])

    map = np.empty((dim[1],dim[0]), dtype=bool)
    imgs = s.tileGenerator(dim) if tiled is True else desync([s.getImgThumb(s["image_work_size"])])

    async for img, tile in imgs:
        img_var = img.std(axis=2)

        map_var = np.bitwise_and(img_var > lower_var, img_var < upper_var)

        img = color.rgb2gray(img)
        local_map = np.bitwise_and(img > lower_thresh, img < upper_thresh)

        local_map = np.bitwise_and(local_map, map_var)
        
        # insert findings to map
        map[tile[1]:(tile[1]+tile[3]),tile[0]:(tile[0]+tile[2])] = local_map

    s["img_mask_" + name] = map > 0

    if strtobool(params.get("invert", "False")):
        s["img_mask_" + name] = ~s["img_mask_" + name]

    prev_mask = s["img_mask_use"]
    s["img_mask_use"] = s["img_mask_use"] & s["img_mask_" + name]

    if strtobool(params.get("upload", "False")):
        uploadAsPolygons(s, s["img_mask_" + name], name)
    else:
        io.imsave(s["outdir"] + os.sep + s["filename"] + "_" + name + ".png", img_as_ubyte(prev_mask & ~s["img_mask_" + name]))

    s.addToPrintList(name,
                     printMaskHelper(params.get("mask_statistics", s["mask_statistics"]), prev_mask, s["img_mask_use"]))

    if len(s["img_mask_use"].nonzero()[0]) == 0:  # add warning in case the final tissue is empty
        logging.warning(f"{s['filename']} - After LightDarkModule.getIntensityThresholdPercent:{name} NO tissue "
                        f"remains detectable! Downstream modules likely to be incorrect/fail")
        s["warnings"].append(f"After LightDarkModule.getIntensityThresholdPercent:{name} NO tissue remains "
                             f"detectable! Downstream modules likely to be incorrect/fail")
    return






async def removeBrightestPixels(s, params):
    logging.info(f"{s['filename']} - \tLightDarkModule.removeBrightestPixels")

    # lower_thresh = float(params.get("lower_threshold", -float("inf")))
    # upper_thresh = float(params.get("upper_threshold", float("inf")))
    #
    # lower_var = float(params.get("lower_variance", -float("inf")))
    # upper_var = float(params.get("upper_variance", float("inf")))

    tiled = strtobool(params.get("tilewise", "True"))
    dim=s.parseDim(s["image_work_size"])

    map = np.empty((dim[1],dim[0]), dtype=bool)
    imgs = s.tileGenerator(dim) if tiled is True else desync([s.getImgThumb(s["image_work_size"])])

    async for img, tile in imgs:
        img = color.rgb2gray(img)

        kmeans = KMeans(n_clusters=3,  n_init=1).fit(img.reshape([-1, 1]))
        brightest_cluster = np.argmax(kmeans.cluster_centers_)
        darkest_point_in_brightest_cluster = (img.reshape([-1, 1])[kmeans.labels_ == brightest_cluster]).min()

        map[tile[1]:(tile[1]+tile[3]),tile[0]:(tile[0]+tile[2])] = img > darkest_point_in_brightest_cluster
    
    s["img_mask_bright"] = map



    if strtobool(params.get("invert", "False")):
        s["img_mask_bright"] = ~s["img_mask_bright"]

    prev_mask = s["img_mask_use"]
    s["img_mask_use"] = s["img_mask_use"] & s["img_mask_bright"]

    io.imsave(s["outdir"] + os.sep + s["filename"] + "_bright.png", img_as_ubyte(prev_mask & ~s["img_mask_bright"]))

    s.addToPrintList("brightestPixels",
                     printMaskHelper(params.get("mask_statistics", s["mask_statistics"]), prev_mask, s["img_mask_use"]))

    if len(s["img_mask_use"].nonzero()[0]) == 0:  # add warning in case the final tissue is empty
        logging.warning(f"{s['filename']} - After LightDarkModule.removeBrightestPixels NO tissue "
                        f"remains detectable! Downstream modules likely to be incorrect/fail")
        s["warnings"].append(f"After LightDarkModule.removeBrightestPixels NO tissue remains "
                             f"detectable! Downstream modules likely to be incorrect/fail")

    return



async def minimumPixelIntensityNeighborhoodFiltering(s,params):
    logging.info(f"{s['filename']} - \tLightDarkModule.minimumPixelNeighborhoodFiltering")
    disk_size = int(params.get("disk_size", 10000))
    threshold = int(params.get("upper_threshold", 200))

    tiled = strtobool(params.get("tilewise", "True"))
    dim=s.parseDim(s["image_work_size"])

    map = np.empty((dim[1],dim[0]), dtype=bool)
    imgs = s.tileGenerator(dim) if tiled is True else desync([s.getImgThumb(s["image_work_size"])])

    async for img, tile in imgs:
        img = color.rgb2gray(img)
        img = (img * 255).astype(np.uint8)
        selem = disk(disk_size)

        imgfilt = rank.minimum(img, selem)
        map[tile[1]:(tile[1]+tile[3]),tile[0]:(tile[0]+tile[2])] = imgfilt > threshold
    
    s["img_mask_bright"] = map

    if strtobool(params.get("invert", "True")):
        s["img_mask_bright"] = ~s["img_mask_bright"]

    prev_mask = s["img_mask_use"]
    s["img_mask_use"] = s["img_mask_use"] & s["img_mask_bright"]

    io.imsave(s["outdir"] + os.sep + s["filename"] + "_bright.png", img_as_ubyte(prev_mask & ~s["img_mask_bright"]))

    s.addToPrintList("brightestPixels",
                     printMaskHelper(params.get("mask_statistics", s["mask_statistics"]), prev_mask, s["img_mask_use"]))

    if len(s["img_mask_use"].nonzero()[0]) == 0:  # add warning in case the final tissue is empty
        logging.warning(f"{s['filename']} - After LightDarkModule.minimumPixelNeighborhoodFiltering NO tissue "
                        f"remains detectable! Downstream modules likely to be incorrect/fail")
        s["warnings"].append(f"After LightDarkModule.minimumPixelNeighborhoodFiltering NO tissue remains "
                             f"detectable! Downstream modules likely to be incorrect/fail")

    return

async def saveEqualisedImage(s,params):
    logging.info(f"{s['filename']} - \tLightDarkModule.saveEqualisedImage")

    tiled = strtobool(params.get("tilewise", "True"))
    dim=s.parseDim(s["image_work_size"])

    out = np.empty((dim[1],dim[0]), dtype=bool)
    imgs = s.tileGenerator(dim) if tiled is True else desync([s.getImgThumb(s["image_work_size"])])

    async for img, tile in imgs:
        img = color.rgb2gray(img)

        out[tile[1]:(tile[1]+tile[3]),tile[0]:(tile[0]+tile[2])] = exposure.equalize_hist((img*255).astype(np.uint8))
    io.imsave(s["outdir"] + os.sep + s["filename"] + "_equalized_thumb.png", img_as_ubyte(out))

    return
