import logging
from pickletools import uint8
import numpy as np
from skimage.filters import sobel
from skimage.color import convert_colorspace, rgb2gray
from histoqc.BaseImage import strtobool, desync


async def getBrightnessGray(s, params):
    prefix = params.get("prefix", None)
    prefix = prefix+"_" if prefix else ""
    logging.info(f"{s['filename']} - \tgetContrast:{prefix}")

    limit_to_mask = strtobool(params.get("limit_to_mask", "True"))
    invert = strtobool(params.get("invert", "False"))
    mask_name = params.get("mask_name", "img_mask_use")

    tiled = strtobool(params.get("tilewise", "true"))
    dim=s.parseDim(s["image_work_size"])

    map = np.empty((0),dtype=np.uint8)
    imgs = s.tileGenerator(dim) if tiled else desync([s.getImgThumb(s["image_work_size"])])

    async for img, tile in imgs:
        img_g = rgb2gray(img)
        if limit_to_mask:
            mask = s[mask_name][tile[1]:(tile[1]+tile[3]),tile[0]:(tile[0]+tile[2])] if not invert else ~s[mask_name][tile[1]:(tile[1]+tile[3]),tile[0]:(tile[0]+tile[2])]
            mask = img_g[mask]
            map = np.concatenate((map, mask))
        else:
            map = np.concatenate((map, img_g.ravel()))
    if map.size == 0:
        map = np.array(-100)
    s.addToPrintList(f"{prefix}grayscale_brightness", str(map.mean()))
    s.addToPrintList(f"{prefix}grayscale_brightness_std", str(map.std()))

    return


async def getBrightnessByChannelinColorSpace(s, params):
    prefix = params.get("prefix", None)
    prefix = prefix + "_" if prefix else ""

    logging.info(f"{s['filename']} - \tgetContrast:{prefix}")

    to_color_space = params.get("to_color_space", "RGB")
    invert = strtobool(params.get("invert", "False"))

    limit_to_mask = strtobool(params.get("limit_to_mask", "True"))
    mask_name = params.get("mask_name", "img_mask_use")

    tiled = strtobool(params.get("tilewise", "True"))
    dim=s.parseDim(s["image_work_size"])
    chanCount=s["image_channel_count"]

    map = []
    for i in range(chanCount): map.append(np.empty((0), np.uint8))

    imgs = s.tileGenerator(dim) if tiled else desync([s.getImgThumb(s["image_work_size"])])

    suffix = "_" + to_color_space
    async for img, tile in imgs:
        if (to_color_space != "RGB"):
            img = convert_colorspace(img, "RGB", to_color_space)

        for chan in range(chanCount):
            vals = img[:, :, chan]
            
            if (limit_to_mask):
                mask = s[mask_name][tile[1]:(tile[1]+tile[3]),tile[0]:(tile[0]+tile[2])] if not invert else ~s[mask_name][tile[1]:(tile[1]+tile[3]),tile[0]:(tile[0]+tile[2])]
                mask = vals[mask]
                map[chan] = np.concatenate((map[chan], mask))
            else:
                map[chan] = np.concatenate((map[chan], vals.ravel()))

    
    for chan in range(chanCount):
        if map[chan].size == 0:
            map[chan] = np.array(-100)
        s.addToPrintList(f"{prefix}chan{chan+1}_brightness{suffix}", str(map[chan].mean()))
        s.addToPrintList(f"{prefix}chan{chan+1}_brightness_std{suffix}", str(map[chan].std()))
    return


# cannot be tiled
def getContrast(s, params):
    prefix = params.get("prefix", None)
    prefix = prefix + "_" if prefix else ""

    logging.info(f"{s['filename']} - \tgetContrast:{prefix}")
    limit_to_mask = strtobool(params.get("limit_to_mask", "True"))
    mask_name = params.get("mask_name", "img_mask_use")

    invert = strtobool(params.get("invert", "False"))


    img = s.getImgThumb(s["image_work_size"])[0]
    img = rgb2gray(img)

    sobel_img = sobel(img) ** 2

    if limit_to_mask:

        mask = s[mask_name] if not invert else ~s[mask_name]

        sobel_img = sobel_img[mask]
        img = img[s["img_mask_use"]]

    if img.size == 0: # need a check to ensure that mask wasn't empty AND limit_to_mask is true, still want to
                      # produce metrics for completeness with warning

        s.addToPrintList(f"{prefix}tenenGrad_contrast", str(-100))
        s.addToPrintList(f"{prefix}michelson_contrast", str(-100))
        s.addToPrintList(f"{prefix}rms_contrast", str(-100))


        logging.warning(f"{s['filename']} - After BrightContrastModule.getContrast: NO tissue "
                        f"detected, statistics are impossible to compute, defaulting to -100 !")
        s["warnings"].append(f"After BrightContrastModule.getContrast: NO tissue remains "
                             f"detected, statistics are impossible to compute, defaulting to -100 !")

        return


    # tenenGrad - Note this must be performed on full image and then subsetted if limiting to mask
    tenenGrad_contrast = np.sqrt(np.sum(sobel_img)) / img.size
    s.addToPrintList(f"{prefix}tenenGrad_contrast", str(tenenGrad_contrast))

    # Michelson contrast
    max_img = img.max()
    min_img = img.min()
    contrast = (max_img - min_img) / (max_img + min_img)
    s.addToPrintList(f"{prefix}michelson_contrast", str(contrast))

    # RMS contrast
    rms_contrast = np.sqrt(pow(img - img.mean(), 2).sum() / img.size)
    s.addToPrintList(f"{prefix}rms_contrast", str(rms_contrast))

    return
