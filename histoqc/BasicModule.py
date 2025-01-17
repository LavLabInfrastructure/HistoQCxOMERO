import logging
import os
from histoqc.BaseImage import printMaskHelper, strtobool
from histoqc.OmeroModule import uploadAsPolygons
from skimage.morphology import remove_small_objects, binary_opening, disk
from skimage import io, img_as_ubyte


def getBasicStats(s, params):
    logging.info(f"{s['filename']} - \tgetBasicStats")
    oim=s["omero_image_meta"]
    ops=s["omero_pixel_store"]
    imgType=oim.getInstrument().getMicroscope().getManufacturer() if oim.getInstrument() else "NA"
    imgLevels=ops.getResolutionLevels() if ops.getResolutionLevels() else "NA"
    imgX=oim.getSizeX() if oim.getSizeX() else "NA"
    imgY=oim.getSizeY() if oim.getSizeY() else "NA"
    imgXRes=oim.getPixelSizeX() if oim.getPixelSizeX() else "NA"
    imgYRes=oim.getPixelSizeY() if oim.getPixelSizeY() else "NA"
    comment=oim.getDescription() if oim.getDescription() else "NA"
    s.addToPrintList("type", imgType)
    s.addToPrintList("levels", imgLevels)
    s.addToPrintList("height", imgY)
    s.addToPrintList("width", imgX)
    s.addToPrintList("mpp_x", imgXRes)
    s.addToPrintList("mpp_y", imgYRes)
    s.addToPrintList("comment", comment.replace("\n", " ").replace("\r", " "))
    return


def finalComputations(s, params):
    mask = s["img_mask_use"]
    s.addToPrintList("pixels_to_use", str(len(mask.nonzero()[0])))


def finalProcessingSpur(s, params):
    logging.info(f"{s['filename']} - \tfinalProcessingSpur")
    disk_radius = int(params.get("disk_radius", "25"))
    selem = disk(disk_radius)
    mask = s["img_mask_use"]
    mask_opened = binary_opening(mask, selem)
    mask_spur = ~mask_opened & mask

    if strtobool(params.get("upload", "False")):
        uploadAsPolygons(s, mask_spur, "spur")
    else:
        io.imsave(s["outdir"] + os.sep + s["filename"] + "_spur.png", img_as_ubyte(mask_spur))

    prev_mask = s["img_mask_use"]
    s["img_mask_use"] = mask_opened
# spur
    s.addToPrintList("spur_pixels",
                     printMaskHelper(params.get("mask_statistics", s["mask_statistics"]), prev_mask, s["img_mask_use"]))

    if len(s["img_mask_use"].nonzero()[0]) == 0:  # add warning in case the final tissue is empty
        logging.warning(
            f"{s['filename']} - After BasicModule.finalProcessingSpur NO tissue remains detectable! Downstream modules likely to be incorrect/fail")
        s["warnings"].append(
            f"After BasicModule.finalProcessingSpur NO tissue remains detectable! Downstream modules likely to be incorrect/fail")


def finalProcessingArea(s, params):
    logging.info(f"{s['filename']} - \tfinalProcessingArea")
    area_thresh = int(params.get("area_threshold", "1000"))
    mask = s["img_mask_use"]

    mask_opened = remove_small_objects(mask, min_size=area_thresh)
    mask_removed_area = ~mask_opened & mask

    if strtobool(params.get("upload", "False")):
        uploadAsPolygons(s, mask_removed_area, "area thresh")
    else:
        io.imsave(s["outdir"] + os.sep + s["filename"] + "_areathresh.png", img_as_ubyte(mask_removed_area))

    prev_mask = s["img_mask_use"]
    s["img_mask_use"] = mask_opened > 0

    s.addToPrintList("areaThresh",
                     printMaskHelper(params.get("mask_statistics", s["mask_statistics"]), prev_mask, s["img_mask_use"]))

    if len(s["img_mask_use"].nonzero()[0]) == 0:  # add warning in case the final tissue is empty
        logging.warning(
            f"{s['filename']} - After BasicModule.finalProcessingArea NO tissue remains detectable! Downstream modules likely to be incorrect/fail")
        s["warnings"].append(
            f"After BasicModule.finalProcessingArea NO tissue remains detectable! Downstream modules likely to be incorrect/fail")
