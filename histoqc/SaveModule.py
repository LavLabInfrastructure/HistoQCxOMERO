import logging
import os
import math
import struct 
from distutils.util import strtobool
import numpy as np
from skimage import measure, io, color, img_as_ubyte, morphology
from omero_rois import mask_from_binary_image
from omero.model import RoiI, PolygonI, AffineTransformI
from omero.rtypes import rint, rdouble, rstring

def blend2Images(img, mask):
    if (img.ndim == 3):
        img = color.rgb2gray(img)
    if (mask.ndim == 3):
        mask = color.rgb2gray(mask)
    img = img[:, :, None] * 1.0  # can't use boolean
    mask = mask[:, :, None] * 1.0
    out = np.concatenate((mask, img, mask), 2)
    return out


def saveFinalMask(s, params):
    logging.info(f"{s['filename']} - \tsaveUsableRegion")

    mask = s["img_mask_use"]
    for mask_force in s["img_mask_force"]:
        mask[s[mask_force]] = 0

    io.imsave(s["outdir"] + os.sep + s["filename"] + "_mask_use.png", img_as_ubyte(mask))

    if strtobool(params.get("use_mask", "True")):  # should we create and save the fusion mask?
        img = s.getImgThumb(s["image_work_size"])
        out = blend2Images(img, mask)
        io.imsave(s["outdir"] + os.sep + s["filename"] + "_fuse.png", img_as_ubyte(out))

    return


def saveThumbnails(s, params):
    logging.info(f"{s['filename']} - \tsaveThumbnail")
    # we create 2 thumbnails for usage in the front end, one relatively small one, and one larger one
    img = s.getImgThumb(params.get("image_work_size", "1.25x"))
    io.imsave(s["outdir"] + os.sep + s["filename"] + "_thumb.png", img)

    img = s.getImgThumb(params.get("small_dim", 500))
    io.imsave(s["outdir"] + os.sep + s["filename"] + "_thumb_small.png", img)
    return


def rgba_to_int(red, green, blue, alpha=255):
    """ Return the color as an Integer in RGBA encoding """
    return int.from_bytes([red, green, blue, alpha],
                      byteorder='big', signed=True)


    # We have a helper function for creating an ROI and linking it to new shapes
def __create_roi(s, oim, shapes):
    ous=s["omero_conn_handle"].getUpdateService()
    # create an ROI, link it to Image
    roi = RoiI()
    # use the omero.model.ImageI that underlies the 'image' wrapper
    roi.setImage(oim._obj)
    for shape in shapes:
        roi.addShape(shape)
    # Save the ROI (saves any linked shapes too)
    ous.saveAndReturnObject(roi)
    ous.close()
    return
    

def uploadMasks(s, params):
    logging.info(f"{s['filename']} - \tuploadMasks")
    od=s["outdir"]
    oim=s["omero_image_meta"]
    conn=s["omero_conn_handle"]
    for name in os.listdir(od):
        file=os.path.join(od, name)
        if os.path.isfile(file):
            if name.endswith('.tsv'):
                annot=conn.createFileAnnfromLocalFile(file)
                for arg in s["orig_command"]:
                    if arg == "Project:*" or arg == "Dataset:*":
                        splitted = arg.strip().split(":")
                        conn.getObject(splitted[0], splitted[1]).linkAnnotation(annot)
                    elif arg.is_digit():
                        oim.linkAnnotation(annot)
                    else :
                        logging.warn("wasn't sure how to handle" + file)
            else: 
                mask=mask_from_binary_image(io.imread(file).astype(bool), text=os.path.basename(file)) # read img, put in uint8, turn to byte array, wrap in a Mask obj
                mask.setWidth(rdouble((oim.getSizeX()-mask.getX()._val))); mask.setHeight(rdouble((oim.getSizeY()-mask.getY()._val)))
                __create_roi(s, oim, [mask]) # upload mask
        else:
            logging.warn(mask.tostring() + ": should have been a file, but was not")
    return


def get_longest_contour(contours):
    contour = contours[0]
    for c in contours:
        if len(c) > len(contour):
            contour = c
    return contour

def scale(A, B):     # fill A with B scaled by kx/ky
    Y = A.shape[0]
    X = A.shape[1]
    ky = Y//B.shape[0]
    kx = X//B.shape[1] # average
    for y in range(0, ky):
        for x in range(0, kx):
            A[y:Y-ky+y:ky, x:X-kx+x:kx] = B


def __create_polygon(contour, x_offset=0, y_offset=0, z=None, t=None, comment=None):
    """ points is 2D list of [[x, y], [x, y]...]"""
    #TODO:contour scaling 
    contour=contour*32
    stride = 64
    coords = []
    # points in contour are adjacent pixels, which is too verbose
    # take every nth point
    for count, xy in enumerate(contour):
        if count%stride == 0:
            coords.append(xy)
    if len(coords) < 2:
        return
    points = ["%s,%s" % (xy[1] + x_offset, xy[0] + y_offset) for xy in coords]
    points = ", ".join(points)
    polygon = PolygonI()
    if z is not None:
        polygon.theZ = rint(z)
    if t is not None:
        polygon.theT = rint(t)
    if comment is not None:
        polygon.setTextValue(rstring(comment))
    polygon.strokeColor = rint(rgba_to_int(255, 255, 255))
    # points = "10,20, 50,150, 200,200, 250,75"
    polygon.points = rstring(points)
    return polygon


def uploadMasksAsPolygons(s, params):
    logging.info(f"{s['filename']} - \tuploadMasksAsPolygons")
    od=s["outdir"]
    oim=s["omero_image_meta"]
    conn=s["omero_conn_handle"]
    for name in os.listdir(od):
        file=os.path.join(od, name)
        if os.path.isfile(file):
            if name.endswith('.tsv'):
                annot=conn.createFileAnnfromLocalFile(file)
                logging.info(s["orig_command"])
                for arg in s["orig_command"]:
                    if arg == "Project:*" or arg == "Dataset:*":
                        splitted = arg.strip().split(":")
                        conn.getObject(splitted[0], splitted[1]).linkAnnotation(annot)
                    elif arg.is_digit():
                        oim.linkAnnotation(annot)
                    else :
                        logging.warn("wasn't sure how to handle" + file)
            else: 
                np_arr=color.rgb2gray(io.imread(file))
                contours = measure.find_contours(np_arr)
                if len(contours) > 0:
                    contour = get_longest_contour(contours)
                    # Only add 1 Polygon per Mask Shape.
                    # First is usually the longest
                    polygon= __create_polygon(contour, z=0, t=0, comment=name)
                    __create_roi(s, oim, [polygon]) # upload mask
        else:
            logging.warn(file.tostring() + ": should have been a file, but was not")
    return

