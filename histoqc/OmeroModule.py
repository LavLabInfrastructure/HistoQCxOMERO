import logging
from skimage import measure, transform
from omero.model import RoiI, PolygonI
from omero.rtypes import rint, rstring

def rgba_to_int(red, green, blue, alpha=255):
    """ Return the color as an Integer in RGBA encoding """
    return int.from_bytes([red, green, blue, alpha],
                      byteorder='big', signed=True)
    

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


# We have a helper function for creating an ROI and linking it to new shapes
def create_roi(s, shapes):
    oim=s["omero_image_meta"]
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


def create_polygon(contour, x_offset=0, y_offset=0, z=None, t=None, comment=None):
    """ points is 2D list of [[x, y], [x, y]...]"""
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
    polygon.points = rstring(points)
    return polygon


def uploadAsPolygons(s, mask, name):
    logging.info(f"{s['filename']} - \tuploadAsPolygon: {name}")
    ops = s["omero_pixel_store"]
    if mask.shape[0] > 10000 or mask.shape[1] > 10000:
        mask=transform.resize(mask, (int(mask.shape[0]/4),int(mask.shape[1]/4)))
    elif mask.shape[0] > 5000 or mask.shape[1] > 5000:
        mask=transform.resize(mask, (int(mask.shape[0]/2),int(mask.shape[1]/2)))
    contours = measure.find_contours(mask)
    if len(contours) > 0:
        #get disparity
        baseRes=ops.getResolutionDescriptions()[0]
        baseRes=(baseRes.sizeY, baseRes.sizeX)
        currRes=mask.shape

        # contour = get_longest_contour(contours) # Only add 1 Polygon per Mask Shape.

        # shorten sort time by trimming list
        contours2=[]
        for contour in contours:
            if len(contour) > 50: contours2.append(contour)
        del contours

        # sort list then get upload top 3
        contours2.sort(reverse=True, key=(lambda k : len(k)))
        polygons=[]
        for contour in contours2:
            contour=(contour/currRes)*baseRes
            polygon=create_polygon(contour, z=0, t=0, comment=name)
            if polygon:
                polygons.append(polygon)
            if len(polygons) > 3 :
                break

        # upload the polygons
        if len(polygons) > 0 :
            create_roi(s, polygons)
        else :
            logging.warning("unable to generate roi for " + name)
