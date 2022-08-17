import logging
import numpy as np
from skimage.transform import resize
import zlib, dill
from distutils.util import strtobool
from omero.gateway import BlitzGateway
from omero import InternalException

def printMaskHelper(type, prev_mask, curr_mask):
    if type == "relative2mask":
        if len(prev_mask.nonzero()[0]) == 0:
            return str(-100)
        else:
            return str(1 - len(curr_mask.nonzero()[0]) / len(prev_mask.nonzero()[0]))
    elif type == "relative2image":
        return str(len(curr_mask.nonzero()[0]) / np.prod(curr_mask.shape))
    elif type == "absolute":
        return str(len(curr_mask.nonzero()[0]))
    else:
        return str(-1)


# this function is seperated out because in the future we hope to have automatic detection of
# magnification if not present in open slide, and/or to confirm openslide base magnification
def getMag(s, params):
    logging.info(f"{s['filename']} - \tgetMag")
    oim = s["omero_image_meta"]
    if oim.getObjectiveSettings() != None :
        mag = oim.getObjectiveSettings().getObjective().getNominalMagnification()
    else :
        res = oim.getPixelSizeX()
        if (res <= .3 ):
            mag = 40
        elif (res <= .6):
            mag = 20
        else:
            mag = 10
        logging.warning(f"{s['filename']} - Unknown base magnification for file")
        s["warnings"].append(f"{s['filename']} - Unknown base magnification for file")
    mag = float(mag)
    return mag


# turns a sync iterable into async generator
async def desync(it):
  for x in it: yield x  


class BaseImage(dict):

    def __init__(self, command, server, id, fname_outdir, params):
        dict.__init__(self)

        self.in_memory_compression = strtobool(params.get("in_memory_compression", "False"))

        self["warnings"] = ['']  # this needs to be first key in case anything else wants to add to it
        self["output"] = []
        self["outdir"] = fname_outdir
        self["orig_command"] = command
        
        # create main handle
        conn = BlitzGateway(server[0],server[1],host=server[2],port=server[3],secure=server[4]) 
        conn.connect()
        conn.c.enableKeepAlive(30)

        # get omero service handles
        self["omero_conn_handle"] = conn
        self["omero_image_meta"] = conn.getObject("image",id)
        self["omero_pixel_store"] = conn.createRawPixelsStore()

        # set ids
        self["omero_pixel_store"].setPixelsId(self["omero_image_meta"].getPixelsId(), False)

        # these 2 need to be first for UI to work
        self.addToPrintList("filename", self["omero_image_meta"].getName())
        self.addToPrintList("comments", " ")

        self["image_base_size"] = (self["omero_image_meta"].getSizeX(), self["omero_image_meta"].getSizeY())
        self["image_work_size"] = params.get("image_work_size", "1.25x")
        self["image_channel_count"] = self["omero_image_meta"].getSizeC()
        self["mask_statistics"] = params.get("mask_statistics", "relative2mask")
        self["base_mag"] = getMag(self, params)
        self.addToPrintList("base_mag", self["base_mag"])

        mask_statistics_types = ["relative2mask", "absolute", "relative2image"]
        if (self["mask_statistics"] not in mask_statistics_types):
            logging.error(
                f"mask_statistic type '{self['mask_statistics']}' is not one of the 3 supported options relative2mask, absolute, relative2image!")
            exit()
        dim=self.parseDim(self["image_work_size"])
        self["img_mask_use"] = np.ones((dim[1],dim[0]), dtype=bool)
        self["img_mask_force"] = []

        self["completed"] = []    

    def __getitem__(self, key):
        value = super(BaseImage, self).__getitem__(key)
        if hasattr(self,"in_memory_compression") and  self.in_memory_compression and key.startswith("img"):
            value = dill.loads(zlib.decompress(value))
        return value

    def __setitem__(self, key, value):
        if hasattr(self,"in_memory_compression") and self.in_memory_compression and key.startswith("img"):
            value = zlib.compress(dill.dumps(value), level=5)
        return super(BaseImage, self).__setitem__(key,value)

    def addToPrintList(self, name, val):
        self[name] = val
        self["output"].append(name)


    # many ways to ask for dimensions (scale factor, res level, explicit size, and desired mag)
    def parseDim(self, dim):
        ops = self["omero_pixel_store"]
        if dim.replace(".", "0", 1).isdigit(): #check to see if dim is a number
            dim = float(dim)
            if dim < 1 and not dim.is_integer():  # specifying a downscale factor from base
                dim = np.asarray(self["image_base_size"]) * dim
            elif dim < 100:  # assume it is a level in the img pyramid instead of a direct request
                lvl = int(dim)
                resolutionCount = ops.getResolutionLevels()
                if lvl >= resolutionCount:
                    lvl = resolutionCount - 1
                    logging.error(
                        f"{self['filename']}: Desired Image Level {dim} does not exist! Instead using level {lvl}! Downstream output may not be correct")
                    self["warnings"].append(
                        f"Desired Image Level {dim} does not exist! Instead using level {lvl}! Downstream output may not be correct")
                ops.setResolutionLevel(lvl)
                desc=ops.getResolutionDescriptions()
                dim = (desc[lvl].sizeX,desc[lvl].sizeY)
                logging.info(
                    f"{self['filename']} - \t\tloading image from level {lvl} of size {dim[0]}x{dim[1]}")
            else:  # assume its an explicit size, *WARNING* this will likely cause different images to have different perceived magnifications!
                logging.info(f"{self['filename']} - \t\tcreating image thumb of size {str(dim)}")
                dim=(dim,dim)
        elif "X" in dim.upper():  # specifies a desired operating magnification
            base_mag = self["base_mag"]
            if base_mag != "NA":  # if base magnification is not known, it is set to NA by basic module
                base_mag = float(base_mag)
            else:  # without knowing base mag, can't use this scaling, push error and exit
                logging.error(
                    f"{self['filename']}: Has unknown or uncalculated base magnification, cannot specify magnification scale: {base_mag}! Did you try getMag?")
                return -1
            target_mag = float(dim.upper().split("X")[0])
            downfactor = target_mag / base_mag
            dim = np.asarray(self["image_base_size"]) * downfactor
        else:
            logging.error(
                f"{self['filename']}: Unknown image level setting: {dim}!")
            return -1
        dim = (int(dim[0]),int(dim[1]))
        self.setClosestRes(dim)
        return dim
        

    # fetches each channel of a given tile (x,y,w,h)
    def getFullTile(self, tile) :
        ops = self["omero_pixel_store"]
        channelCount = self["image_channel_count"]
        arr = np.zeros([tile[3], tile[2], channelCount], dtype=np.uint8)
        for c in range(channelCount) :
            tmp = np.frombuffer(ops.getTile(0,c,0,tile[0],tile[1],tile[2],tile[3]), dtype=np.uint8)
            tmp.shape=(tile[3],tile[2])
            arr[..., c] = tmp
        return arr


    # gets whole image outright as opposed to an interable
    def getFullImg(self) :
        ops = self["omero_pixel_store"]
        channelCount = self["image_channel_count"]
        dim = ops.getResolutionDescriptions()[(ops.getResolutionLevels()-1)-ops.getResolutionLevel()]
        arr = np.zeros([dim.sizeY,dim.sizeX,channelCount], dtype=np.uint8)
        for c in range(channelCount) :
            tmp=np.frombuffer(ops.getPlane(0,c,0), dtype=np.uint8)
            tmp.shape=(dim.sizeY, dim.sizeX)
            arr[...,c] = tmp     
        return arr


    # sets the pixel store to closest resolution level without going under the desired res
    def setClosestRes(self, dim) :
        ops = self["omero_pixel_store"]
        # for each resolution of this image
        resolutions=ops.getResolutionDescriptions()
        for i in range(ops.getResolutionLevels()) :
            res=resolutions[i]
            currDif=(res.sizeX-dim[0],res.sizeY-dim[1])
            # if the prev res was the closest without going under, use it
            if currDif[0] < 0 or currDif[1] < 0:
                # we need to add one for i (prev res was correct) and remove one from getResLevels(1 to 0 index), so nice
                ops.setResolutionLevel(ops.getResolutionLevels()-i)
                self["image_tile_size"] = self["omero_pixel_store"].getTileSize()
                return


    # async tile generator 
    async def tileGenerator(self, dim):
        self.setClosestRes(dim)
        tileSize = self["image_tile_size"]
        async for idxY in desync(range(0,dim[1],tileSize[1])):
            tileSize = self["image_tile_size"]
            if dim[1]-idxY < tileSize[1]:
                tileSize=(tileSize[0],dim[1]-idxY)
            async for idxX in desync(range(0,dim[0],tileSize[0])):
                try:
                    if dim[0]-idxX < tileSize[0]:
                        tileSize=(dim[0]-idxX, tileSize[1])
                    tile=(idxX, idxY, tileSize[0], tileSize[1])
                    img = self.getFullTile(tile), tile
                    #io.imsave(self["outdir"]+ os.sep +self["filename"]+"_"+str(idxX)+"_"+str(idxY)+".png", img[0]) 
                except InternalException as e:
                    logging.info(e)
                yield img


    # fetches a thumbnail based on provided dimension(s)
    def getImgThumb(self, dim):
        dim = self.parseDim(dim) # convert to x,y tuple
        self.setClosestRes(dim) # set res level to closest without going under
        arr = self.getFullImg() # fetch the image at the closest res
        # if the closest res wasn't right, resize
        if dim[1] != len(arr) or dim[0] != len(arr[0]) : 
            arr = resize(arr, (dim[1],dim[0]))
        return arr, (0,0,dim[0],dim[1]) # tuple is starting pos (compatibility for tiled calcs)