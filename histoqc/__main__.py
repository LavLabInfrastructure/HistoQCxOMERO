import argparse
import configparser
import datetime
import logging
import multiprocessing
import os
from posixpath import split
import sys
import tempfile
from functools import partial

from omero.gateway import BlitzGateway
from omero.sys import Parameters

from histoqc._pipeline import BatchedResultFile
from histoqc._pipeline import MultiProcessingLogManager
from histoqc._pipeline import load_pipeline
from histoqc._pipeline import log_pipeline
from histoqc._pipeline import move_logging_file_handler
from histoqc._pipeline import setup_logging
from histoqc._pipeline import setup_plotting_backend
from histoqc._worker import worker
from histoqc._worker import worker_setup
from histoqc._worker import worker_success
from histoqc._worker import worker_error
from histoqc.config import read_config_template
from histoqc.data import managed_pkg_data


@managed_pkg_data
def main(argv=None, client=None):
    """main entry point for histoqc pipelines"""
    if argv is None:
        argv = sys.argv[1:]
            

    parser = argparse.ArgumentParser(prog="histoqc", description='Run HistoQC main quality control pipeline for digital pathology images')
    parser.add_argument('object_id',
                        help="OMERO object id(s)"
                             "(You can use * for all images, or Project:00/Dataset:00 to specify image groups)",
                        nargs="+", type=str)
    parser.add_argument('-s', '--server', 
                        help="Skips being prompted about which omero server to sign in on",
                        default="",
                        type=str)
    parser.add_argument('-p', '--port',
                        help="Skips being prompted about which port to sign in on",
                        default="4063",
                        type=str)
    parser.add_argument('--secure',
                        help="We will try to guess ssl but if you're on an ssl port use this",
                        default=None,
                        type=bool)
    parser.add_argument('-u', '--user',
                        help="Skips being prompted on which user to log in as",
                        default="",
                        type=str)
    parser.add_argument('-g', '--group',
                        help="Avoids being prompted for a group",
                        default="",
                        type=str)
    parser.add_argument('-w', '--password',
                        help="Skips being prompted for the password of the user",
                        default="",
                        type=str)
    parser.add_argument('-o', '--outdir',
                        help="outputdir, defining this will prevent uploading back to the server",
                        default="",
                        type=str)
    parser.add_argument('-P', '--basepath',
                        help="base path to add to file names,"
                             " helps when producing data using existing output file as input",
                        default="",
                        type=str)
    parser.add_argument('-c', '--config',
                        help="config file to use, either by name supplied by histoqc.config (e.g., v2.1) or filename",
                        type=str)
    parser.add_argument('-f', '--force',
                        help="force overwriting of existing files",
                        action="store_true")
    parser.add_argument('-b', '--batch',
                        help="break results file into subsets of this size",
                        type=int,
                        default=None)
    parser.add_argument('-n', '--nprocesses',
                        help="number of processes to launch",
                        type=int,
                        default=1)
    parser.add_argument('--symlink', metavar="TARGET_DIR",
                        help="create symlink to outdir in TARGET_DIR",
                        default=None)
    args = parser.parse_args(argv)

    # --- multiprocessing and logging setup -----------------------------------

    setup_logging(capture_warnings=True, filter_warnings='ignore')
    mpm = multiprocessing.Manager()
    lm = MultiProcessingLogManager('histoqc', manager=mpm)

    # --- parse the pipeline configuration ------------------------------------

    config = configparser.ConfigParser()
    if not args.config:
        lm.logger.warning(f"Configuration file not set (--config), using default")
        config.read_string(read_config_template('default'))
    elif os.path.exists(args.config):
        config.read(args.config) #Will read the config file
    else:
        lm.logger.warning(f"Configuration file {args.config} assuming to be a template...checking.")
        config.read_string(read_config_template(args.config))

    # --- provide models, pen and templates as fallbacks from package data ----

    managed_pkg_data.inject_pkg_data_fallback(config)

    # --- load the process queue (error early) --------------------------------

    _steps = log_pipeline(config, log_manager=lm)
    process_queue = load_pipeline(config)

    # --- check symlink target ------------------------------------------------

    if args.symlink is not None:
        if not os.path.isdir(args.symlink):
            lm.logger.error("error: --symlink {args.symlink} is not a directory")
            return -1

    # --- create output directory and move log --------------------------------
    if args.outdir : 
        args.outdir = os.path.expanduser(args.outdir)
        os.makedirs(args.outdir, exist_ok=True)
    else :
        args.outdir = tempfile.mkdtemp()
    move_logging_file_handler(logging.getLogger(), args.outdir)


    if BatchedResultFile.results_in_path(args.outdir):
        if args.force:
            lm.logger.info("Previous run detected....overwriting (--force set)")
        else:
            lm.logger.info("Previous run detected....skipping completed (--force not set)")

    results = BatchedResultFile(args.outdir,
                                manager=mpm,
                                batch_size=args.batch,
                                force_overwrite=args.force)

    # --- document configuration in results -----------------------------------

    results.add_header(f"start_time:\t{datetime.datetime.now()}")
    results.add_header(f"pipeline: {' '.join(_steps)}")
    results.add_header(f"outdir:\t{os.path.realpath(args.outdir)}")
    results.add_header(f"config_file:\t{os.path.realpath(args.config) if args.config is not None else 'default'}")
    results.add_header(f"command_line_args:\t{' '.join(argv)}")

    # --- log in to omero, prompt for info if needed -------------------------
    if client == None:
        try:
            conn = BlitzGateway(args.user, args.password, host=args.server, port=args.port, secure=args.secure)
            conn.connect()
        except Exception as e:
            logging.error("Failed to connect")
            logging.error(e)
            exit(1)
    else:
        conn = BlitzGateway(client_obj=client)
    conn.c.enableKeepAlive(60)

    # --- parse and check omero object ids (there are 2 options) ------------------------
    args.object_id=''.join(args.object_id) # list to string
    # if tsv, grab ome objects
    if type(args.object_id) is str and args.object_id.endswith('.tsv') :
        ids = []
        with open(args.object_id, 'rt') as f :
            for line in f :
                if line[0] == "#": continue 
                obj = line.strip().split("\t")
                if len(obj) > 1:
                    ids.append(f"{obj[0]}:{obj[1]} ")
                else :
                    ids.append(f"{obj[0]} ")
        args.object_id = ids
    else:
        args.object_id = args.object_id.split(' ')
        length=len(args.object_id)
        if type(args.object_id) is list and length > 1: 
            args.object_id.pop(length-1)

    # parse ome objs to image ids  
    service=conn.getContainerService()
    ids = []
    for objId in args.object_id: #for obj in input, find all related images
        splitObj=objId.split(':')
        if splitObj[0].lower() == "image": objId=splitObj[1] # image:(imgId) is unneeded
        if objId.isdigit(): #is img id
            ids.append(int(objId))
        else : # else 
            for obj in splitObj[1].split(','):
                if obj.isdigit():
                    api_return = service.getImages(splitObj[0],[int(obj)],Parameters())
                    for val in api_return :
                        ids.append(val._id._val)
    service.close()
    lm.logger.info("-" * 80)
    num_imgs = len(ids)
    lm.logger.info(f"Number of files detected by pattern:\t{num_imgs}")

    # --- start worker processes ----------------------------------------------

    _shared_state = {
        'process_queue': process_queue,
        'config': config,
        'outdir': args.outdir,
        'log_manager': lm,
        'lock': mpm.Lock(),
        'shared_dict': mpm.dict(),
        'num_imgs': num_imgs,
        'force': args.force,
        'command': args.object_id,
    }
    failed = mpm.list()
    setup_plotting_backend(lm.logger)
    try:
        if args.nprocesses > 1:

            with lm.logger_thread():
                with multiprocessing.Pool(processes=args.nprocesses,
                                          initializer=worker_setup,
                                          initargs=(config,)) as pool:
                    try:
                        for idx, id in enumerate(ids):
                            _ = pool.apply_async(
                                func=worker,
                                args=(idx, id, conn),
                                kwds=_shared_state,
                                callback=partial(worker_success, result_file=results),
                                error_callback=partial(worker_error, failed=failed),
                            )

                    finally:
                        pool.close()
                        pool.join()

        else:
            for idx, id in enumerate(ids):
                try:
                    _success = worker(idx, id, conn, **_shared_state)
                except Exception as exc:
                    worker_error(exc, failed)
                    continue
                else:
                    worker_success(_success, results)

    except KeyboardInterrupt:
        lm.logger.info("-----REQUESTED-ABORT-----\n")

    else:
        lm.logger.info("----------Done-----------")

    finally:
        lm.logger.info(f"There are {len(failed)} explicitly failed images (available also in error.log),"
                       " warnings are listed in warnings column in output")
        for id, error, tb in failed:
            lm.logger.info(f"{id}\t{error}\n{tb}")

    if args.symlink is not None:
        origin = os.path.realpath(args.outdir)
        target = os.path.join(
            os.path.realpath(args.symlink),
            os.path.basename(origin)
        )
        try:
            os.symlink(origin, target, target_is_directory=True)
            lm.logger.info("Symlink to output directory created")
        except (FileExistsError, FileNotFoundError):
            lm.logger.error(
                f"Error creating symlink to output in '{args.symlink}', "
                f"Please create manually: ln -s {origin} {target}"
            )
    return 0

if __name__ == "__main__":
    sys.exit(main())
