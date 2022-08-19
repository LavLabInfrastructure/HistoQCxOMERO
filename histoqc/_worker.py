"""histoqc worker functions"""
import asyncio
from asyncio.log import logger
import inspect
import os
import shutil
from typing import Coroutine

from histoqc.BaseImage import BaseImage
from histoqc._pipeline import load_pipeline
from histoqc._pipeline import setup_plotting_backend
# --- worker functions --------------------------------------------------------

def worker_setup(c):
    """needed for multiprocessing worker setup"""
    setup_plotting_backend()
    load_pipeline(config=c)


def worker(idx, id, server, *,
           process_queue, config, outdir, log_manager, lock, shared_dict, num_imgs, force, command):
    """pipeline worker function"""

    # --- output directory preparation --------------------------------
    fname_outdir = os.path.join(outdir, str(id))
    if os.path.isdir(fname_outdir):  # directory exists
        if not force:
            log_manager.logger.warning(
                f"{id} already seems to be processed ({fname_outdir} exists),"
                " skipping. To avoid this behavior use --force"
            )
            return
        else:
            # remove entire directory to ensure no old files are present
            shutil.rmtree(fname_outdir)
    # create output dir
    os.makedirs(fname_outdir)
    
    log_manager.logger.info(f"----Working on:\t{id}\t\t{idx+1} of {num_imgs}")

    try:
        import time; start_time = time.time()
        s = BaseImage(command, server, id, fname_outdir, dict(config.items("BaseImage.BaseImage")))
        for process, process_params in process_queue:
            process_params["lock"] = lock
            process_params["shared_dict"] = shared_dict

            if inspect.iscoroutinefunction(process):
                asyncio.run(process(s,process_params))
            else:
                process(s, process_params)

            s["completed"].append(process.__name__)
        log_manager.logger.info("--- %s seconds ---" % (time.time() - start_time))

    except Exception as exc:
        # reproduce histoqc error string
        _oneline_doc_str = exc.__doc__.replace('\n', '')
        err_str = f"{exc.__class__} {_oneline_doc_str} {exc}"

        log_manager.logger.error(
            f"{id} - Error analyzing file (skipping): \t {err_str}"
        )
        if exc.__traceback__.tb_next is not None:
            func_tb_obj = str(exc.__traceback__.tb_next.tb_frame.f_code)
        else:
            func_tb_obj = str(exc.__traceback__)

        exc.__histoqc_err__ = (id, err_str, func_tb_obj)
        raise exc

    else:
        # TODO:
        #   the histoqc workaround below is due an implementation detail in BaseImage:
        #   BaseImage keeps an OpenSlide instance stored under os_handle and leaks a
        #   file handle. This will need fixing in BaseImage.
        #   -> best solution would be to make BaseImage a contextmanager and close
        #      and cleanup the OpenSlide handle on __exit__
        s["omero_image_meta"] = None
        s["omero_pixel_store"] = None  # need to get rid of handles because it can't be pickled
        return s


def worker_success(s, result_file):
    """success callback"""
    if s is None:
        return

    with result_file:
        if result_file.is_empty_file():
            result_file.write_headers(s)
        _fields = '\t'.join([str(s[field]) for field in s['output']])
        _warnings = '|'.join(s['warnings'])
        result_file.write_line("\t".join([_fields, _warnings]))


def worker_error(e, failed):
    """error callback"""
    logger.info(failed)
    if hasattr(e, '__histoqc_err__'):
        id, err_str, tb = e.__histoqc_err__
    else:
        # error outside of pipeline
        # todo: it would be better to handle all of this as a decorator
        #   around the worker function
        id, err_str, tb = "N/A", f"error outside of pipeline {e!r}", None
    failed.append((id, err_str, tb))
