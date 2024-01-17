#! /usr/bin/env python
#
#  Copyright 2020 California Institute of Technology
#
# EMIT Radiometric Calibration code
# Author: David R Thompson, david.r.thompson@jpl.nasa.gov

import scipy.linalg
import os, sys, os.path
import numpy as np
from spectral.io import envi
from datetime import datetime, timezone
from numpy import linalg, polyfit, polyval
import json
import logging
import argparse
import multiprocessing
import ray
import pylab as plt

# Import some PRM-specific functions
my_directory, my_executable = os.path.split(os.path.abspath(__file__))

#my_directory = '/Users/achlus/data1/repos/airborne_sds/prm_l1b_radiance'
sys.path.append(my_directory + '/utils/')
from prm_read import read_frames, read_frames_metadata
from prm_pedestal import fix_pedestal
from prm_panel_ghost import panel_ghost_corr_prism

from fpa import FPA, frame_embed, frame_extract
from fixbad import fix_bad
from fixosf import fix_osf
from fixlinearity import fix_linearity
from fixscatter import fix_scatter
from fixghost import fix_ghost
from fixelectronicghost import fix_electronic_ghost
from fixghostraster import build_ghost_matrix
from fixghostraster import build_ghost_blur
from darksubtract import subtract_dark
from leftshift import left_shift_twice

BAD_FLAG = -9000

header_template = """ENVI
description = {{PRISM calibrated spectral radiance (units: uW nm-1 cm-2 sr-1)}}
samples = {ncolumns}
lines = {lines}
bands = {nchannels}
header offset = 0
file type = ENVI Standard
data type = 4
interleave = bil
byte order = 0
wavelength units = Nanometers
wavelength = {{{wavelength_string}}}
fwhm = {{{fwhm_string}}}
band names = {{{band_names_string}}}
masked pixel noise = {masked_pixel_noise}
"""


def find_header(infile):
    if os.path.exists(infile+'.hdr'):
        return infile+'.hdr'
    elif os.path.exists('.'.join(infile.split('.')[:-1])+'.hdr'):
        return '.'.join(infile.split('.')[:-1])+'.hdr'
    else:
        raise FileNotFoundError('Did not find header file')


class Config:

    def __init__(self, fpa, mode):

        # Load calibration file data
        current_mode   = fpa.modes[mode]

        # Move this outside, to the main function
        if hasattr(fpa,'left_shift_twice') and fpa.left_shift_twice:
            # left shift, returning to the 16 bit range.
            self.dark = left_shift_twice(self.dark)

        if hasattr(fpa,'spectral_calibration_file'):
            _, self.wl_full, self.fwhm_full = \
                 np.loadtxt(fpa.spectral_calibration_file).T * 1000
        else:
            self.wl_full, self.fwhm_full = None

        if 'flat_field_file' in current_mode.keys():
            self.flat_field_file = current_mode['flat_field_file']
            self.flat_field = np.fromfile(self.flat_field_file,
                 dtype = np.float32).reshape((2, fpa.native_rows, fpa.native_columns))
            self.flat_field = self.flat_field[0,:,:]
            self.flat_field[np.logical_not(np.isfinite(self.flat_field))] = 0
        else:
            self.flat_field = None

        if 'radiometric_coefficient_file' in current_mode.keys():
            self.radiometric_coefficient_file = current_mode['radiometric_coefficient_file']
            self.radiometric_calibration, self.radiometric_uncert,_ = \
                 np.loadtxt(self.radiometric_coefficient_file).T
        else:
            self.radiometric_calibration, self.radiometric_uncert = None, None

        # Load ghost configuration and construct the matrix
        if hasattr(fpa,'panel_ghost_file'):
            with open(fpa.panel_ghost_file,'r') as fin:
                self.panel_ghost = json.load(fin)
        else:
            self.panel_ghost = None

@ray.remote
def calibrate_raw_remote(frames, fpa, config):
    return calibrate_raw(frames, fpa, config)

def calibrate_raw(frames, fpa, config):

    if len(frames.shape) == 2:
      frames = np.reshape(frames,(1,frames.shape[0],frames.shape[1]))

    noises = []
    output_frames = []
    for _f in range(frames.shape[0]):
        frame = frames[_f,...]
        noise = -9999

        ## Don't calibrate a bad frame
        if not np.all(frame <= BAD_FLAG):

            # Left shift, returning to the 16 bit range.
            if hasattr(fpa,'left_shift_twice') and fpa.left_shift_twice:
               frame = left_shift_twice(frame)

            # Dark state subtraction
            frame = subtract_dark(frame, config.dark)

            ## Delete telemetry
            if hasattr(fpa,'ignore_first_row') and fpa.ignore_first_row:
               frame[0,:] = frame[1,:]

            # Raw noise calculation
            if hasattr(fpa,'masked_columns'):
                noise = np.nanmedian(np.std(frame[:,fpa.masked_columns],axis=0))
            elif hasattr(fpa,'masked_rows'):
                noise = np.nanmedian(np.std(frame[fpa.masked_rows,:],axis=1))
            else:
                noise = -1

            #Pedestal shift
            frame = fix_pedestal(frame, fpa)

            # Electronic ghost
            if config.panel_ghost is not None:
                frame = panel_ghost_corr_prism(frame[:,np.newaxis,:].T, config.panel_ghost)[:,0,:].T

            frame = frame * config.flat_field

            # Fix bad pixels, and any nonfinite results from the previous
            # operations
            flagged = np.logical_not(np.isfinite(frame))
            frame[flagged] = 0
            # frame = fix_bad(frame, bad, fpa)

            # Absolute radiometry
            if config.radiometric_calibration is not None:
                # Account for data channel, avoid the two SWIR channels
                frame[1:] = (frame[1:].T * config.radiometric_calibration[:-2]).T

            # Catch NaNs
            frame[np.logical_not(np.isfinite(frame))]=0

        # Clip the channels to the appropriate size, if needed
        if fpa.extract_subframe:
            frame = frame[:,fpa.first_distributed_column:(fpa.last_distributed_column + 1)]
            frame = frame[fpa.first_distributed_row:(fpa.last_distributed_row + 1),:]
        output_frames.append(frame)
        noises.append(noise)

    # Replace all bad data flags with -9999
    output_frames = np.stack(output_frames)
    output_frames[output_frames<=(BAD_FLAG+1e-6)] = np.nan

    noises = np.array(noises)
    if np.sum(noises != -9999) > 0:
        noises = np.nanmedian(noises[noises != -9999])
    else:
        noises = -9999

    # Co-add
    output_frames = np.nanmean(output_frames,axis=0)
    output_frames[np.isnan(output_frames)] = -9999

    return output_frames, noises


def main():

    description = "Spectroradiometric Calibration"

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('input_file', default='')
    parser.add_argument('config_file', default='')
    parser.add_argument('output_file', default='')
    parser.add_argument('--mode', default = 'default')
    parser.add_argument('--level', default='DEBUG',
            help='verbosity level: INFO, ERROR, or DEBUG')
    parser.add_argument('--log_file', type=str, default=None)
    parser.add_argument('--max_jobs', type=int, default=40)
    parser.add_argument('--debug_mode', action='store_true')
    parser.add_argument('--binfac', type=str, default=None)

    # sys.argv = [
    # "script_name",
    # "/Users/achlus/data1/prm/raw/prm20230411t192437_raw",
    # "/Users/achlus/data1/repos/airborne_sds/prm_l1b_radiance/config/PRISM_radiance_test_config.json",
    # "/Users/achlus/data1/prm/rdn/prm20230411t192437_RDN",
    # "--mode","default",  # mode argument
    # "--level", "DEBUG",  # level argument
    # "--max_jobs", "10",  # max_jobs argument
    # "--debug_mode",  # debug_mode flag
    # "--binfac", "/Users/achlus/data1/prm/ort/prm20230416t213508_L1B_ORT_main_b78fd43e.binfac"  # binfac argument
    # ]

    args = parser.parse_args()

    fpa = FPA(args.config_file)
    config = Config(fpa, args.mode)

    #Find binfac file if not provided
    if args.binfac is None:
        args.binfac = args.input_file + '.binfac'
        if os.path.isfile(args.binfac) is False:
            logging.error(f'binfac file not found at expected location: {args.binfac}')
            raise ValueError('Binfac file not found - see log for details')

    try:
        binfac = int(args.binfac)
    except:
        binfac = int(np.genfromtxt(args.binfac))

    # Set up logging
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    if args.log_file is None:
        logging.basicConfig(format='%(message)s', level=args.level)
    else:
        logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s',
            level=args.level, filename=args.log_file)

    logging.info('Starting calibration')
    raw = 'Start'
    debug = args.debug_mode

    infile = envi.open(find_header(args.input_file))

    if int(infile.metadata['data type']) == 2:
        dtype = np.int16
    elif int(infile.metadata['data type']) == 12:
        dtype = np.uint16
    elif int(infile.metadata['data type']) == 4:
        dtype = np.float32
    else:
        raise ValueError('Unsupported data type')
    if infile.metadata['interleave'] != 'bil':
        raise ValueError('Unsupported interleave')

    rows = int(infile.metadata['bands']) - 1 # extra band is metadata
    columns = int(infile.metadata['samples'])
    lines_analyzed = 0
    nframe = fpa.native_rows * fpa.native_columns *binfac
    noises = []

    # Read metadata from RAW ang file
    logging.debug('Reading metadata')
    frame_meta, num_read, frame_obcv = read_frames_metadata(args.input_file, 500000, fpa.native_rows, fpa.native_columns, 0)

    dark_frame_idxs = np.where(frame_obcv == 2)[0]
    science_frame_idxs = np.where(frame_obcv[dark_frame_idxs[-1]+1:])[0] + dark_frame_idxs[-1] + 1

    #Apply shutter offset
    science_frame_idxs = science_frame_idxs[(fpa.shutter_offset):]

    binned_lines=  len(science_frame_idxs)//binfac

    logging.debug('Found {len(dark_frame_idxs)} dark frames and {len(science_frame_idxs)} science frames')

    if np.all(science_frame_idxs - science_frame_idxs[0] == np.arange(len(science_frame_idxs))) is False:
        logging.error('Science frames are not contiguous, cannot proceed')
        raise AttributeError('Science frames are not contiguous')

    # Read dark
    dark_frames, _, _, _ = read_frames(args.input_file, fpa.num_dark_frames_use, fpa.native_rows, fpa.native_columns, dark_frame_idxs[10])
    config.dark = np.mean(dark_frames,axis=0)
    config.dark_std = np.std(dark_frames,axis=0)
    del dark_frames
    logging.debug('Dark read complete, beginning calibration')
    ray.init(runtime_env={"working_dir":my_directory + '/utils/'})

    fpa_id = ray.put(fpa)
    current_frame = science_frame_idxs[0]
    lines_processed = 0
    noises = []

    with open(args.input_file,'rb') as fin:
        fin.seek(science_frame_idxs[0]*fpa.native_columns* fpa.native_rows*2)

        with open(args.output_file,'wb') as fout:
            raw = np.fromfile(fin, count=nframe, dtype=dtype)
            jobs = []

            while lines_processed < binned_lines:
                current_frame+=binfac
                # Read frames of data
                raw = np.array(raw, dtype=np.float32)
                frames = raw.reshape((binfac,fpa.native_rows,fpa.native_columns))

                if lines_processed%10==0:
                    logging.info(f'Calibrating lines {current_frame} - {current_frame+binfac}')

                jobs.append(calibrate_raw_remote.remote(frames, fpa_id, config))
                lines_processed += 1

                if len(jobs) == args.max_jobs:
                    # Write to file
                    result = ray.get(jobs)
                    for frame, noise in result:
                        np.asarray(frame, dtype=np.float32).tofile(fout)
                        noises.append(noise)
                    jobs = []

                # Read next chunk
                raw = np.fromfile(fin, count=nframe, dtype=dtype)

            # Do any final jobs
            result = ray.get(jobs)
            for frame, noise in result:
                np.asarray(frame, dtype=np.float32).tofile(fout)
                noises.append(noise)

    # Form output metadata strings
    wl = config.wl_full.copy()
    fwhm = config.fwhm_full.copy()

    if fpa.extract_subframe:
        ncolumns = fpa.last_distributed_column - fpa.first_distributed_column + 1
        nchannels = fpa.last_distributed_row - fpa.first_distributed_row + 1
        wl = wl[fpa.first_distributed_row:fpa.last_distributed_row+1]
        fwhm = fwhm[fpa.first_distributed_row:fpa.last_distributed_row+1]
    else:
        nchannels, ncolumns = fpa.native_rows, fpa.native_columns

    band_names_string = ','.join(['channel_'+str(i) \
       for i in range(len(wl))])
    fwhm_string =  ','.join([str(w) for w in fwhm])
    wavelength_string = ','.join([str(w) for w in wl])

    params = {}
    params['masked_pixel_noise'] = np.nanmedian(np.array(noises))
    params['run_command_string'] = ' '.join(sys.argv)
    params['input_files_string'] = ''
    for var in dir(fpa):
       if var.endswith('_file'):
          params['input_files_string'] = params['input_files_string'] + \
             ' %s=%s'%(var,getattr(fpa,var))
    params['lines'] =  binned_lines

    params.update(**locals())
    with open(args.output_file+'.hdr','w') as fout:
        fout.write(header_template.format(**params))

    logging.info('Done')


if __name__ == '__main__':

    main()
