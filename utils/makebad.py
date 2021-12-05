# David R Thompson
import argparse, sys, os
import numpy as np
import pylab as plt
from glob import glob
from spectral.io import envi
from scipy.stats import norm
from scipy.linalg import solve, inv
from astropy import modeling
from sklearn.linear_model import RANSACRegressor
from skimage.filters import threshold_otsu
import json


def find_header(infile):
  if os.path.exists(infile+'.hdr'):
    return infile+'.hdr'
  elif os.path.exists('.'.join(infile.split('.')[:-1])+'.hdr'):
    return '.'.join(infile.split('.')[:-1])+'.hdr'
  else:
    raise FileNotFoundError('Did not find header file')

# bad row, column ranges (inclusive, zero indexed)
manual_bads=[(1271,(5,6)),
  (1256,193),
  (1231,63),
  ((1168,1170),(40,43)),
  (1164,310),
  (1164,383),
  (1141,(315,316)),
  (1140,312),
  (1139,312),
  (1138,(311,313)),
  (1123,75),
  (1033,113),
  (1005,114),
  (963,281),
  (962,(281,283)),
  (951,270),
  (948,270),
  (945,343),
  (899,105),
  (858,424),
  ((828,844),(436,455)),
  (828,268),
  (794,(362,363)),
  ((769,781),(409,416)),
  (752,158),
  (752,163),
  (746,448),
  (729,311),
  (678,401),
  (625,53),
  (584,416),
  (573,353),
  (569,364),
  (568,364),
  (529,350),
  (490,257),
  (462,327),
  (442,217),
  (434,(201,202)),
  (325,(168,169)), 
  (308,237),
  (307,468),
  (301,440),
  (295,148),
  (238,54),
  (231,100),
  (206,336),
  (202,401),
  (201,461),
  (189,110),
  (184,130),
  (167,247),
  (157,188),
  (119,54),
  (108,355),
  (89,479),
  (81,390),
  (26,273),
  (3,114)]


def main():

    description = "Calculate Flat field"

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('input')
    parser.add_argument('--cue_channel',default=148,type=int)
    parser.add_argument('--ref_lo',default=99,type=int)
    parser.add_argument('--ref_hi',default=1180,type=int)
    parser.add_argument('--hw_lo',default=50,type=int)
    parser.add_argument('--hw_hi',default=180,type=int)
    parser.add_argument('--selection',type=str,default='spatial')
    parser.add_argument('--badmap_out',type=str,default=None)
    parser.add_argument('output')
    args = parser.parse_args()

    infile = envi.open(find_header(args.input))
 
    if int(infile.metadata['data type']) == 2:
        dtype = np.uint16
    elif int(infile.metadata['data type']) == 4:
        dtype = np.float32
    else:
        raise ValueError('Unsupported data type')
    if infile.metadata['interleave'] != 'bil':
        raise ValueError('Unsupported interleave')


    rows = int(infile.metadata['bands'])
    columns = int(infile.metadata['samples'])
    lines = int(infile.metadata['lines'])
    nframe = rows * columns
    margin=2

    flat  = np.zeros((rows,columns))
    count = np.zeros((rows,columns))
    sumsq = np.zeros((rows,columns))
    ref = np.zeros((lines,columns))
    allctrs,alllines = [],[]
    with open(args.input,'rb') as fin:

        for line in range(lines):

            # Read a frame of data
            frame = np.fromfile(fin, count=nframe, dtype=dtype)
            frame = np.array(frame.reshape((rows, columns)),dtype=np.float32)
            ref[line,:] = frame[args.cue_channel, :]

    thresh = np.sort(ref,axis=0)
    thresh = thresh[-10,:]

    with open(args.input,'rb') as fin:

        for line in range(lines):

            # Read a frame of data
            frame = np.fromfile(fin, count=nframe, dtype=dtype)
            frame = np.array(frame.reshape((rows, columns)),dtype=np.float32)
            reference = frame[args.cue_channel, :]
            use = np.where(reference>thresh)[0]

            print(line,len(use),np.median(use),thresh)
            flat[:,use] = flat[:,use] + frame[:,use] 
            count[:,use] = count[:,use] + 1
            sumsq[:,use] = sumsq[:,use] + pow(frame[:,use],2)

        mean_sumsq = sumsq / count
        flat = flat / count

        rowmean = flat[:,30:1250].mean(axis=1)
        rowstdev = flat[:,30:1250].std(axis=1)
        stdev = np.sqrt(mean_sumsq - pow(flat,2))
        stdev[np.logical_not(np.isfinite(stdev))] = 0
        bad = np.logical_or(np.logical_or(stdev==0,
              (abs(flat.T-rowmean)>rowstdev*20).T),stdev>100)
    
    bad[:,:25] = 0
    bad[:,1265:] = 0

    for bad_cols, bad_rows in manual_bads:
        if type(bad_rows)==int:
            rows_range = [bad_rows]
        else:
            rows_range = range(bad_rows[0],bad_rows[1]+1)
        if type(bad_cols)==int:
            cols_range = [bad_cols]
        else:
            cols_range = range(bad_cols[0],bad_cols[1]+1)
        for col in cols_range:
            for row in rows_range:
                bad[row,col] = 1
   #plt.hist(stdev.flatten(),500)
   #plt.figure()
   #plt.imshow(bad)
   #plt.show()
    bads = 0
    bad_map = bad.copy()
    bad_map = np.array(bad_map,dtype=np.int16)
    for column in range(bad_map.shape[1]):
        state_machine = 0
        for row in range(bad_map.shape[0]):
            if bad[row,column]:
                state_machine = state_machine + 1
                bad_map[row,column] = -state_machine
                print(row,column,state_machine)
                bads = bads + 1
            else:
                state_machine = 0
    print('total bads:',bads)
    bad_map = bad_map.reshape((rows,columns,1))
    envi.save_image(args.output+'.hdr',
        bad_map, interleave='bsq', ext='', force=True)

if __name__ == '__main__':

    main()
