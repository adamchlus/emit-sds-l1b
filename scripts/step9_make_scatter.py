# David R Thompson

import sys, os
import numpy as np
from glob import glob
import pylab as plt
from spectral.io import envi

# Set this to true if we have already completed a full calibration solution
# We can then apply the EMIT calibration process for testing our SRF/CRF 
# correction matrices
validate = True

if True:

    # First combine the data from all the point spread function measurements
    # This requires running basic electronic corrections on all datasets first
    # Record the resulting Gaussian fits in some text files

    files = glob('/beegfs/scratch/drt/20211114_EMIT_Infield/20211114_InFieldScatter/*linear')
    files.sort()
    cmds = ['python','/home/drt/src/emit-sds-l1b/utils/makescatter.py','--target_row','40','--spatial']+files+['>>','spatial_params.txt']
    os.system(' '.join(cmds))


    files = glob('/beegfs/scratch/drt/20211114_EMIT_Infield/20211115_InFieldScatter/*linear')
    files.sort()
    cmds = ['python','/home/drt/src/emit-sds-l1b/utils/makescatter.py','--target_row','940']+files+['>>','spectral_params.txt']
    os.system(' '.join(cmds))

if True:

  curves = []

  # Now build scatter correction matrices.  Do it twice: first with a null (no-op)
  # correction for comparison, and second for real.
  for magnitude in [0,1]: 

    cmd = 'python ../utils/combinescatter.py --manual '+str(magnitude)+' --spatial  ../scripts/spatial_params.txt ../data/EMIT_SpatialScatter_20211226' 
    os.system(cmd)

    cmd = 'python ../utils/combinescatter.py --manual '+str(magnitude)+' ../scripts/spectral_params.txt ../data/EMIT_SpectralScatter_20211226' 
    os.system(cmd)
 
    # Evaluate the result by calibrating a test image
    if validate:

        # Test image for calibration purposes
        testdir = '/beegfs/scratch/drt/20211114_EMIT_Infield/20211116_041506_UTC_InFieldScatter/' 
        darkfile = testdir+'20211116_042012_UTC_InFieldScatter_dark.raw'
        dnfile = testdir+'20211116_042022_UTC_InFieldScatter_580p0nm.raw'
        rdnfile = dnfile.replace('.raw','_rdn')
        cmd = 'python ../emitrdn.py --dark_file %s %s %s' % (darkfile,dnfile,rdnfile)
        os.system(cmd)
        
        # Extract the point spread function in the spectral dimension
        I = envi.open(rdnfile+'.hdr').load()
        I = np.squeeze(np.mean(I,axis=0)).T
        band = np.argmax(np.mean(I,axis=1))
        I = np.squeeze(I[band,:])
        
        # Plot the result to the screen
        curves.append(I)
        plt.semilogy(I)
        plt.show()

  # Save pretty plots
  if validate:
      np.savetxt('EMIT_l1bplots_SRF.txt', np.array(curves).T)
 
 
 
 
 
