import numpy as np
from scipy.signal import savgol_filter
from spectral.io import envi


def detect_shutter_states(input_file,band_num = 290,threshold = .05,transition_frames = 850,dark_buffer = 50):
    """Determine shutter states from image data.

       Assumes dark frame collection at start and end of flightline

    Args:
        input_file: Raw file path
        band_num: Band number to use for detection
        threshold: Slope threshold for detecting shutter closure
        transition_frames: Number of frames to open/close the shutter
        dark_buffer : Number of frames to buffer at shutter closure

    Returns:
        (np.array): Shutter states
                    -0 : Shutter closed, start of collection
                    -1 : Shutter opening
                    -2 : Shutter full open
                    -3 : Shutter closing
                    -4 : Shutter closed, end of collection
    """

    img = envi.open(input_file+ '.hdr')
    along_track_mean = img.read_band(band_num).mean(axis=1)
    #Calculate along track brightness slope
    along_track_deriv = np.abs(savgol_filter(along_track_mean,101,1,deriv=1))

    shutter_state= np.full(img.nrows,2)

    #Find dark boundaries
    dark_1_end,dark_2_start = np.argwhere(along_track_deriv > threshold)[[0,-1]].flatten()
    #Buffer at transition
    dark_1_end -=dark_buffer
    dark_2_start +=dark_buffer

    shutter_state[:dark_1_end]=0
    shutter_state[dark_1_end:dark_1_end+transition_frames]=1
    shutter_state[dark_2_start-transition_frames:dark_2_start]=3
    shutter_state[dark_2_start:]=4

    return shutter_state
