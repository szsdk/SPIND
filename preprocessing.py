# Copyright © 2018 Liu Lab, Beijing Computational Science Research Center <http://liulab.csrc.ac.cn>

# Authors:
#   2017-2018      Xuanxuan Li <lxx2011011580@gmail.com>
#   2017-2018      Chufeng Li <chufengl@asu.edu>
#   2017      Richard Kirian <rkirian@asu.edu>
#   2017      Nadia Zatsepin <Nadia.Zatsepin@asu.edu>
#   2017      John Spence <spence@asu.edu>
#   2017      Haiguang Liu <hgliu@csrc.ac.cn>

# This file is part of SPIND.

# SPIND is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# SPIND is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with SPIND.  If not, see <http://www.gnu.org/licenses/>.


"""
Usage: pytohn preprocessing.py config.yml original_peak_file output_peak_file

Note: 
* Original peak file should be a txt file consistes 4 colums, including 
    peakx(in pixel) peaky(in pixel) intensity snr
* Exp. parameters should be specified in config.yml.
"""

import sys
import yaml
import numpy as np
from numpy.linalg import norm 


h_ = 4.135667662E-15  # Planck constant in eV*s
c_ = 2.99792458E8  # light speed in m/sec

def det2fourier(det_xy, wave_length, det_dist):
  nb_xy = len(det_xy)
  det_dist = np.ones(nb_xy) * det_dist
  det_dist = np.reshape(det_dist, (-1, 1))
  q1 = np.hstack((det_xy, det_dist))
  q1_norm = np.sqrt(np.diag(q1.dot(q1.T)))
  q1_norm = q1_norm.reshape((-1, 1)).repeat(3, axis=1)
  q1 = q1 / q1_norm
  q0 = np.asarray([0., 0., 1.])
  q0 = q0.reshape((1,-1)).repeat(nb_xy, axis=0)
  q = 1. / wave_length * (q1 - q0)
  return q


config_file = sys.argv[1]
print('loading configuration from %s' % config_file)
config = yaml.load(open(config_file))
peaks_file = sys.argv[2]
print('loading peaks from %s' % peaks_file)
peaks = np.loadtxt(peaks_file)
output_file = sys.argv[3]
print("writing processed peak file to %s" % output_file)

wave_length = config['wave length']
det_dist = config['detector distance']
pixel_size = config['pixel size']

Npeaks = peaks.shape[0]
peaks_xy = peaks[:, 0:2] * pixel_size
qs = det2fourier(peaks_xy, wave_length, det_dist)
res = 1. / norm(qs, axis=1) * 1E10  # in angstrom

data = np.zeros((Npeaks, 8))
data[:, 0:2] = peaks_xy
data[:, 2] = peaks[:, 2]
data[:, 3] = peaks[:, 3]
data[:, 4:7] = qs
data[:, 7] = res
np.savetxt(output_file, data, fmt='%.5e %.5e %.5e %5.2f %.5e %.5e %.5e %5.2f')