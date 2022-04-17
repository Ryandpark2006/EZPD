import os
from app import APP_ROOT
import tensorflow as tf
import numpy as np
import mne
import matplotlib.pyplot as plt
# from mpld3 import fig_to_html
import scipy
import matplotlib
import pandas as pd
matplotlib.use('Agg')
from datetime import datetime

def NormalizeData(data):
  return (data - np.min(data)) / (np.max(data) - np.min(data))

def eeg_prediction():
  path = os.path.join(APP_ROOT, 'temp/eeg.bdf')
  raw = mne.io.read_raw_bdf(path, verbose=0)
  raw.crop(0,60).load_data()
  picks = mne.pick_types(raw.info, eeg=True, eog=False, stim=False)
  raw.notch_filter(np.arange(60, 241, 60), picks=picks, filter_length='auto', phase='zero', verbose=0)
  raw.filter(None, 60, fir_design='firwin')
  psd, fe = mne.time_frequency.psd_welch(raw, fmin=0, fmax=100, tmin=0, tmax=60, n_fft=2048, n_overlap=512, n_per_seg=None, picks=None, proj=False, n_jobs=1, reject_by_annotation=True, average='mean', window='hamming', verbose=None)

  # VISUALIZATION
  psdA, feA = mne.time_frequency.psd_welch(raw, fmin=0, fmax=40, tmin=0, tmax=60, n_fft=4096, n_overlap=256,
                                          n_per_seg=None,
                                          picks=None, proj=False, n_jobs=10, reject_by_annotation=True, average='mean',
                                          window='hamming', verbose=None)
  electrodes_list = pd.read_csv(os.path.join(APP_ROOT, 'temp/electrodes.tsv'), delimiter='\t').drop(
    labels=['name', 'type', 'material'], axis=1).to_numpy().tolist()
  locs_3d = [(x[0], x[1], x[2]) for x in electrodes_list]

  def aep(locations):
    r_min = 100000
    new_locs = []
    for location in locations:
      if r_min > np.sqrt(np.sum(np.square(location))):
        r_min = np.sqrt(np.sum(np.square(location)))

    for location in locations:
      r = np.sqrt((location[0]) ** 2 + (location[1]) ** 2 + (location[2] - r_min) ** 2)
      theta = np.arctan(location[1] / (location[0] + 0.000001))
      if location[0] < 0:
        theta += np.pi
      x = r * np.cos(theta)
      y = r * np.sin(theta)
      new_locs.append((x, y))
    return new_locs

  locs_2d = aep(locs_3d)

  x_loc = np.array([x[0] for x in locs_2d])
  y_loc = np.array([x[1] for x in locs_2d])

  x_loc = (x_loc - min(x_loc)) / (max(x_loc) - min(x_loc))
  y_loc = (y_loc - min(y_loc)) / (max(y_loc) - min(y_loc))

  dct = {'delta': (0, 4), 'theta': (4, 8), 'alpha': (8, 12), 'beta': (12, 25), 'gamma': (25, 10000)}

  def getFreqBand(f):
    for k in dct:
      if f >= dct[k][0] and f < dct[k][1]: return k
    return ''

  def rel_psd_electrode(row):
    bands = {k: 0 for k in dct}
    sumPower = 0
    for i, x in enumerate(row):
      bands[getFreqBand(feA[i])] += x
      sumPower += x
    return {k: bands[k] / sumPower for k in bands}

  key = {'delta': 0, 'theta': 1, 'alpha': 2, 'beta': 3, 'gamma': 4}
  x_loc = x_loc[:-1];
  y_loc = y_loc[:-1]

  def rel_bands():
    out = {k: [] for k in key}
    for i in range(32):
      rel = rel_psd_electrode(psdA[i])
      for k in rel:
        out[k].append(rel[k])
    return out

  relBands = rel_bands()
  for k in relBands:
    plt.close("all")
    N = 1000  # number of points for interpolation
    xy_center = [0.5, 0.5]  # center of the plot
    radius = 0.45  # radius

    x, y = x_loc, y_loc
    z = np.array(relBands[k])
    xi = np.linspace(0, 1, N)
    yi = np.linspace(0, 1, N)
    zi = scipy.interpolate.griddata((x, y), z, (xi[None, :], yi[:, None]), method='cubic')
    dr = xi[1] - xi[0]
    for i in range(N):
      for j in range(N):
        r = np.sqrt((xi[i] - xy_center[0]) ** 2 + (yi[j] - xy_center[1]) ** 2)
        if (r - dr / 2) > radius:
          zi[j, i] = "nan"

    # make figure
    fig = plt.figure()

    # set aspect = 1 to make it a circle
    ax = fig.add_subplot(111, aspect=1)

    CS = ax.contourf(xi, yi, zi, 60, cmap=plt.cm.jet, zorder=1)
    ax.contour(xi, yi, zi, 15, colors="grey", zorder=2)
    # cax = fig.add_axes([ax.get_position().x1,ax.get_position().y0,0.02,ax.get_position().y1-ax.get_position().y0])

    cbar = fig.colorbar(CS, ax=ax)
    circle = matplotlib.patches.Circle(xy=xy_center, radius=radius, edgecolor="k", facecolor="white", zorder=0.5)
    ax.add_patch(circle)
    c2 = matplotlib.patches.Circle(xy=xy_center, radius=radius, edgecolor="k", facecolor='none', zorder=2.1)
    ax.add_patch(c2)

    for loc, spine in ax.spines.items():
      spine.set_linewidth(0)

    # remove the ticks
    ax.set_xticks([])
    ax.set_yticks([])

    # Add some body parts. Hide unwanted parts by setting the zorder low
    # add two ears
    circle = matplotlib.patches.Ellipse(xy=[0 / 4 + 0.05, 2 / 4], width=0.5 / (4 * (0.5 / 0.45)),
                                        height=1.0 / (4 * (0.5 / 0.45)), angle=0, edgecolor="k", facecolor="w",
                                        zorder=0)
    ax.add_patch(circle)
    circle = matplotlib.patches.Ellipse(xy=[4 / 4 - 0.05, 2 / 4], width=0.5 / (4 * (0.5 / 0.45)),
                                        height=1.0 / (4 * (0.5 / 0.45)), angle=0, edgecolor="k", facecolor="w",
                                        zorder=0)
    ax.add_patch(circle)
    # add a nose
    xy = [[1.5 / 4, 3 / 4 - 0.05], [2 / 4, 4.5 / 4 - 0.05], [2.5 / 4, 3 / 4 - 0.05]]
    polygon = matplotlib.patches.Polygon(xy=xy, facecolor="w", zorder=0, edgecolor='k')
    ax.add_patch(polygon)

    # set axes limits
    ax.set_xlim(-0.5 / 4, 4.5 / 4)
    ax.set_ylim(-0.5 / 4, 4.5 / 4)
    ax.set_title(f'{k.capitalize()} Frequencies')
    plt.savefig(os.path.join(APP_ROOT, f'static/{k}.png'))
  now = datetime.now().time()
  print("Time After Viz =", now)
  psd = NormalizeData(np.asarray(psd))
  model = tf.keras.models.load_model(os.path.join(APP_ROOT, 'PSDtypeModel.h5'))
  type_pred = model.predict(np.array([psd]))[0]
  type = np.argmax(type_pred)
  print(type)
  type_pred = type_pred[type]
  os.remove(path)
  if type == 0:
    return [(type, type_pred), (-1, -1), (-1, -1)]
  else:
    loc_model = tf.keras.models.load_model(os.path.join(APP_ROOT, 'locModel.h5'))
    sev_model = tf.keras.models.load_model(os.path.join(APP_ROOT, 'severityModel.h5'))
    loc_pred = loc_model.predict(np.array([psd]))[0][0]
    loc = 1 if loc_pred > 0.5 else 0; loc_pred = loc_pred if loc == 1 else 1.0 - loc_pred
    sev_pred = sev_model.predict(np.array([psd]))[0][0]
    sev = 1 if sev_pred > 0.5 else 0; sev_pred = sev_pred if sev == 1 else 1.0 - sev_pred
    return [(type, type_pred), (loc, loc_pred), (sev, sev_pred)]