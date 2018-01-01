#!/usr/bin/env python3
from nistats import regression
from nistats.design_matrix import make_design_matrix
import nibabel as nb
import numpy as np
import os, pandas, sys, pdb, argparse
from os.path import join as pjoin
from nilearn import plotting
from nilearn.signal import butterworth
import pylab as plt

parser = argparse.ArgumentParser(description='stuff')
parser.add_argument('img_file', help='4d nifti file path')
parser.add_argument('tsv_file', help='tsv file containing nuisance regressors to be removed!')
parser.add_argument('out_path', help='output directory for saving new data file')
parser.add_argument('--col_names', help='which columns of TSV file to include as nuisance regressors. defaults to all.',
                    nargs="+")
parser.add_argument('--hp_filter', help='frequency cut-off for high pass filter (removing low frequencies). Recommend '
                    '.009 Hz')
parser.add_argument('--lp_filter', help='frequency cut-off for low pass filter (removing high frequencies). Recommend '
                    '.1 Hz for non-task data')

args = parser.parse_args()

nii_ext = '.nii.gz'
tsv_ext= '.tsv'

# get files
img_file = args.img_file
tsv_file = args.tsv_file
data_dir = args.out_path

base_file = os.path.basename(img_file)
save_img_file = pjoin(data_dir, base_file[0:base_file.find('.')] + \
                      '_NR' + nii_ext)

# read in files
img = nb.load(img_file)
data = img.get_data()
df = pandas.read_csv(tsv_file, '\t', na_values='n/a')

Ntrs = df.as_matrix().shape[0]
print('# of TRs: ' + str(Ntrs))
assert(Ntrs==data.shape[len(data.shape)-1])

# select columns to use as nuisance regressors
str_append = ' - ALL regressors in CSV'
if args.col_names:
    df = df[args.col_names]
    str_append = ' - SELECTED regressors in CSV'

# fill in missing nuisance values with mean for that variable
for col in df.columns:
    if sum(df[col].isnull()) > 0:
        print('Filling in ' + str(sum(df[col].isnull())) + ' NaN value for ' + col)
        df[col] = df[col].fillna(np.mean(df[col]))

# dm = df.as_matrix()
print('# of Confound Regressors: ' + str(len(df.columns)))

# implement HP filter in regression
TR = img.header.get_zooms()[-1]
frame_times = np.arange(Ntrs) * TR
if args.hp_filter:
    period_cutoff = 1./float(args.hp_filter)
    df = make_design_matrix(frame_times, period_cut=period_cutoff, add_regs=df.as_matrix(), add_reg_names=df.columns.tolist())
    # fn adds intercept into dm

    hp_cols = [col for col in df.columns if 'drift' in col]
    print('# of High-pass Filter Regressors: ' + str(len(hp_cols)))
else:
    # add in intercept column into data frame
    df['constant'] = 1

dm = df.as_matrix()

# prep data
data = np.reshape(data, (-1, Ntrs))
data_mean = np.mean(data, axis=1)
Nvox = len(data_mean)

# setup and run regression
model = regression.OLSModel(dm)
results = model.fit(data.T)

# apply low-pass filter
if args.lp_filter:
    # input to butterworth fn is time x voxels
    low_pass = float(args.lp_filter)
    Fs = 1./TR
    if low_pass >= Fs/2:
        raise ValueError('Low pass filter cutoff if too close to the Nyquist frequency (%s)' % (Fs/2))

    results.resid = butterworth(results.resid, sampling_rate=Fs, low_pass=low_pass, high_pass=None)

# add mean back into data
clean_data = results.resid.T + np.reshape(data_mean, (Nvox, 1)) # add mean back into residuals

# save out new data file
clean_data = np.reshape(clean_data, img.shape).astype('float32')
new_img = nb.Nifti1Image(clean_data, img.affine)
new_img.to_filename(save_img_file)

# mean img
#mean_img = nb.Nifti1Image(np.reshape(data_mean, (data.shape[0], data.shape[1], data.shape[2])), affine)
