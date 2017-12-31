#!/usr/bin/env python3
from nistats import regression
import nibabel as nb
import numpy as np
from glob import glob
import os, pandas, sys, pdb, argparse
from os.path import join as pjoin
from nilearn import plotting
import pylab as plt

parser = argparse.ArgumentParser(description='stuff')
parser.add_argument('img_file', help='4d nifti file path')
parser.add_argument('tsv_file', help='tsv file containing nuisance regressors to be removed!')
parser.add_argument('out_path', help='output directory for saving new data file')
parser.add_argument('--col_names', help='which columns of TSV file to include as nuisance regressors. defaults to all.', nargs="+")
# can index dataframe using list of strings

args = parser.parse_args()

# img_file = sys.argv[0]
# tsv_file = sys.argv[1]

nii_ext = '.nii.gz'
tsv_ext= '.tsv'

# get files
# data_dir = pjoin(os.environ['HOME'], 'Documents', 'sample_data')
# img_file = glob(pjoin(data_dir, '*' + nii_ext))
# tsv_file = glob(pjoin(data_dir, '*' + tsv_ext))

img_file = args.img_file
tsv_file = args.tsv_file
data_dir = args.out_path

# assert(len(img_file)==1)
# assert(len(tsv_file)==1)
# img_file = img_file[0]
# tsv_file = tsv_file[0]
base_file = os.path.basename(img_file)
save_img_file = pjoin(data_dir, base_file[0:base_file.find('.')] + \
                      '_NR' + nii_ext)

# read in files
img = nb.load(img_file)
data = img.get_data()
df = pandas.read_csv(tsv_file, '\t', na_values='n/a')

# # remove columns with missing values
# df.dropna(axis=1, inplace=True)

Ntrs = df.as_matrix().shape[0]
print('# of TRs: ' + str(Ntrs))
assert(Ntrs==data.shape[len(data.shape)-1])

if args.col_names:
    df = df[args.col_names]

# fill in missing values with mean for that variable
for col in df.columns:
    if sum(df[col].isnull()) > 0:
        print('Filling in ' + str(sum(df[col].isnull())) + ' NaN value for ' + col)
        df[col] = df[col].fillna(np.mean(df[col]))

# add in intercept column into data frame
df['Int'] = 1

conf = df.as_matrix()
print('# of Confound Regressors: ' + str(len(df.columns)) + ' [Including Intercept]')

# prep data
data_re = np.reshape(data, (-1, Ntrs))
data_mean = np.mean(data_re, axis=1)
Nvox = len(data_mean)

# setup and run regression
model = regression.OLSModel(conf)
results = model.fit(data_re.T)
new_data = results.resid.T + np.reshape(data_mean, (Nvox, 1)) # add mean back into residuals
new_data_re = np.reshape(new_data, data.shape).astype('float32')

# save out new data file
affine = img.affine
new_img = nb.Nifti1Image(new_data_re, affine)
new_img.to_filename(save_img_file)

# mean img
#mean_img = nb.Nifti1Image(np.reshape(data_mean, (data.shape[0], data.shape[1], data.shape[2])), affine)
