from nistats import regression
import nibabel as nb
import numpy as np
from glob import glob
import os, pandas, sys, pdb
from os.path import join as pjoin
from nilearn import plotting
import pylab as plt

# img_file = sys.argv[0]
# tsv_file = sys.argv[1]

nii_ext = '.nii.gz'
tsv_ext= '.tsv'

# get files
data_dir = pjoin(os.environ['HOME'], 'Documents', 'sample_data')
img_file = glob(pjoin(data_dir, '*' + nii_ext))
tsv_file = glob(pjoin(data_dir, '*' + tsv_ext))
assert(len(img_file)==1)
assert(len(tsv_file)==1)
img_file = img_file[0]
tsv_file = tsv_file[0]
base_file = os.path.basename(img_file)
save_img_file = pjoin(data_dir, base_file[0:base_file.find('.')] + \
                      '_NR' + nii_ext)


# read in files
img = nb.load(img_file)
data = img.get_data()
df = pandas.read_csv(tsv_file, '\t', na_values='n/a')

# add in intercept column into data frame
df['Int'] = 1

# remove columns with missing values
df.dropna(axis=1, inplace=True)

Ntrs = df.as_matrix().shape[0]
print('# of TRs: ' + str(Ntrs))
assert(Ntrs==data.shape[len(data.shape)-1])
conf = df.as_matrix()
print('# of Confound Regressors: ' + str((conf.shape[1])-1))

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
print(save_img_file)
new_img.to_filename(save_img_file)

# mean img
#mean_img = nb.Nifti1Image(np.reshape(data_mean, (data.shape[0], data.shape[1], data.shape[2])), affine)
