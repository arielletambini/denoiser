#!/usr/bin/env python3
import argparse
from denoiser.denoiser import denoise

parser = argparse.ArgumentParser(description='Function for performing nuisance regression. Saves resulting output '
                                             'nifti file, information about nuisance regressors and motion (html '
                                             'report), and outputs nibabel object containing clean data')
parser.add_argument('img_file', help='4d nifti img: file path or nibabel object loaded into memory')
parser.add_argument('tsv_file', help='tsv file containing nuisance regressors to be removed')
parser.add_argument('out_path', help='output directory for saving new data file')
parser.add_argument('--col_names',
                    help='which columns of TSV file to include as nuisance regressors. defaults to ALL columns.',
                    nargs="+")
parser.add_argument('--hp_filter', help='frequency cut-off for high pass filter (removing low frequencies). Recommend '
                                        '.009 Hz')
parser.add_argument('--lp_filter', help='frequency cut-off for low pass filter (removing high frequencies). Recommend '
                                        '.1 Hz for non-task data')
parser.add_argument('--out_figure_path',
                    help='output directory for saving figures. Defaults to location of out_path + _figures')
parser.add_argument('--fd_col_name',
                    help='Column in tsv_file that contains Framewise Displacement values.')


args = parser.parse_args()

img_file = args.img_file
tsv_file = args.tsv_file
out_path = args.out_path
col_names = args.col_names
hp_filter = args.hp_filter
lp_filter = args.lp_filter
out_figure_path = args.out_figure_path
fd_col_name = args.fd_col_name

if __name__ == "__main__":
    denoise(img_file, tsv_file, out_path, col_names, hp_filter, lp_filter, out_figure_path, fd_col_name)
