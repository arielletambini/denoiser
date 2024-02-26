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
parser.add_argument('--FD_thr',
                    help='Set a custom FD threshold (in mm) for use when counting high-motion volumes.')
parser.add_argument('--bids',
                    help='If set to True, denoiser will expect inputs to follow the BIDS derivaties spec,\
                            and Denoiser outputs will also follow the spec (and include required JSON files).')
parser.add_argument('--strategy_name',
                    help='Required if bids=True. Specify a name for the nuisance cleaning strategy to be\
                            included in the variant field of the output filename.')
parser.add_argument('--template_file',
                    help='Required if running denoiser as part of a Nipype workflow. Absolute path to an HTML\
                            template for use when creating visual reports.')
parser.add_argument('--sink_link',
                    help='Required if running denoiser as part of a Nipype workflow. Dummy variable which is needed\
                            to force Nipype to wait for the output directory to be created before running denoiser.\
                            The output of the workflow DataSink node should be passed into sink_link.')

args = parser.parse_args()

img_file = args.img_file
tsv_file = args.tsv_file
out_path = args.out_path
col_names = args.col_names
hp_filter = args.hp_filter
lp_filter = args.lp_filter
out_figure_path = args.out_figure_path
fd_col_name = args.fd_col_name
FD_thr = args.FD_thr
bids = args.bids
strategy_name = args.strategy_name
template_file = args.template_file
sink_link = args.sink_link

if __name__ == "__main__":
    denoise(img_file, tsv_file, out_path, col_names, hp_filter, lp_filter,
            out_figure_path, fd_col_name, FD_thr, bids, strategy_name, template_file, sink_link)

