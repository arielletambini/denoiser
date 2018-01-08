#!/usr/bin/env python3
from nistats import regression
from nistats import reporting
from nistats.design_matrix import make_design_matrix
import nibabel as nb
import numpy as np
import os, pandas, sys, pdb, argparse, copy, scipy, jinja2
from os.path import join as pjoin
from nilearn import plotting
from nilearn.signal import butterworth
import matplotlib
import pylab as plt
import seaborn as sns
from nilearn._utils.niimg import load_niimg

parser = argparse.ArgumentParser(description='stuff')
parser.add_argument('img_file', help='4d nifti img: file path or nibabel object loaded into memory')
parser.add_argument('tsv_file', help='tsv file containing nuisance regressors to be removed')
parser.add_argument('out_path', help='output directory for saving new data file')
parser.add_argument('--col_names', help='which columns of TSV file to include as nuisance regressors. defaults to ALL columns.',
                    nargs="+")
parser.add_argument('--hp_filter', help='frequency cut-off for high pass filter (removing low frequencies). Recommend '
                    '.009 Hz')
parser.add_argument('--lp_filter', help='frequency cut-off for low pass filter (removing high frequencies). Recommend '
                    '.1 Hz for non-task data')
parser.add_argument('--out_figure_path', help='output directory for saving figures. Defaults to location of out_path + _figures')

args = parser.parse_args()

img_file = args.img_file
tsv_file = args.tsv_file
out_path = args.out_path
col_names = args.col_names
hp_filter = args.hp_filter
lp_filter = args.lp_filter
out_figure_path = args.out_figure_path

def denoise(img_file, tsv_file, out_path, col_names=False, hp_filter=False, lp_filter=False, out_figure_path=False):

    nii_ext = '.nii.gz'
    FD_thr = [.5]
    sc_range = np.arange(-1, 3)
    constant = 'constant'

    # read in files
    img = load_niimg(img_file)
    # get file info
    img_name = os.path.basename(img.get_filename())
    save_img_file = pjoin(out_path, img_name[0:img_name.find('.')] + \
                          '_NR' + nii_ext)
    data = img.get_data()
    df_orig = pandas.read_csv(tsv_file, '\t', na_values='n/a')
    df = copy.deepcopy(df_orig)
    Ntrs = df.as_matrix().shape[0]
    print('# of TRs: ' + str(Ntrs))
    assert (Ntrs == data.shape[len(data.shape) - 1])

    # select columns to use as nuisance regressors
    str_append = '  [ALL regressors in CSV]'
    if col_names:
        df = df[col_names]
        str_append = '  [SELECTED regressors in CSV]'

    # fill in missing nuisance values with mean for that variable
    for col in df.columns:
        if sum(df[col].isnull()) > 0:
            print('Filling in ' + str(sum(df[col].isnull())) + ' NaN value for ' + col)
            df[col] = df[col].fillna(np.mean(df[col]))
    print('# of Confound Regressors: ' + str(len(df.columns)) + str_append)

    # implement HP filter in regression
    TR = img.header.get_zooms()[-1]
    frame_times = np.arange(Ntrs) * TR
    if hp_filter:
        hp_filter = float(hp_filter)
        assert(hp_filter > 0)
        period_cutoff = 1. / hp_filter
        df = make_design_matrix(frame_times, period_cut=period_cutoff, add_regs=df.as_matrix(),
                                add_reg_names=df.columns.tolist())
        # fn adds intercept into dm

        hp_cols = [col for col in df.columns if 'drift' in col]
        print('# of High-pass Filter Regressors: ' + str(len(hp_cols)))
    else:
        # add in intercept column into data frame
        df[constant] = 1
        print('No High-pass Filter Applied')

    dm = df.as_matrix()

    # prep data
    data = np.reshape(data, (-1, Ntrs))
    data_mean = np.mean(data, axis=1)
    Nvox = len(data_mean)

    # setup and run regression
    model = regression.OLSModel(dm)
    results = model.fit(data.T)
    if not hp_filter:
        results_orig_resid = copy.deepcopy(results.resid) # save for rsquared computation

    # apply low-pass filter
    if lp_filter:
        # input to butterworth fn is time x voxels
        low_pass = float(lp_filter)
        Fs = 1. / TR
        if low_pass >= Fs / 2:
            raise ValueError('Low pass filter cutoff if too close to the Nyquist frequency (%s)' % (Fs / 2))

        results.resid = butterworth(results.resid, sampling_rate=Fs, low_pass=low_pass, high_pass=None)
        print('Low-pass Filter Applied: < ' + low_pass)

    # add mean back into data
    clean_data = results.resid.T + np.reshape(data_mean, (Nvox, 1))  # add mean back into residuals

    # save out new data file
    clean_data = np.reshape(clean_data, img.shape).astype('float32')
    new_img = nb.Nifti1Image(clean_data, img.affine)
    new_img.to_filename(save_img_file)

    ######### generate Rsquared map for confounds only
    if hp_filter:
        # first remove low-frequency information from data
        hp_cols.append(constant)
        model_first = regression.OLSModel(df[hp_cols].as_matrix())
        results_first = model_first.fit(data.T)
        results_first_resid = copy.deepcopy(results_first.resid)
        del results_first, model_first

        # compute sst - borrowed from matlab
        sst = np.square(np.linalg.norm(results_first_resid -
                                       np.mean(results_first_resid, axis=0), axis=0))

        # now regress out 'true' confounds to estimate their Rsquared
        nr_cols = [col for col in df.columns if 'drift' not in col]
        model_second = regression.OLSModel(df[nr_cols].as_matrix())
        results_second = model_second.fit(results_first_resid)

        # compute sse - borrowed from matlab
        sse = np.square(np.linalg.norm(results_second.resid, axis=0))

        del results_second, model_second, results_first_resid

    elif not hp_filter:
        # compute sst - borrowed from matlab
        sst = np.square(np.linalg.norm(data.T -
                                       np.mean(data.T, axis=0), axis=0))

        # compute sse - borrowed from matlab
        sse = np.square(np.linalg.norm(results_orig_resid, axis=0))

        del results_orig_resid

    # compute rsquared of nuisance regressors
    zero_idx = scipy.logical_and(sst == 0, sse == 0)
    sse[zero_idx] = 1
    sst[zero_idx] = 1  # would be NaNs - become rsquared = 0
    rsquare = 1 - np.true_divide(sse, sst)
    rsquare[np.isnan(rsquare)] = 0

    ######### Visualizing DM & outputs
    fontsize = 12
    fontsize_title = 14
    if not out_figure_path:
        out_figure_path = save_img_file[0:save_img_file.find('.')] + '_figures'

    if not os.path.isdir(out_figure_path):
        os.mkdir(out_figure_path)
    png_append = '_' + img_name[0:img_name.find('.')] + '.png'
    print('Output directory: ' + out_figure_path)

    # DM corr matrix
    cm = df[df.columns[0:-1]].corr()
    mask = np.zeros_like(cm, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    sz = 8
    if cm.shape[0] > sz:
        sz = sz + ((cm.shape[0] - sz) * .3)
    fig, ax = plt.subplots(figsize=(sz, sz))
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    sns.heatmap(cm, mask=mask, cmap=cmap, center=0, vmax=cm[cm < 1].max().max(), vmin=cm[cm < 1].min().min(),
                square=True, linewidths=.5, cbar_kws={"shrink": .6})
    ax.set_xticklabels(ax.get_xticklabels(), rotation=60, ha='right', fontsize=fontsize)
    ax.set_yticklabels(cm.columns.tolist(), rotation=-30, va='bottom', fontsize=fontsize)
    ax.set_title('Nuisance Corr. Matrix', fontsize=fontsize_title)
    plt.tight_layout()
    file_corr_matrix = pjoin(out_figure_path, 'Corr_matrix_regressors' + png_append)
    fig.savefig(file_corr_matrix)
    plt.close(fig)
    del fig, ax

    # DM of Nuisance Regressors (all)
    tr_label = 'TR (Volume #)'
    fig, ax = plt.subplots(figsize=(4, sz))
    reporting.plot_design_matrix(df, ax=ax)
    ax.set_title('Nuisance Design Matrix', fontsize=fontsize_title)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=60, ha='right', fontsize=fontsize)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=fontsize)
    ax.set_ylabel(tr_label, fontsize=fontsize)
    plt.tight_layout()
    file_design_matrix = pjoin(out_figure_path, 'Design_matrix' + png_append)
    fig.savefig(file_design_matrix)
    plt.close(fig)
    del fig, ax

    # FD timeseries plot
    FD = 'FD'
    poss_names = ['FramewiseDisplacement', FD, 'framewisedisplacement', 'fd']
    fd_idx = [df_orig.columns.__contains__(i) for i in poss_names]
    if np.sum(fd_idx) > 0:
        FD_name = poss_names[fd_idx == True]
        if sum(df_orig[FD_name].isnull()) > 0:
            df_orig[FD_name] = df_orig[FD_name].fillna(np.mean(df_orig[FD_name]))
        y = df_orig[FD_name].as_matrix()
        Nremove = []
        sc_idx = []
        for thr_idx, thr in enumerate(FD_thr):
            idx = y >= thr
            sc_idx.append(copy.deepcopy(idx))
            for iidx in np.where(idx)[0]:
                for buffer in sc_range:
                    curr_idx = iidx + buffer
                    if curr_idx >= 0 and curr_idx <= len(idx):
                        sc_idx[thr_idx][curr_idx] = True
            Nremove.append(np.sum(sc_idx[thr_idx]))

        Nplots = len(FD_thr)
        sns.set(font_scale=1.5)
        sns.set_style('ticks')
        fig, axes = plt.subplots(Nplots, 1, figsize=(12, 4), squeeze=False)
        sns.despine()
        bound = .4
        for curr in np.arange(0, Nplots):
            axes[curr, 0].plot(y)
            axes[curr, 0].plot((-bound, Ntrs + bound), FD_thr[curr] * np.ones((1, 2))[0], '--', color='black')
            axes[curr, 0].scatter(np.arange(0, Ntrs), y, s=20)

            if Nremove[curr] > 0:
                info = scipy.ndimage.measurements.label(sc_idx[curr])
                for cluster in np.arange(1, info[1] + 1):
                    temp = np.where(info[0] == cluster)[0]
                    axes[curr, 0].axvspan(temp.min() - bound, temp.max() + bound, alpha=.5, color='red')
                axes[curr, 0].set_ylabel('Framewise Disp. (' + FD + ')')
                axes[curr, 0].set_title(FD + ': ' + str(100 * Nremove[curr] / Ntrs)[0:4]
                                        + '% of scan (' + str(Nremove[curr]) + ' volumes) would be scrubbed (FD thr.= ' +
                                        str(FD_thr[curr]) + ')')
                plt.text(Ntrs + 1, FD_thr[curr] - .01, FD + ' = ' + str(FD_thr[curr]), fontsize=fontsize)
                axes[curr, 0].set_xlim((-bound, Ntrs + 8))
        plt.tight_layout()
        axes[curr, 0].set_xlabel(tr_label)
        file_fd_plot = pjoin(out_figure_path, FD + '_timeseries' + png_append)
        fig.savefig(file_fd_plot)
        plt.close(fig)
        del fig, axes
        print(FD + ' timeseries plot saved')

    else:
        print(FD + ' not found: ' + FD + ' timeseries not plotted')
        file_fd_plot = None

    # Display T-stat maps for nuisance regressors
    # create mean img
    img_size = (img.shape[0], img.shape[1], img.shape[2])
    mean_img = nb.Nifti1Image(np.reshape(data_mean, img_size), img.affine)
    mx = []
    for idx, col in enumerate(df.columns):
        if not 'drift' in col and not constant in col:
            con_vector = np.zeros((1, df.shape[1]))
            con_vector[0, idx] = 1
            con = results.Tcontrast(con_vector)
            mx.append(np.max(np.absolute([con.t.min(), con.t.max()])))
    mx = .8 * np.max(mx)
    t_png = 'Tstat_'
    file_tstat = []
    for idx, col in enumerate(df.columns):
        if not 'drift' in col and not constant in col:
            con_vector = np.zeros((1, df.shape[1]))
            con_vector[0, idx] = 1
            con = results.Tcontrast(con_vector)
            m_img = nb.Nifti1Image(np.reshape(con, img_size), img.affine)

            title_str = col + ' Tstat'
            fig = plotting.plot_stat_map(m_img, mean_img, threshold=3, colorbar=True, display_mode='z', vmax=mx,
                                         title=title_str,
                                         cut_coords=7)
            file_temp = pjoin(out_figure_path, t_png + col + png_append)
            fig.savefig(file_temp)
            file_tstat.append({'name': col, 'file': file_temp})
            plt.close()
            del fig, file_temp
            print(title_str + ' map saved')

    # Display R-sq map for nuisance regressors
    m_img = nb.Nifti1Image(np.reshape(rsquare, img_size), img.affine)
    title_str = 'Nuisance Rsq'
    mx = .95 * rsquare.max()
    fig = plotting.plot_stat_map(m_img, mean_img, threshold=.2, colorbar=True, display_mode='z', vmax=mx,
                                 title=title_str,
                                 cut_coords=7)
    file_rsq_map = pjoin(out_figure_path, 'Rsquared' + png_append)
    fig.savefig(file_rsq_map)
    plt.close()
    del fig
    print(title_str + ' map saved')

    templateLoader = jinja2.FileSystemLoader(searchpath="/")
    templateEnv = jinja2.Environment(loader=templateLoader)

    TEMPLATE_FILE = pjoin(os.getcwd(), "report_template.html")
    template = templateEnv.get_template(TEMPLATE_FILE)

    templateVars = {"img_file": img_file,
                    "save_img_file": save_img_file,
                    "Ntrs": Ntrs,
                    "tsv_file": tsv_file,
                    "col_names": col_names,
                    "hp_filter": hp_filter,
                    "lp_filter": lp_filter,
                    "file_design_matrix": file_design_matrix,
                    "file_corr_matrix": file_corr_matrix,
                    "file_fd_plot": file_fd_plot,
                    "file_rsq_map": file_rsq_map,
                    "file_tstat": file_tstat
                    }

    outputText = template.render(templateVars)

    html_file = pjoin(out_figure_path, img_name[0:img_name.find('.')] + '.html')
    with open(html_file, "w") as f:
        f.write(outputText)

    print('')
    print('HTML report: ' + html_file)

denoise(img_file, tsv_file, out_path, col_names, hp_filter, lp_filter, out_figure_path)