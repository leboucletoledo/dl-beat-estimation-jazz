import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import librosa as lbrs
import numpy as np

PARAMS = {'legend.fontsize': 'x-large',
        'figure.figsize': (24, 8),
        'figure.dpi': 80,
        'figure.titlesize': 24,
        'axes.labelsize': 'large',
        'axes.titlesize':'xx-large',
        'xtick.labelsize':'large',
        'ytick.labelsize':'large',
         }

COLORS = ['royalblue',
          'orange',
          'green',
          'firebrick',
          'darkcyan',
          'violet',
          ]

plt.rcParams.update(PARAMS)

def get_spectrogram(waveform, 
                    sr, 
                    bt_annotations, 
                    dnbt_annotations=None, 
                    bt_predictions=None,
                    dnbt_predictions=None,
                    hop_length=512,
                    spec_type='log', 
                    title='', 
                    spec_plot=True,
                    x_lims=[],
                    ):

    if spec_type=='log':
        spec_amp = np.abs(lbrs.stft(waveform, hop_length=hop_length))
        img = lbrs.amplitude_to_db(spec_amp, ref=np.max)
        color = 'r'
        cmap='BuGn'
        fmax=None
    elif spec_type=='mel':
        spec_mel = lbrs.feature.melspectrogram(y=waveform, sr=sr)
        img = lbrs.power_to_db(spec_mel, ref=np.max)
        color = 'w'
        cmap='viridis'
        fmax=8000
        
    fig, ax = plt.subplots(nrows=1, ncols=1, sharex=True)
    spec = lbrs.display.specshow(img, 
                                 x_axis='time', 
                                 y_axis=spec_type, 
                                 sr=sr, 
                                 hop_length=512, 
                                 cmap=cmap, 
                                 fmax=fmax
                                 )
    ax.set(title=title)
    if bt_predictions is not None and dnbt_predictions is not None:
        ax.vlines(dnbt_annotations, 
                  hop_length * 2, 
                  sr / 2, 
                  color=color, 
                  label='Downbeat annotations')
        ax.vlines(bt_annotations, 
                  hop_length * 2, 
                  sr / 2, 
                  linestyles='dotted', 
                  color=color, 
                  label='Beat annotations',
                  )
        ax.vlines(dnbt_predictions, 
                  0, 
                  hop_length, 
                  color='m', 
                  label='Downbeat predictions',
                  )
        ax.vlines(bt_predictions, 
                  0, 
                  hop_length, 
                  linestyles='dotted', 
                  color='m', 
                  label='Beat predictions',
                  )
    elif dnbt_annotations is not None:
        ax.vlines(dnbt_annotations, 
                  0, 
                  sr / 2, 
                  color=color, 
                  label='Downbeat annotations',
                  )
        ax.vlines(bt_annotations, 
                  0, 
                  sr / 2, 
                  linestyles='dotted', 
                  color=color, 
                  label='Beat annotations',
                  )
    else:
        ax.vlines(bt_annotations, 
                  0, 
                  sr / 2, 
                  linestyles='dotted', 
                  color=color, 
                  label='Beat annotations',
                  )
    
    if x_lims:
        assert len(x_lims)==2, 'x_lims must be of length 2'
        ax.set_xlim(left=x_lims[0], 
                    right=x_lims[1],
                    )
    
    ax.set_title(title)
    fig.colorbar(spec, ax=ax, format="%+2.f dB")
    ax.set_xlabel('Time (s)')
    ax.legend(loc='upper right', ncol=2)
    ax.set_ylabel('Frequency (Hz)')
    ax.xaxis.set_major_formatter(mtick.ScalarFormatter())

    if spec_plot:
        plt.show()
    else:
        plt.close()
    
    return fig, spec

def get_barplot(dict_ind_metrics, 
                eval_type, 
                title='', 
                y_label='', 
                track_range=[],
                ):
    
    track_list = list(dict_ind_metrics.keys())
    
    # Select a range of track metrics to plot
    if track_range:
        _rng = track_range
        assert len(_rng)==2 and _rng[0]<_rng[1]
        track_list = track_list[_rng[0]:_rng[1]]
    
    eval_list = list(dict_ind_metrics[track_list[0]].keys())
    metric_list = list(dict_ind_metrics[track_list[0]][eval_list[0]].keys())
    x = np.arange(len(track_list))
    
    # Bar width and separation
    width_list = [-5/2, -3/2, -1/2, 1/2, 3/2, 5/2] 
    width = 0.15

    fig = plt.figure()
    ax = fig.add_subplot()
    ax.set_prop_cycle('color',
                      COLORS,
                      )
    
    for m, w in zip(metric_list, width_list):
        res = []
        for track in track_list:
            res.append(dict_ind_metrics[track][eval_type][m])
        rects = ax.bar(x + width * w, res, width)

        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom',
                        rotation=45,
                        )

    ax.set_ylabel(y_label)
    ax.set_ylim(top=1.2)
    ax.set_title(eval_type)
    ax.set_xticks(x)
    ax.set_xticklabels(track_list, ha='right', rotation=60, )
    ax.legend(metric_list, loc='upper right', ncol=3)
    fig.suptitle(title)
    plt.show()

def get_boxplot(dict_ind_metrics, eval_type, title='', y_label=''):
    track_list = list(dict_ind_metrics.keys())
    eval_list = list(dict_ind_metrics[track_list[0]].keys())
    metric_list = list(dict_ind_metrics[track_list[0]][eval_list[0]].keys())
    
    metric_arr = np.zeros((len(track_list), len(metric_list)))
    for i, t in enumerate(track_list):      
        for j, m in enumerate(metric_list):
            metric_arr[i][j] = dict_ind_metrics[t][eval_type][m]
    
    fig = plt.figure(figsize=(16, 6))
    ax = fig.add_subplot()
    bp = ax.boxplot(metric_arr, 
                    patch_artist=True, 
                    labels=metric_list,
                    manage_ticks=False,
                    )

    for i in range(len(bp['boxes'])):
        bp['boxes'][i].set(facecolor=COLORS[i])
        bp['medians'][i].set(color='black', linestyle='--')

    ax.set_ylabel(y_label)
    ax.set_ylim(top=1.10, bottom= -0.10)
    ax.set_title(eval_type)
    fig.suptitle(title)
    plt.xticks(range(1, len(metric_list)+1), metric_list)
    plt.show()