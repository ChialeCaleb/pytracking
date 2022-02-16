from pytracking.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.eotb_path = '/data/wjl/FE108/'
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.network_path = '/home/dspwjl/workspace/networks/'    # Where tracking networks are stored.
    settings.result_plot_path = '/home/dspwjl/workspace/result_plots/'
    settings.results_path = '/home/dspwjl/workspace/tracking_results/'    # Where to store tracking results
    settings.segmentation_path = '/home/dspwjl/workspace/segmentation_results/'
    settings.tn_packed_results_path = ''

    return settings

