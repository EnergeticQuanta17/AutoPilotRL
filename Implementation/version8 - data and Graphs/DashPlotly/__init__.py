from .app3_study_table import study_table as StudyTable
from .app6_plotting_all_one_graph import scatter_lines_dropdown as ScatterLine
from .app10_all_trials_histogram import color_scale_histogram as ColorScaleHistogram
from .app11_correlation_between_two_hyps import correlation_hyp_vs_hyp as InterCorrelation
from .app17_3d import dim3_correlation as TrivariateCorrelation
from .app18_box_plots_on_all_hyps import boxplot_hyperparameters as BoxPlotCheckBox
from .app19_normal_distribution import normal_distribution_fit as NormalDistributionFitter
from .app22_fit_beta_distribution import beta_distribution as BetaDistributionFitter


__all__ = [
    'StudyTable',
    'ScatterLine',
    'ColorScaleHistogram',
    'InterCorrelation',
    'TrivariateCorrelation',
    'BoxPlotCheckBox',
    'NormalDistributionFitter',
    'BetaDistributionFitter',
    
]