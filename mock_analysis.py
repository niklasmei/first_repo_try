from first_repo_try.data.data_tools import generate_data
from first_repo_try.analysis.analysis_tools import analyse_data
from first_repo_try.plotting.plotting_tools import plot_analysis

raw_data = generate_data()
fit_results = analyse_data(raw_data=raw_data)

plot_analysis(raw_data=raw_data,
              fit_results=fit_results)