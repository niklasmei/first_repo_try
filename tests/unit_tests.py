import os.path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# This will produce the path to the test data on any OS and machine,
# if run inside unit_tests.py

# Strictly needed
TEST_DATA_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", 'tests', 'test_data')
)

def generate_data(n: int = 1000) -> pd.DataFrame:
    """Generate data points.

    Args:
        n: Number of datapoints. Defaults to 1000.

    Returns:
        dataframe with raw data.
    """
    np.random.seed(42)
    x = np.linspace(0,2500, n)
    noise_component = np.random.rand(n)
    y = (x + x*noise_component/3)
    return pd.DataFrame({'x': x, 'y': y})

def analyse_data(raw_data: pd.DataFrame) -> pd.DataFrame:
    """Produce and analyse raw data.

       Function fits 16th and 84th percentiles.

    Args:
        raw_data: a dataframe containing the raw data.

    Returns:
        pd.DataFrame containing fit results.
    """
    pct16 = []
    pct84 = []
    x_mean = []
    bins = np.arange(raw_data['x'].min(),raw_data['x'].max(), 100)
    for k in range(len(bins) -1):
        idx = (raw_data['x'] >= bins[k]) & (raw_data['x'] < bins[k+1])
        pct16.append(np.percentile(raw_data['y'][idx],16))
        pct84.append(np.percentile(raw_data['y'][idx],84))
        x_mean.append(np.mean(raw_data['x'][idx]))
    return pd.DataFrame({'pct16': pct16, 'pct84': pct84, 'x_mean': x_mean})

def plot_analysis(raw_data: pd.DataFrame,
                  fit_results: pd.DataFrame) -> None:
    """Plot the results of the analysis.

    Args:
        raw_data: a dataframe containing the raw data.
        fit_results: a dataframe containing the results of the analysis.
    """
    ax = plt.subplot(111)
    ax.set_axisbelow(True)
    ax.scatter(raw_data['x'], raw_data['y'], label = 'raw data', color = 'grey', alpha = 0.5)
    ax.plot(fit_results['x_mean'], fit_results['pct16'], label = '16th percentile')
    ax.plot(fit_results['x_mean'], fit_results['pct84'], label = '84th percentile')
    ax.legend(frameon = False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_title('A Mock Scientific Result')
    ax.set_xlabel('x-variable [arb.]', size = 14)
    ax.set_ylabel('y-variable [arb.]', size = 14)
    ax.grid(True)


def test_data_generator():
    df1 = generate_data()
    df2 = pd.read_parquet(TEST_DATA_DIR+ '\\raw_data.parquet')
    assert df1.equals(df2)

def test_analyse_data():
    df1 = analyse_data(generate_data())
    df2 = pd.read_parquet(TEST_DATA_DIR+ '\\fit_results.parquet')
    assert df1.equals(df2)

# def test_full_analysis():
#     raw_data = generate_data()
#     fit_results = analyse_data(raw_data)
#     plot_analysis(raw_data = raw_data, fit_results = fit_results)

test_data_generator()
test_analyse_data()
test_full_analysis()