import scipy as sc
from aethos.visualizations import visualize as viz


def run_2sample_ttest(
    group1: str, group2: str, train_data, t_type: str, output_file, **kwargs
):
    """
    Helper function to run 2 sample ttest.
    
    Parameters
    ----------
    group1 : str
        Column for group 1 to compare.
    
    group2 : str, optional
        Column for group 2 to compare, by default None

    train_data : DataFrame
        Train Data

    t_type : str
        T-test type

    output_file : str
        File name to output
    
    Returns
    -------
    list
        T test statistic, P value        
    """

    data_group1 = train_data[group1].tolist()
    data_group2 = train_data[group2].tolist()

    if t_type == "ind":
        results = sc.stats.ttest_ind(
            data_group1, data_group2, nan_policy="omit", **kwargs
        )
    else:
        results = sc.stats.ttest_rel(data_group1, data_group2, nan_policy="omit",)

    matrix = [
        ["", "Test Statistic", "p-value"],
        ["Sample Data", results[0], results[1]],
    ]

    viz.create_table(matrix, True, output_file)

    return results
