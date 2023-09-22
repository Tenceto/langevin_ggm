import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
from scipy.optimize import curve_fit

from ggm_estimation.utils import _lambda_generic

pd.options.mode.chained_assignment = None  # default='warn'

def compute_estimation_performance(filename, tuneable_methods, fixed_methods, train_size, threshold_grid, metric, col_x_axis):
    df = pd.read_csv(filename, sep=";", index_col=0).reset_index(drop=True)
    # df["seed"] = np.repeat(np.arange(len(df) / 4), 4)

    if metric == "accuracy":
        metric_fun = accuracy_score
    elif metric == "f1":
        metric_fun = f1_score
    elif metric == "auc":
        metric_fun = roc_auc_score
    else:
        return None

    for col in ["real_values"] + [f"pred_{method}" for method in tuneable_methods + fixed_methods]:
        df[col] = df[col].apply(lambda x: np.array(eval(x)))

    train_seeds = np.random.choice(df["seed"].unique(), int(train_size * len(df) / df[col_x_axis].nunique()), replace=False)
    df_train = df[df["seed"].isin(train_seeds)]
    df_test = df[~ df["seed"].isin(train_seeds)]
    print("Training samples:", df_train["seed"].nunique())
    print("Test samples:", df_test["seed"].nunique())

    final_scores = dict()
    for method in tuneable_methods:
        scores_per_threshold = []
        for th in threshold_grid:
            df_train[metric] = df_train.apply(lambda row: metric_fun(row["real_values"], row[f"pred_{method}"] >= th), axis=1)
            scores = df_train.groupby(col_x_axis)[metric].mean()
            scores_per_threshold.append(scores.to_dict())

        # Convert the list of dictionaries to a dictionary of lists
        scores_per_threshold = {k: [dic[k] for dic in scores_per_threshold] for k in scores_per_threshold[0]}
        # Get the best threshold for each case
        best_thresholds = {k: threshold_grid[np.argmax(v)] for k, v in scores_per_threshold.items()}

        df_test[metric] = df_test.apply(lambda row: metric_fun(row["real_values"], row[f"pred_{method}"] >= best_thresholds[row["num_obs"]]), axis=1)
        final_scores[method] = df_test.groupby(col_x_axis)[metric].mean().to_dict()

    for method in fixed_methods:
        df_test[metric] = df_test.apply(lambda row: metric_fun(row["real_values"], row[f"pred_{method}"]), axis=1)
        final_scores[method] = df_test.groupby(col_x_axis)[metric].mean().to_dict()
    
    return final_scores, best_thresholds


def plot_results(accuracy_dict, labels, title="", output_file=None, colors=None, 
                 linestyles=None, xlabel=r"Number of observations $k$", ylabel="Edge prediction accuracy", logscale=True,
                 legend_loc="best", ylims=None, legend_ncol=1):
    if linestyles is None:
        linestyles = {method: "-" for method in accuracy_dict.keys()}
    accuracy_dict = {k: accuracy_dict[k] for k in labels.keys()}
    for method in accuracy_dict.keys():
        if colors is not None:
            plt.plot(list(accuracy_dict[method].keys()), list(accuracy_dict[method].values()),
                     marker="o", label=labels[method], color=colors[method], linestyle=linestyles[method])
        else:
            plt.plot(list(accuracy_dict[method].keys()), list(accuracy_dict[method].values()),
                     marker="o", label=labels[method], linestyle=linestyles[method])
    if ylims is not None:
        plt.ylim(ylims)
    plt.legend(loc=legend_loc, ncol=legend_ncol)
    if logscale:
        plt.xscale("log")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid()
    if output_file is not None:
        plt.savefig(output_file, bbox_inches="tight")
    else:
        plt.show()


def plot_glasso_tuning(filename):
    tuning_data = pd.read_csv(filename, sep=";").groupby(["num_obs", "lambda"])["metric"].mean().to_frame()
    num_obs_list = tuning_data.index.get_level_values("num_obs").unique()
    max_per_num_obs = tuning_data.groupby(level=0)['metric'].idxmax().apply(lambda x: x[1])

    popt, _ = curve_fit(_lambda_generic, max_per_num_obs.index, max_per_num_obs.values)

    _, axs = plt.subplots(1, 2, figsize=(12, 4))
    for num_obs in num_obs_list:
        axs[0].plot(tuning_data.loc[num_obs]["metric"].index, tuning_data.loc[num_obs]["metric"].values, label=rf"$k = {num_obs}$")
    axs[1].plot(max_per_num_obs.index, max_per_num_obs.values, marker="o", label="Data")
    axs[1].plot(num_obs_list, _lambda_generic(num_obs_list, *popt), label="New fit")
    # axs[1].plot(num_obs_list, lambda_glasso_selector(graph_type, nans, n_proportional, one_zero_ratio)(num_obs_list).values, "--", label="Stored fit")
    axs[0].set_xscale("log")
    axs[0].set_xlabel(r"$\lambda$")
    axs[0].set_ylabel(f"Metric")
    axs[0].legend()
    axs[1].set_xscale("log")
    axs[1].set_xlabel(r"$k$")
    axs[1].set_ylabel(r"Optimal $\lambda$")
    axs[1].legend()
    print(popt)
    plt.show()