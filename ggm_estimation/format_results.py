import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score

def compute_accuracy_best_threshold(results_file, methods, remove_num_obs=[], grid_size=50, 
                                    col_x_axis="num_obs", metric="accuracy"):
    df = pd.read_csv(results_file, sep=";", index_col=0).reset_index(drop=True)
    df = df[~ df[col_x_axis].isin(remove_num_obs)]
    # df = df.iloc[:100]

    print(len(df))

    if metric == "accuracy":
        metric_fun = accuracy_score
    elif metric == "f1":
        metric_fun = f1_score
    elif metric == "auc":
        metric_fun = roc_auc_score

    threshold_grid = np.linspace(0.0, 1.0, grid_size)
    all_thresholds = dict()
    all_accuracies = dict()

    for col in ["real_values"] + [f"pred_{method}" for method in methods]:
        df[col] = df[col].apply(lambda x: np.array(eval(x)))
    
    for method in methods:
        accuracy_per_threshold = list()
        for th in threshold_grid:
            # df["accuracy"] = df.apply(lambda row: np.mean((row[f"pred_{method}"] >= th) == row["real_values"]), axis=1)
            df["accuracy"] = df.apply(lambda row: metric_fun(row["real_values"], row[f"pred_{method}"] >= th), axis=1)
            accuracies = df.groupby(col_x_axis)["accuracy"].mean()
            accuracy_per_threshold.append(accuracies.to_dict())
        accuracy_per_threshold = {k: [dic[k] for dic in accuracy_per_threshold] for k in accuracy_per_threshold[0]}

        thresholds_per_num_obs = dict()
        accuracies_per_num_obs = dict()
        for _, (num_obs, accuracies) in enumerate(accuracy_per_threshold.items()):
            best_th = threshold_grid[np.argmax(accuracies)]
            thresholds_per_num_obs.update({num_obs: best_th})
            accuracies_per_num_obs.update({num_obs: np.max(accuracies)})
        all_thresholds.update({method: thresholds_per_num_obs})
        all_accuracies.update({method: accuracies_per_num_obs})
    
    return all_thresholds, all_accuracies


def compute_accuracy_no_threshold(results_file, methods, remove_num_obs=[]):
    df = pd.read_csv(results_file, sep=";", index_col=0).reset_index(drop=True)
    df = df[~ df["num_obs"].isin(remove_num_obs)]
    all_accuracies = dict()

    for col in ["real_values"] + [f"pred_{method}" for method in methods]:
        df[col] = df[col].apply(lambda x: np.array(eval(x)))
    
    for method in methods:
        df["accuracy"] = df.apply(lambda row: np.mean((row[f"pred_{method}"]) == row["real_values"]), axis=1)
        accuracies = df.groupby("num_obs")["accuracy"].mean()
        all_accuracies.update({method: accuracies.to_dict()})
    
    return all_accuracies


def plot_results(accuracy_dict, labels, title, output_file=None, colors=None, 
                 linestyles=None, xlabel=r"Number of observations $k$", ylabel="Edge prediction accuracy", logscale=True,
                 legend_loc="best", ylims=None, legend_ncol=1):
    if linestyles is None:
        linestyles = {method: "-" for method in accuracy_dict.keys()}
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