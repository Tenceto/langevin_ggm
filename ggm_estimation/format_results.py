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


def transform_results(results, methods):
    results["real_values"] = results["real_values"].apply(lambda x: np.array(eval(x)))
    for method in methods:
        results[f"pred_{method}"] = results[f"pred_{method}"].apply(lambda x: np.array(eval(x)))
        results[f"matched_{method}"] = results.apply(lambda row: np.mean(row[f"pred_{method}"] == row["real_values"]), axis=1)
        results[f"f1_{method}"] = results.apply(lambda row: f1_score(row[f"pred_{method}"], row["real_values"]), axis=1)

    matched_edges = results[[f"matched_{method}" for method in methods]].mean()
    matched_edges.index = [s[len("matched_"):] for s in matched_edges.index]
    matched_edges.name = "matched_edges_mean"

    f1 = results[[f"f1_{method}" for method in methods]].mean()
    f1.index = [s[len("f1_"):] for s in f1.index]
    f1.name = "f1_score_mean"

    matched_edges_std = results[[f"matched_{method}" for method in methods]].std()
    matched_edges_std.index = [s[len("matched_"):] for s in matched_edges_std.index]
    matched_edges_std.name = "matched_edges_std"

    f1_std = results[[f"f1_{method}" for method in methods]].std()
    f1_std.index = [s[len("f1_"):] for s in f1.index]
    f1_std.name = "f1_score_std"

    grouped_results = pd.merge(matched_edges, f1, left_index=True, right_index=True, how="outer")
    # grouped_results = pd.merge(grouped_results, max_posterior, left_index=True, right_index=True, how="outer")
    grouped_results = pd.merge(grouped_results, matched_edges_std, left_index=True, right_index=True, how="outer")
    grouped_results = pd.merge(grouped_results, f1_std, left_index=True, right_index=True, how="outer")

    try:
        results["best_likelihood"] = results[[f"likelihood_{method}" for method in methods]].idxmax(axis=1)
        
        likelihoods = results[[f"likelihood_{method}" for method in methods]].mean()
        likelihoods.index = [s[len("likelihood_"):] for s in likelihoods.index]
        likelihoods.name = "likelihoods_mean"

        max_posterior = results[[f"likelihood_{method}" for method in methods]].idxmax(axis=1).value_counts() / len(results)
        max_posterior.index = [s[len("likelihood_"):] for s in max_posterior.index]
        max_posterior.name = "max_posterior"

        grouped_results = pd.merge(grouped_results, likelihoods, left_index=True, right_index=True, how="outer")
    except:
        pass
    # grouped_results = grouped_results.fillna(0.0)

    return grouped_results

def show_results_table(grouped_results):
    cm = mpl.colormaps["RdYlGn"]
    cm_reversed = cm.reversed()

    f1_range = (grouped_results['f1_score'].min(), grouped_results['f1_score'].max())
    matched_edges_range = (grouped_results['matched_edges'].min(), grouped_results['matched_edges'].max())
    max_posterior_range = (grouped_results['max_posterior'].min(), grouped_results['max_posterior'].max())

    return grouped_results.style.background_gradient(cmap=cm, subset=['matched_edges'], vmin=matched_edges_range[0], vmax=matched_edges_range[1])\
                                .background_gradient(cmap=cm_reversed, subset=['f1_score'], vmin=f1_range[0], vmax=f1_range[1])\
                                .background_gradient(cmap=cm_reversed, subset=['max_posterior'], vmin=max_posterior_range[0], vmax=max_posterior_range[1])