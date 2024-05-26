import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
from scipy.optimize import curve_fit
import matplotlib.ticker as tick

from utils.ggm_utils import lambda_generic

pd.options.mode.chained_assignment = None  # default='warn'

def compute_estimation_performance(filename, tuneable_methods, fixed_methods, train_size, threshold_grid, metric, col_x_axis, 
                                   n_splits=5, agg_fun="mean"):
    if metric == "accuracy":
        metric_fun = accuracy_score
    elif metric == "f1":
        metric_fun = f1_score
    elif metric == "auc":
        metric_fun = roc_auc_score
    else:
        return None

    df = pd.read_csv(filename, sep=";", index_col=0).reset_index(drop=True)

    for col in ["real_values"] + [f"pred_{method}" for method in tuneable_methods + fixed_methods]:
        df[col] = df[col].apply(lambda x: np.array(eval(x)))

    if len(tuneable_methods) != 0 and train_size is not None and threshold_grid is not None and n_splits is not None:
        seeds = df["seed"].unique()
        print("Samples:", len(seeds))
        final_scores = 0
        final_variances = 0

        for split in range(n_splits):
            train_seeds = np.random.choice(seeds, int(train_size * len(seeds)), replace=False)
            df_train = df[df["seed"].isin(train_seeds)]
            df_test = df[~ df["seed"].isin(train_seeds)]
            # print("Training samples:", df_train["seed"].nunique())
            # print("Test samples:", df_test["seed"].nunique())

            new_scores, new_variances = _select_threshold_validation_set(df_train, df_test, tuneable_methods, 
                                                                         threshold_grid, metric_fun, col_x_axis, 
                                                                         agg_fun)
            
            final_scores += new_scores
            final_variances += new_variances
        
        final_scores = (final_scores / n_splits).to_dict()
        final_stds = np.sqrt((final_variances / n_splits)).to_dict()
    
    if 'final_scores' not in locals():
        final_scores = dict()
    if 'final_stds' not in locals():
        final_stds = dict()

    for method in fixed_methods:
        df['metric'] = df.apply(lambda row: metric_fun(row["real_values"], row[f"pred_{method}"]), axis=1)
        if agg_fun == "mean":
            final_scores[method] = df.groupby(col_x_axis)['metric'].mean().to_dict()
        elif agg_fun == "median":
            final_scores[method] = df.groupby(col_x_axis)['metric'].median().to_dict()
        final_stds[method] = df.groupby(col_x_axis)['metric'].std().to_dict()
    
    return final_scores, final_stds


def _select_threshold_validation_set(df_train, df_test, tuneable_methods, threshold_grid, metric_fun, col_x_axis, agg_fun):
    final_scores = dict()
    final_variances = dict()
    for method in tuneable_methods:
        if metric_fun != roc_auc_score:
            scores_per_threshold = []
            for th in threshold_grid[method]:
                df_train["metric"] = df_train.apply(lambda row: metric_fun(row["real_values"], row[f"pred_{method}"] >= th), axis=1)
                if agg_fun == "mean":
                    scores = df_train.groupby(col_x_axis)['metric'].mean()
                elif agg_fun == "median":
                    scores = df_train.groupby(col_x_axis)['metric'].median()
                scores_per_threshold.append(scores.to_dict())

            # Convert the list of dictionaries to a dictionary of lists
            scores_per_threshold = {k: [dic[k] for dic in scores_per_threshold] for k in scores_per_threshold[0]}
            # Get the best threshold for each case
            best_thresholds = {k: threshold_grid[method][np.argmax(v)] for k, v in scores_per_threshold.items()}

            df_test['metric'] = df_test.apply(lambda row: metric_fun(row["real_values"], row[f"pred_{method}"] >= best_thresholds[row[col_x_axis]]), axis=1)
        else:
            df_test['metric'] = df_test.apply(lambda row: metric_fun(row["real_values"], row[f"pred_{method}"]), axis=1)
        
        if agg_fun == "mean":
            final_scores[method] = df_test.groupby(col_x_axis)['metric'].mean().to_dict()
        elif agg_fun == "median":
            final_scores[method] = df_test.groupby(col_x_axis)['metric'].median().to_dict()

        final_variances[method] = df_test.groupby(col_x_axis)['metric'].var().to_dict()

    return pd.DataFrame(final_scores), pd.DataFrame(final_variances)


def plot_results(accuracy_dict, stds_grids=None, labels=None, title="", output_file=None, colors=None,
                 linestyles=None, xlabel=r"$k$", ylabel="Edge prediction accuracy", logscale=True,
                 legend_loc="best", ylims=None, legend_ncol=1, marker_size=6, linewidth=1.5,
                 legend_out=False, marker_list=None):
    
    if marker_list is None:
        marker_list = ['o', 'v', '^', 's', 'p', 'P', '+', 'X', 'd']
    
    fig, ax = plt.subplots()
    if linestyles is None:
        linestyles = {method: "-" for method in accuracy_dict.keys()}
    accuracy_dict = {k: accuracy_dict[k] for k in labels.keys()}
    for i, method in enumerate(accuracy_dict.keys()):
        x_values = np.array(list(accuracy_dict[method].keys()))
        y_values = np.array(list(accuracy_dict[method].values()))
        if stds_grids is not None:
            std_values = np.array(list(stds_grids[method].values()))
        if colors is not None:
            ax.plot(x_values, y_values,
                     marker=marker_list[i], label=labels[method], 
                     color=colors[method], linestyle=linestyles[method],
                     markersize=marker_size, linewidth=linewidth)
            if stds_grids is not None:
                ax.fill_between(x_values, 
                                y_values - std_values, 
                                y_values + std_values, 
                                color=colors[method], alpha=0.3)
        else:
            ax.plot(x_values, y_values,
                     marker=marker_list[i], label=labels[method], linestyle=linestyles[method],
                     markersize=marker_size, linewidth=linewidth)
            if stds_grids is not None:
                ax.fill_between(x_values, 
                                y_values - std_values, 
                                y_values + std_values, 
                                alpha=0.3)
    if ylims is not None:
        ax.set_ylim(ylims)
    ax.legend(loc=legend_loc, ncol=legend_ncol)
    if logscale:
        ax.set_xscale("log")
        ax.get_xaxis().set_major_formatter(tick.ScalarFormatter())
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(list(accuracy_dict[method].keys()), labels=list(accuracy_dict[method].keys()))
    ax.set_xticks([], minor=True)
    # Add only horizontal gridlines
    ax.yaxis.grid(True)
    if legend_out:
        # Shrink current axis by 20%
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        # Put a legend to the right of the current axis
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    if output_file is not None:
        plt.savefig(output_file, bbox_inches="tight")
    else:
        plt.show()


def plot_glasso_tuning(all_results, figsize=(10, 4), old_fitted_coefs=None, output_file=None):
    all_results = all_results.groupby(["num_obs", "lambda"]).mean()
    num_obs_list = all_results.index.get_level_values("num_obs").unique()
    max_per_num_obs = all_results.groupby(level=0)['f1'].idxmax().apply(lambda x: x[1])
    popt, _ = curve_fit(lambda_generic, max_per_num_obs.index, max_per_num_obs.values)

    # Plot the results
    _, axs = plt.subplots(1, 2, figsize=figsize)
    for num_obs in num_obs_list:
        axs[0].plot(all_results.loc[num_obs]["f1"].index, all_results.loc[num_obs]["f1"].values, label=rf"$k = {num_obs}$")
    axs[1].plot(max_per_num_obs.index, max_per_num_obs.values, marker="o", label="Data", linestyle=None)
    axs[1].plot(num_obs_list, lambda_generic(num_obs_list, *popt), label="Fitted curve")
    if old_fitted_coefs is not None:
        axs[1].plot(num_obs_list, lambda_generic(num_obs_list, *old_fitted_coefs), label="Old fitted curve")
    axs[0].set_xscale("log")
    axs[0].set_xlabel(r"$\lambda$")
    axs[0].set_ylabel("F1 Score")
    axs[0].legend(loc="upper right")
    axs[0].grid()

    axs[1].set_xscale("log")
    axs[1].set_xlabel(r"$k$")
    axs[1].set_ylabel(r"Optimal $\lambda$")
    axs[1].legend()
    axs[1].grid()

    print("Optimal parameters:")
    print("a =", popt[0])
    print("b =", popt[1])
    print("c =", popt[2])

    plt.tight_layout()

    if output_file is not None:
        plt.savefig(output_file)
    else:
        plt.show()


def color_fader(c1, c2, mix=0):
    c1 = np.array(mpl.colors.to_rgb(c1))
    c2 = np.array(mpl.colors.to_rgb(c2))
    return mpl.colors.to_hex((1 - mix) * c1 + mix * c2)
