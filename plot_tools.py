from datetime import datetime
from itertools import islice, cycle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.dates import date2num


def run_all_plots_MLE_first_model(Z, X, Y, mle, nit, roots_file, prob_of_features, specific_information="", title=None):
    """
    The function plots a colorful scatter plot of p0t*qi vs kit/nit,a sized colorful scatter plot and p0t
    vs time. It also saves the plots as svg.
    :param Z: all kit/nit , which means prob from the data.
    :param X: list of all days measured.
    :param Y: list of all feature vectors.
    :param mle: performance of the class mle
    :param roots_file: the adress of the roots damp file which was created by MLE
    :param specific_information: words to add for the name of the svg plots
    :return: None
    """
    # calculate the mul of pot and qi
    roots = np.load(roots_file, allow_pickle=True)
    qi_series = mle.calculate_qi_based_on_MLE(roots)
    p0t_mul_qi_df = pd.DataFrame(1, index=qi_series.index, columns=X)
    p0t_mul_qi_df = p0t_mul_qi_df.multiply(qi_series, axis='index') * roots
    real_prob = ((1 - np.exp((-1) * p0t_mul_qi_df)).mul(prob_of_features, axis='index')).sum()
    real_prob_per_day_per_features = ((1 - np.exp((-1) * p0t_mul_qi_df)))
    # colorful scatter plot:
    fig = plt.figure(figsize=(30, 20))
    ax = fig.add_subplot(111)
    sc = ax.scatter(real_prob_per_day_per_features, Z, alpha=0.2,
                    c=list(islice(cycle([date2num(i) for i in X]), len(X) * len(Y))),
                    cmap=cm.jet)
    # X string to dates
    X_dates = [datetime.strptime(i, "%Y-%m-%d") for i in X]
    sm = cm.ScalarMappable(cmap=cm.jet)
    sm._A = []
    cbar = plt.colorbar(sm)
    # Change the numeric ticks into ones that match the x-axis
    cbar.ax.set_yticklabels([i.strftime(format='%Y-%m') for i in X_dates][0::30])
    ax.set_xlabel("1-e^(p0t * qi)")
    ax.set_ylabel("Kt / Nt")
    ax.set_title("scatter of Kt/Nt towards 1-e^(p0t*qi) Global Model")
    plt.savefig("plots/first_model_colorful_scatter" + specific_information + ".svg")
    plt.show()

    # scatter plot with size according to N_i_t:
    fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot(111)
    ax.scatter(p0t_mul_qi_df, Z, alpha=0.2, s=mle.N.unstack().T,
               c=list(islice(cycle([date2num(i) for i in X]), len(X) * len(Y))), cmap=cm.jet)
    # X string to dates
    X_dates = [datetime.strptime(i, "%Y-%m-%d") for i in X]
    # creating color map of dates:
    sm = cm.ScalarMappable(cmap=cm.jet)
    sm._A = []
    cbar = plt.colorbar(sm)
    # Change the numeric ticks into ones that match the x-axis
    cbar.ax.set_yticklabels([i.strftime(format='%Y-%m') for i in X_dates][0::30])
    ax.set_xlabel("p0t * qi")
    ax.set_ylabel("K_i_t / N_i_t")
    ax.set_title("Scatter of Ki_t/Ni_t towards p0t*qi")
    plt.xlim(0, 1)
    plt.savefig("plots/first_model_colorful_sized_scatter" + specific_information + ".svg")
    plt.show()

    # real prob with features sized scatter:
    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(111)
    ax.scatter(real_prob_per_day_per_features, Z, alpha=0.2, s=nit.unstack().T,
               c=list(islice(cycle([date2num(i) for i in X]), len(X) * len(Y))), cmap=cm.jet)
    # X string to dates
    X_dates = [datetime.strptime(i, "%Y-%m-%d") for i in X]
    # creating color map of dates:
    sm = cm.ScalarMappable(cmap=cm.jet)
    sm._A = []
    # cbar = plt.colorbar(sm)
    # Change the numeric ticks into ones that match the x-axis
    # cbar.ax.set_yticklabels([i.strftime(format='%Y-%m') for i in X_dates][0::30])
    ax.set_xlabel("POS(i,t)")  # "1-e^(p0t * qi)"
    ax.set_ylabel("Kt / Nt")
    ax.set_title("Scatter of Kt/Nt towards POS(i,t) Global Model")  # 1-e^(p0t*qi
    plt.xlim(0, 1)
    plt.savefig("fig for template/sized_scatter_0.1.png")
    plt.savefig("fig for template/sized_scatter_0.1.pdf")
    plt.savefig("fig for template/sized_scatter_0.1.svg")
    plt.savefig("fig for template/sized_scatter_0.1.eps")
    plt.show()

    # p0t roots plot:
    fig = plt.figure(figsize=(45, 12))
    ax = fig.add_subplot(111)
    ax.scatter(X, roots)
    # ax.plot(X, roots, marker='.')
    plt.xticks(rotation=45, fontsize=20)
    every_nth = 12
    for n, label in enumerate(ax.xaxis.get_ticklabels()):
        if n % every_nth != 0:
            label.set_visible(False)
    plt.yticks(fontsize=25)
    ax.margins(tight=True)
    ax.set_xlabel("time", fontsize=30)
    ax.set_ylabel("p0(t)", fontsize=30)
    ax.set_title("p0(t) vs time", fontsize=30)
    fig.savefig("plots/first_model_p0t_vs_time" + specific_information + ".svg")
    fig.show()

    # plot real prob vs time:
    fig = plt.figure(figsize=(45, 12))
    ax = fig.add_subplot(111)
    ax.scatter(real_prob.index, real_prob, c="darkviolet")
    # ax.plot(X, roots, marker='.')
    plt.xticks(rotation=45)
    every_nth = 12
    for n, label in enumerate(ax.xaxis.get_ticklabels()):
        if n % every_nth != 0:
            label.set_visible(False)
    plt.yticks()
    # plt.yscale("log")-only if with_p0t is true
    ax.margins(tight=True)
    if title is not None:
        ax.set_title(title, fontsize=30)
    else:
        ax.set_title("Probability to be Positive vs Time as Measured in Global Model", fontsize=35)
    ax.set_xlabel("Time", fontsize=30)
    ax.set_ylabel("Probability to be positive", fontsize=30)
    fig.tight_layout()
    fig.savefig("figures_for_article/global_vs_time_scatter_0.1.svg")
    fig.savefig("figures_for_article/global_vs_time_scatter_0.1.png")
    fig.show()

    # plot real prob vs time line:
    fig = plt.figure(figsize=(20, 20))  # (45, 12)
    ax = fig.add_subplot(111)
    ax.plot(real_prob.index, real_prob, c="darkviolet")
    # ax.plot(X, roots, marker='.')
    plt.xticks(rotation=45)
    every_nth = 12
    for n, label in enumerate(ax.xaxis.get_ticklabels()):
        if n % every_nth != 0:
            label.set_visible(False)
    plt.yticks()
    # plt.yscale("log")-only if with_p0t is true
    ax.margins(tight=True)
    if title is not None:
        ax.set_title(title)
    else:
        ax.set_title("Probability to be Positive vs Time as Measured in Global Model")
    ax.set_xlabel("Time", fontsize=30)
    ax.set_ylabel("Probability to be positive", fontsize=30)
    fig.tight_layout()
    fig.savefig("figures_for_article/global_vs_time_line_0.1.png")
    fig.savefig("figures_for_article/global_vs_time_line_0.1.svg")
    fig.show()

    # plot real prob vs current estimate:
    plot_ours_vs_current(real_prob, mle.N, mle.K)
    # plot ours vs avg estimate
    plot_average_fraction_and_average_num_vs_model_probs(real_prob, mle.N, mle.K)


def plot_ours_vs_current(prob_to_be_positive, N: pd.DataFrame, K: pd.DataFrame, title=None):
    K_div_N = K.unstack().T.sum() / N.unstack().T.sum()
    K_sum = K.unstack().T.sum()

    fig = plt.figure(figsize=(45, 12))
    ax = fig.add_subplot(111)
    ax.plot(K_div_N, alpha=0.5, c="b", label="Current estimate")
    ax.scatter(prob_to_be_positive.index, prob_to_be_positive, c="darkviolet", label="Our global estimate")
    plt.xticks(rotation=45, fontsize=20)
    every_nth = 12
    for n, label in enumerate(ax.xaxis.get_ticklabels()):
        if n % every_nth != 0:
            label.set_visible(False)
    plt.yticks()
    # plt.yscale("log")-only if with_p0t is true
    ax.margins(tight=True)
    if title is not None:
        plt.title(title)
    else:
        plt.title("Our Global Model Estimate vs Current Estimate")
    ax.set_xlabel("time")
    ax.set_ylabel("Daily prob to be positive")
    fig.legend()

    if title:
        fig.savefig("plots/static_vs_current" + title + ".svg")
    else:
        fig.savefig("plots/static_vs_current.svg")
    fig.show()


def plot_average_fraction_and_average_num_vs_model_probs(prob_to_be_positive, N: pd.DataFrame, K: pd.DataFrame,
                                                         from_day: int = 7):
    K_div_N = K.unstack().T.sum() / N.unstack().T.sum()
    K_sum = K.unstack().T.sum()
    total = K_div_N[:from_day - 1]
    total_K = K_sum[:from_day - 1]
    for i in range(from_day - 1, len(K_div_N)):
        tmp = K_div_N[i - 6:i + 1]
        total = total.append(pd.Series(tmp.mean(), index=[K_div_N.index[i]]))
        tmp = K_sum[i - 6:i + 1]
        total_K = total_K.append(pd.Series(tmp.mean(), index=[K_div_N.index[i]]))
    fig = plt.figure(figsize=(30, 30))
    ax = fig.add_subplot(111)
    ax.plot(total, alpha=0.7, label="Current 7 days average estimate")
    ax.scatter(prob_to_be_positive.index, prob_to_be_positive, c="darkviolet", label="Our global estimate")
    # order the xsticks:
    every_nth = 40
    for n, label in enumerate(ax.xaxis.get_ticklabels()):
        if n % every_nth != 0:
            label.set_visible(False)

    # plt.xticks(rotation=45)
    plt.ylabel("Probability to be positive", color="b", fontsize=45)
    ax.set_xlabel("Time", fontsize=45)
    plt.twinx()
    plt.plot(total_K, color="r", alpha=0.7)
    plt.ylabel("7 days ago number of positives", color="r", fontsize=45)
    plt.title("Our Global Model vs Current 7 Days Ago Average Estimate", fontsize=45)
    fig.tight_layout()
    fig.savefig("fig for template/current_vs_global_0.1.svg")
    fig.savefig("fig for template/current_vs_global_0.1.png")
    plt.show()
