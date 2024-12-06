from IPython.display import display, HTML

display(HTML("<style>.container { width:95% !important; }</style>"))
import glob, os
import numpy as np
import networkx as nx
from networkx.utils import py_random_state
import random
import scipy as sp
from scipy.stats import ranksums, ttest_ind

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from matplotlib import ticker, cm

np.set_printoptions(precision=3, linewidth=200, suppress=True)
sns.set_context("poster")
sns.set_style("ticks")


### interpret the fname
def decode_fname(string, keywords):
    strings = string.split('_')
    results = [-1 for _ in keywords]
    for i, string in enumerate(strings):
        for idx, keyword in enumerate(keywords):
            if keyword == '':
                if i == 3:
                    try:
                        results[idx] = string
                    except:
                        continue
            elif keyword in string and keyword[0] == string[0]:
                results[idx] = string.replace(keyword, '')
                try:
                    results[idx] = float(results[idx])
                except:
                    continue
    return results


### collect raw data from a batch of experiments (e.g., 3942 experiments)
def collect_data(task_name, base_dir, pattern, epoch, trainseed_num, seed_start):
    save_dir = '/Users/khoapham/msu/graph2nn/analysis/final_results/{}'.format(task_name)
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    for epoch_cur in range(0, epoch + 1):
        print('{}:epoch{}'.format(task_name, epoch_cur))
        os.chdir(base_dir)
        names_all = []
        results_all = []
        for current_dir in glob.glob("*{}*".format(pattern)):
            #     print(current_dir)
            os.chdir('{}/{}'.format(base_dir, current_dir))
            fname = '{}/{}/results_epoch{}.txt'.format(base_dir, current_dir, epoch_cur)
            # if os.path.isfile(fname) and 'starttrainseed{}'.format(seed_start) in fname
            if os.path.isfile(fname):

                result = np.loadtxt(fname, delimiter=',')
                if len(result.shape) == 1:
                    result = result[np.newaxis, :]
                if result.shape[0] > trainseed_num:
                    result = result[-trainseed_num:]
                if result.shape == (trainseed_num, 8):
                    names_all.append(decode_fname(current_dir, task_keywords))
                    results_all.append(result)
        names_all = np.stack(names_all, axis=0)
        results_all = np.stack(results_all, axis=0)

        np.save(
            '/Users/khoapham/msu/graph2nn/analysis/final_results/{}/names_{}_epoch{}.npy'.format(task_name, task_name,
                                                                                                 epoch_cur), names_all)
        np.save(
            '/Users/khoapham/msu/graph2nn/analysis/final_results/{}/results_{}_epoch{}.npy'.format(task_name, task_name,
                                                                                                   epoch_cur),
            results_all)


### align graph_configs and experiment results
def align_results(names_all, results_all, trainseed_num, baseline_transform, graph_num, node_num=64,
                  return_sweet_spot=False):
    results_mean = np.mean(results_all, axis=1)
    results_std = np.std(results_all, axis=1)
    top1_mean = results_mean[:, 3]
    top1_std = results_std[:, 3]

    # Fix baseline handling
    mask = names_all[:, 0] == baseline_transform
    if np.any(mask):
        top1_baseline_mean = float(top1_mean[mask][0])
        top1_baseline_std = float(top1_std[mask][0])
    else:
        print("Baseline not found")
        top1_baseline_mean, top1_baseline_std = 0.0, 0.0

    order = np.argsort(top1_mean)
    best = np.argmin(top1_mean) if 'linear' not in baseline_transform else order[2]

    print('baseline', round(top1_baseline_mean, 2), round(top1_baseline_std, 2))
    print('best', round(top1_mean[best], 2), round(top1_std[best], 2))
    print('top 10\n', np.stack((top1_mean, top1_std), axis=1)[order[:10]].round(2))

    # Compute significant differences
    top1_best = results_all[best, :, 3]
    significant = []
    for i in range(results_all.shape[0]):
        stats, p = ttest_ind(top1_best, results_all[i, :, 3])
        significant.append(stats >= 0 or p / 2 >= 0.05)

    # Process graph configurations
    graph_configs = np.load(f'/Users/khoapham/msu/graph2nn/analysis/graphs_n{node_num}_{graph_num}.npy')
    graph_configs = graph_configs[graph_configs[:, 1] < 1]  # Exclude complete graph
    result_final = []
    significant_new = []

    for i in range(graph_configs.shape[0]):
        key = [str(round(graph_configs[i, 1], 6)), str(round(graph_configs[i, 2], 6)), 'sum',
               str(round(graph_configs[i, 3], 0))]
        loc = np.where((names_all[:, 4:8] == key).all(axis=1))[0]

        if len(loc) > 0:
            temp = np.zeros(graph_configs.shape[1] + 2)
            temp[:graph_configs.shape[1]] = graph_configs[i, :]
            temp[graph_configs.shape[1]:] = [top1_mean[loc[0]], top1_std[loc[0]]]  # Fix array handling
            result_final.append(temp)
            significant_new.append(significant[loc[0]])
        else:
            print('Not found:', key)

    result_temp = np.stack(result_final, axis=0)[significant_new]
    print('Sweet spot clustering min:', result_temp[:, 4].min(), 'max:', result_temp[:, 4].max())
    print('Sweet spot path min:', result_temp[:, 5].min(), 'max:', result_temp[:, 5].max())

    # Sweet spot calculation
    sweet_spot = [result_temp[:, 4].min(), result_temp[:, 5].min(),
                  result_temp[:, 4].max() - result_temp[:, 4].min(),
                  result_temp[:, 5].max() - result_temp[:, 5].min()]

    if return_sweet_spot:
        return result_final, sweet_spot
    else:
        return result_final

### align graph_configs and experiment results for each bin
# used when multiple graphs present in a bin
def align_results_bin(names_all, results_all, trainseed_num, baseline_transform, graph_num, node_num=64, return_sweet_spot=False):
    ### View results over different number of graphs
    graph_configs = np.load(f'/Users/khoapham/msu/graph2nn/analysis/graphs_n{node_num}_{graph_num}.npy')
    graph_configs = graph_configs[graph_configs[:, 1] < 1]  # Exclude complete graph

    result_final = []
    graph_configs_missing = []
    significant_new = []

    for i in range(graph_configs.shape[0]):
        key = [
            str(float(round(graph_configs[i, 1], 6))),
            str(float(round(graph_configs[i, 2], 6))),
            'sum',
            str(round(graph_configs[i, 3], 0))
        ]
        loc = np.where((names_all[:, 4:8] == key).all(axis=1))[0]

        if len(loc) > 0:
            for j in range(results_all.shape[1]):
                temp = np.zeros(graph_configs.shape[1] + 2)
                temp[:graph_configs.shape[1]] = graph_configs[i, :]
                temp[graph_configs.shape[1]:] = [np.mean(results_all[loc, j, 3]), 1]  # Safely assign mean value
                result_final.append(temp)
        else:
            print('Not found:', key, 'order:', i)
            graph_configs_missing.append(i)

    try:
        baseline_id = names_all[:, 0] == baseline_transform
    except:
        print('Baseline not found')
        baseline_id = None

    if baseline_id is not None:
        result_baseline = results_all[baseline_id[0]]
        for i in range(result_baseline.shape[0]):
            result_final.append(np.array([node_num, 1, 0, 0, 1, 1, result_baseline[i, 3], 1]))

    result_final = np.array(result_final)  # Convert to NumPy array

    # Bin-related logic
    bin_num = 52
    result_plot = result_final

    if bin_num == 52:
        bins_clustering = np.linspace(0, 1, 15 + 1)  # Clustering
        bins_path = np.linspace(1, 4.5, 15 + 1)  # Path
        bins_sparsity = np.linspace(0, 1, 15 + 1)  # Sparsity
        graph_configs_52 = np.load(f'/Users/khoapham/msu/graph2nn/analysis/graphs_n64_52.npy')
        graph_configs_52 = np.concatenate((graph_configs_52, [[-1, -1, -1, -1, 1, 1]]))

    # Path, clustering
    digits_clustering = np.digitize(result_plot[:, 4], bins_clustering, right=True)
    digits_path = np.digitize(result_plot[:, 5], bins_path)

    result_sum = np.zeros((len(bins_path) + 1, len(bins_clustering) + 1))
    result_count = np.zeros((len(bins_path) + 1, len(bins_clustering) + 1))

    for i in range(digits_clustering.shape[0]):
        result_sum[digits_path[i], digits_clustering[i]] += result_plot[i, -2]
        result_count[digits_path[i], digits_clustering[i]] += 1

    # Avoid divide by zero
    result_mean = np.divide(result_sum, result_count, where=result_count != 0)
    result_mean[result_count == 0] = np.nan  # Explicitly set empty bins to NaN

    if return_sweet_spot:
        return result_final, result_mean
    else:
        return result_final



def sort_results(names_all, results_all):
    rank_base = [names_all[:, i].astype(str) for i in range(names_all.shape[1])][::-1]
    order = np.lexsort(rank_base)
    names_all = names_all[order]
    results_all = results_all[order]
    return names_all, results_all


def compute_count(channel, group):
    divide = channel // group
    remain = channel % group

    out = np.zeros(group, dtype=int)
    out[:remain] = divide + 1
    out[remain:] = divide
    return out


@py_random_state(3)
def ws_graph(n, k, p, seed=1):
    """Returns a ws-flex graph, k can be real number in [2,n]
    """
    assert k >= 2 and k <= n
    # compute number of edges:
    edge_num = int(round(k * n / 2))
    count = compute_count(edge_num, n)
    G = nx.Graph()
    for i in range(n):
        source = [i] * count[i]
        target = range(i + 1, i + count[i] + 1)
        target = [node % n for node in target]
        G.add_edges_from(zip(source, target))
    # rewire edges from each node
    nodes = list(G.nodes())
    for i in range(n):
        u = i
        target = range(i + 1, i + count[i] + 1)
        target = [node % n for node in target]
        for v in target:
            if seed.random() < p:
                w = seed.choice(nodes)
                # Enforce no self-loops or multiple edges
                while w == u or G.has_edge(u, w):
                    w = seed.choice(nodes)
                    if G.degree(u) >= n - 1:
                        break  # skip this rewiring
                else:
                    G.remove_edge(u, v)
                    G.add_edge(u, w)
    return G


@py_random_state(4)
def connected_ws_graph(n, k, p, tries=100, seed=1):
    """Returns a connected ws-flex graph.
    """
    for i in range(tries):
        # seed is an RNG so should change sequence each call
        G = ws_graph(n, k, p, seed)
        if nx.is_connected(G):
            return G
    raise nx.NetworkXError('Maximum number of tries exceeded')


def generate_graph(message_type='ws', n=16, sparsity=0.5, p=0.2,
                   directed=False, seed=123):
    degree = n * sparsity
    if message_type == 'ws':
        graph = connected_ws_graph(n=n, k=degree, p=p, seed=seed)
    return graph


def extend_bin(bin):
    step = bin[1] - bin[0]
    bin = np.insert(bin, 0, bin[0] - step)
    bin = np.append(bin, bin[-1] + step)
    return bin


### 2-D graph stats vs. NN performance
def plot_2d(result_final, task_name, bin_num, simple=False, mark_best=True, sweet_spot=None, save_npy=False):
    cbar_label = 'Top-1 Error (%)'
    result_plot = np.array(result_final)  # Ensure result_final is a NumPy array

    ### Node = 64
    # Bin definitions
    if bin_num == 3942:
        bins_clustering = np.linspace(0, 1, 15 * 9 + 1)
        bins_path = np.linspace(1, 4.5, 15 * 9 + 1)
    elif bin_num == 449:
        bins_clustering = np.linspace(0, 1, 15 * 3 + 1)
        bins_path = np.linspace(1, 4.5, 15 * 3 + 1)
    elif bin_num == 52:
        bins_clustering = np.linspace(0, 1, 15 + 1)
        bins_path = np.linspace(1, 4.5, 15 + 1)

    # Clustering: Column 4; Path: Column 5
    digits_clustering = np.digitize(result_plot[:, 4], bins_clustering, right=True)
    digits_path = np.digitize(result_plot[:, 5], bins_path)

    result_sum = np.zeros((len(bins_path) + 1, len(bins_clustering) + 1))
    result_count = np.zeros((len(bins_path) + 1, len(bins_clustering) + 1))

    for i in range(digits_clustering.shape[0]):
        result_sum[digits_path[i], digits_clustering[i]] += result_plot[i, -2]
        result_count[digits_path[i], digits_clustering[i]] += 1

    # Safely handle division by zero
    result_mean = np.zeros_like(result_sum)
    nonzero_mask = result_count != 0
    result_mean[nonzero_mask] = result_sum[nonzero_mask] / result_count[nonzero_mask]
    result_mean[~nonzero_mask] = np.nan  # Set invalid bins to NaN

    ### Plotting
    sns.set_context("poster")
    fig = plt.figure(figsize=(15, 20))
    ax = fig.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # Heatmap
    surf = ax.pcolorfast(
        extend_bin(bins_clustering), extend_bin(bins_path), result_mean,
        cmap=plt.cm.Blues_r,
        vmin=np.nanmin(result_mean), vmax=np.nanmax(result_mean)
    )

    if mark_best and sweet_spot is not None and len(sweet_spot) == 4:
        rect = patches.Rectangle(
            (float(sweet_spot[0]), float(sweet_spot[1])),
            float(sweet_spot[2]),
            float(sweet_spot[3]),
            linewidth=5, edgecolor='#b02318', facecolor='none', linestyle='--'
        )
        ax.add_patch(rect)

    if simple:
        plt.xlabel('C', fontsize=60)
        plt.ylabel('L', fontsize=60)
        plt.xticks(fontsize=56)
        plt.yticks([1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5], fontsize=56)
    else:
        plt.xlabel('Clustering Coefficient (C)', fontsize=48)
        plt.ylabel('Average Path Length (L)', fontsize=48)
        plt.xticks(fontsize=40)
        plt.yticks([1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5], fontsize=40)

    cbar = fig.colorbar(surf, shrink=0.8, aspect=16, pad=0.15, orientation='horizontal')
    tick_locator = ticker.MaxNLocator(nbins=4)
    cbar.locator = tick_locator
    cbar.update_ticks()

    cbar.set_label(cbar_label, rotation=0, labelpad=10, fontsize=48)
    cbar.ax.tick_params(labelsize=40)

    plt.savefig(f'/Users/khoapham/msu/graph2nn/analysis/final_results/figs/{task_name}_resolution{bin_num}.png', dpi=125, bbox_inches='tight')

    if save_npy:
        np.save(f'/Users/khoapham/msu/graph2nn/analysis/final_results/npys/{task_name}_resolution{bin_num}.npy', result_mean[~np.isnan(result_mean)])


### 1-D slice of graph stats vs. NN performance
def plot_1d_slice(result_final, task_name, title_name, more_stats=None):
    # Ensure result_final is a NumPy array
    result_final = np.array(result_final)

    sns.set_context("poster")
    sns.set_style("ticks")

    current_palette = sns.color_palette('muted', n_colors=5)
    title = '{}'.format(title_name)

    point_color = '#a9a9a9'
    line_color = current_palette[0]
    line_width = 10
    point_size = 100

    # Fix range of clustering
    result_plot = result_final[np.logical_and(result_final[:, 4] >= 0.4, result_final[:, 4] <= 0.6)]

    fig = plt.figure(figsize=(16, 15))
    ax = plt.gca()
    if 'resnet' in task_name:
        ax.set_xlim(1.25, 3.1)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.1f'))

    # Correct regplot call
    sns.regplot(
        x=result_plot[:, 5],  # Path length
        y=result_plot[:, -2],  # Top-1 error
        order=2,  # Polynomial regression
        scatter_kws={'s': point_size, 'color': point_color},  # Scatter points
        line_kws={'linewidth': line_width, 'color': line_color},  # Regression line
        ci=None,  # Disable confidence interval
        truncate=False  # Show full regression line
    )

    plt.xlabel('L', fontsize=80)
    plt.ylabel('Top-1 Error', fontsize=80)
    plt.xticks([1.0, 1.5, 2, 2.5, 3, 3.5], fontsize=72)
    plt.yticks(fontsize=72)
    plt.savefig(f'/Users/khoapham/msu/graph2nn/analysis/final_results/figs/{task_name}_1dslice_path.png', dpi=75, bbox_inches='tight')
    plt.show()

    # Fix range of path
    result_plot = result_final[np.logical_and(result_final[:, 5] >= 2, result_final[:, 5] <= 2.5)]

    fig = plt.figure(figsize=(16, 15))
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    sns.regplot(
        x=result_plot[:, 4],  # Clustering coefficient
        y=result_plot[:, -2],  # Top-1 error
        order=2,
        scatter_kws={'s': point_size, 'color': point_color},
        line_kws={'linewidth': line_width, 'color': line_color},
        ci=None,
        truncate=False
    )

    plt.xlabel('C', fontsize=80)
    plt.ylabel('Top-1 Error', fontsize=80)
    plt.xticks([0, 0.2, 0.4, 0.6, 0.8], fontsize=72)
    plt.yticks(fontsize=72)
    plt.savefig(f'/Users/khoapham/msu/graph2nn/analysis/final_results/figs/{task_name}_1dslice_cluster.png', dpi=75, bbox_inches='tight')
    plt.show()



### Only run this block if you want to analyze raw data
### SKIP this block if you want to analyze with provided data
comment = 'v1'
task_keywords = ['trans','talkmode','num',
    'message','sparsity','p', 'agg',
    'graphseed','starttrainseed','endtrainseed','keep',
    'add1x1','upper','match','epoch'
    ]

### Example: loading experimental results from mlp_cifar10
task_name = 'vit_cifar10'
base_dir = "/Users/khoapham/msu/graph2nn/analysis/checkpoint/cifar10/vit_cifar10/best" # path to experiment results
pattern = 'vit_bs128_1gpu'
epoch = 100
trainseed_num = 1
seed_start = 1
collect_data(task_name, base_dir, pattern, epoch, trainseed_num, seed_start)


# -------------------
### Plots for MLP on CIFAR10
task_name = 'mlp_cifar10'
title_name = '5-layer 512-d MLP on Cifar-10'
trainseed_num = 5
epoch = 200
baseline_transform = 'linear'
graph_num = 3942

names_all = np.load('/Users/khoapham/msu/graph2nn/analysis/final_results/{}/names_{}_epoch{}.npy'.format(task_name, task_name, epoch))
results_all = np.load('/Users/khoapham/msu/graph2nn/analysis/final_results/{}/results_{}_epoch{}.npy'.format(task_name, task_name, epoch))
print(task_name, names_all.shape, results_all.shape)

result_final = align_results(names_all, results_all, trainseed_num, baseline_transform, graph_num)
plot_1d_slice(result_final, task_name, title_name)
bin_range = [i for i in [3942] if i <= graph_num]
for bin_num in bin_range:
    plot_2d(result_final, task_name, bin_num, simple=True)

result_final, sweet_spot = align_results_bin(names_all, results_all, trainseed_num, baseline_transform, graph_num,
                                             return_sweet_spot=True)
bin_range = [i for i in [52] if i <= graph_num]
for bin_num in bin_range:
    plot_2d(result_final, task_name, bin_num, simple=True, sweet_spot=sweet_spot)