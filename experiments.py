import warnings
import multiprocessing
from multiprocessing import Process, Pool

import numpy as np

from network_attacks import instantaneous_attack, incremental_attack, incremental_random_failure, \
    instantaneous_random_failure
from networks import oregon
from utils.data_saver import save_results
from utils.network_generation import get_poisson_net, get_power_law_net
from utils.visualization import plot_clustering_distribution, plot_metric_distribution, save_network_tracking

warnings.filterwarnings("ignore", category=UserWarning)


def instantaneous_attack_powerlaw(exp_num_nodes, exp_removal_ratios, exp_ks, is_random_attack):
    for n_nodes in exp_num_nodes:
        for k in exp_ks:
            net = get_power_law_net(n_nodes, k, verbose=False)
            if not is_random_attack:
                min_path, max_path, cluster_size_ratios = instantaneous_attack(net=net,
                                                                               removal_rates=exp_removal_ratios,
                                                                               verbose=True)

                file_name = "powerlaw-inst-attack-k=%.2f-n_nodes=%d" % (k, n_nodes)
                title = "Instantaneous Attack - Powerlaw nodes=%d k=%.2f " % (n_nodes, k)
            else:
                min_path, max_path, cluster_size_ratios = instantaneous_random_failure(net=net,
                                                                                       removal_rates=exp_removal_ratios,
                                                                                       verbose=True)

                file_name = "powerlaw-inst-failure-k=%.2f-n_nodes=%d" % (k, n_nodes)
                title = "Instantaneous Failure - Powerlaw nodes=%d k=%.2f " % (n_nodes, k)

            save_results(min_path, max_path, cluster_size_ratios, file_name)
            plot_metric_distribution([min_path, max_path],
                                     exp_removal_ratios,
                                     y_label="path length",
                                     labels=[r'$d_{min}$', r'$d_{max}$'],
                                     title=title,
                                     filename=file_name + '_distr.png')
            clusters_info = [x[0] for x in cluster_size_ratios]
            plot_clustering_distribution(clusters_info,
                                         exp_removal_ratios,
                                         y_label="clusterings",
                                         labels=[r'$S$', r'$\langle s \rangle$'],
                                         title=title,
                                         filename=file_name + '_clust.png')


def instantaneous_attack_poisson(exp_num_nodes, exp_removal_ratios, exp_mus, is_random_attack):
    for n_nodes in exp_num_nodes:
        for mu in exp_mus:
            net = get_poisson_net(n_nodes=n_nodes, mu=mu, verbose=True)

            if not is_random_attack:
                min_path, max_path, cluster_size_ratios = instantaneous_attack(net=net,
                                                                               removal_rates=exp_removal_ratios,
                                                                               verbose=True)

                file_name = "poisson-inst-attack-mu=%.2f-n_nodes=%d" % (mu, n_nodes)
                title = "Instantaneous Attack - Poisson nodes=%d mu=%.2f " % (n_nodes, mu)

            else:
                min_path, max_path, cluster_size_ratios = instantaneous_random_failure(net=net,
                                                                                       removal_rates=exp_removal_ratios,
                                                                                       verbose=True)

                file_name = "poisson-inst-failure-mu=%.2f-n_nodes=%d" % (mu, n_nodes)
                title = "Instantaneous Failure - Poisson nodes=%d mu=%.2f " % (n_nodes, mu)
            save_results(min_path, max_path, cluster_size_ratios, file_name)

            plot_metric_distribution([min_path, max_path],
                                     exp_removal_ratios,
                                     y_label="path length",
                                     labels=[r'$d_{min}$', r'$d_{max}$'],
                                     title=title,
                                     filename=file_name + '_distr.png')
            clusters_info = [x[0] for x in cluster_size_ratios]
            plot_clustering_distribution(clusters_info,
                                         exp_removal_ratios,
                                         y_label="clusterings",
                                         labels=[r'$S$', r'$\langle s \rangle$'],
                                         title=title,
                                         filename=file_name + '_clust.png')


def incremental_attack_poisson(exp_removal_rate, exp_max_rate, exp_num_nodes, exp_mus, is_random_attack, track_net=1):
    for n_nodes in exp_num_nodes:
        for mu in exp_mus:
            net = get_poisson_net(n_nodes=n_nodes, mu=mu, verbose=False)
            if not is_random_attack:
                file_name = "poisson-incr-attack-mu=%.2f-n_nodes=%d" % (mu, n_nodes)
                title = "Incremental Attack - Poisson nodes=%d mu=%.2f " % (n_nodes, mu)

                min_path, max_path, cluster_size_ratios, network_tracking = incremental_attack(net=net,
                                                                                               removal_rate=exp_removal_rate,
                                                                                               max_rate=exp_max_rate,
                                                                                               verbose=True,
                                                                                               track_net_num=track_net)

            else:
                min_path, max_path, cluster_size_ratios, network_tracking = incremental_random_failure(net=net,
                                                                                                       removal_rate=exp_removal_rate,
                                                                                                       max_rate=exp_max_rate,
                                                                                                       track_net_num=track_net)

                file_name = "poisson-incr-failure-mu=%.2f-n_nodes=%d" % (mu, n_nodes)
                title = "Incremental Failure - Poisson nodes=%d mu=%.2f " % (n_nodes, mu)
            save_results(min_path, max_path, cluster_size_ratios, file_name)
            save_network_tracking(network_tracking, title, file_name)
            steps = int(exp_max_rate / exp_removal_rate)
            plot_metric_distribution([min_path, max_path],
                                     np.cumsum([exp_removal_rate] * steps),
                                     y_label="path length",
                                     labels=[r'$d_{min}$', r'$d_{max}$'],
                                     title=title,
                                     filename=file_name + '_distr.png')

            clusters_info = [x[0] for x in cluster_size_ratios]
            plot_clustering_distribution(clusters_info,
                                         np.cumsum([exp_removal_rate] * steps),
                                         y_label="clusterings",
                                         labels=[r'$S$', r'$\langle s \rangle$'],
                                         title=title,
                                         filename=file_name + '_clust.png')


def incremental_attack_powerlaw(exp_removal_rate, exp_max_rate, exp_num_nodes, exp_ks, is_random_attack, track_net=1):
    for n_nodes in exp_num_nodes:
        for k in exp_ks:
            net = get_power_law_net(n_nodes, k, verbose=False)
            if not is_random_attack:
                min_path, max_path, cluster_size_ratios, network_tracking = incremental_attack(net=net,
                                                                                               removal_rate=exp_removal_rate,
                                                                                               max_rate=exp_max_rate,
                                                                                               track_net_num=track_net)

                file_name = "powerlaw-incr-attack-k=%.2f-n_nodes=%d" % (k, n_nodes)
                title = "Incremental Attack - Powerlaw nodes=%d k=%.2f " % (n_nodes, k)
            else:
                min_path, max_path, cluster_size_ratios, network_tracking = incremental_random_failure(net=net,
                                                                                                       removal_rate=exp_removal_rate,
                                                                                                       max_rate=exp_max_rate,
                                                                                                       track_net_num=track_net)

                file_name = "powerlaw-incr-failure-k=%.2f-n_nodes=%d" % (k, n_nodes)
                title = "Incremental Failure - Powerlaw nodes=%d k=%.2f " % (n_nodes, k)

            save_results(min_path, max_path, cluster_size_ratios, file_name)
            save_network_tracking(network_tracking, title, file_name)
            steps = int(exp_max_rate / exp_removal_rate)

            plot_metric_distribution([min_path, max_path],
                                     np.cumsum([exp_removal_rate] * steps),
                                     y_label="path length",
                                     labels=[r'$d_{min}$', r'$d_{max}$'],
                                     title=title,
                                     filename=file_name + '_distr.png')

            clusters_info = [x[0] for x in cluster_size_ratios]
            plot_clustering_distribution(clusters_info,
                                         np.cumsum([exp_removal_rate] * steps),
                                         y_label="clusterings",
                                         labels=[r'$S$', r'$\langle s \rangle$'],
                                         title=title,
                                         filename=file_name + '_clust.png')


def incremental_attack_(net, file_name, title, exp_removal_rate, exp_max_rate, is_random_attack, track_net=1):
    if not is_random_attack:
        min_path, max_path, cluster_size_ratios, network_tracking = incremental_attack(net=net,
                                                                                       removal_rate=exp_removal_rate,
                                                                                       max_rate=exp_max_rate,
                                                                                       verbose=True,
                                                                                       track_net_num=track_net)
    else:
        min_path, max_path, cluster_size_ratios, network_tracking = incremental_random_failure(net=net,
                                                                                               removal_rate=exp_removal_rate,
                                                                                               max_rate=exp_max_rate,
                                                                                               track_net_num=track_net)
    save_results(min_path, max_path, cluster_size_ratios, file_name)
    save_network_tracking(network_tracking, title, file_name)
    steps = int(exp_max_rate / exp_removal_rate)
    plot_metric_distribution([min_path, max_path],
                             np.cumsum([exp_removal_rate] * steps),
                             y_label="path length",
                             labels=[r'$d_{min}$', r'$d_{max}$'],
                             title=title,
                             filename=file_name + '_distr.png')

    clusters_info = [x[0] for x in cluster_size_ratios]
    plot_clustering_distribution(clusters_info,
                                 np.cumsum([exp_removal_rate] * steps),
                                 y_label="clusterings",
                                 labels=[r'$S$', r'$\langle s \rangle$'],
                                 title=title,
                                 filename=file_name + '_clust.png')


if __name__ == "__main__":
    exp_removal_rate = 0.0025
    exp_removal_ratios = np.linspace(0.0, 0.5, 10)

    exp_max_rate = 0.05
    exp_num_nodes = [10000]  # Test the attacks with different sizes of networks
    exp_mus = [4]
    exp_ks = [2.6]

    for is_random_attack in [False]:
        # Poisson
        # incremental_attack_poisson(exp_removal_rate, exp_max_rate, exp_num_nodes, exp_mus, is_random_attack, track_net=7)
        # instantaneous_attack_poisson(exp_num_nodes, exp_removal_ratios, exp_mus, is_random_attack)

        # Scale Free
        # incremental_attack_powerlaw(exp_removal_rate, exp_max_rate, exp_num_nodes, exp_ks, is_random_attack, track_net=7)
        # instantaneous_attack_powerlaw(exp_num_nodes, exp_removal_ratios, exp_ks, is_random_attack)

        # Oregon
        net = oregon.get_network()
        file_name = title = "oregon"
        # oregon.degree_distribution(net)
        incremental_attack_(net, file_name, title, exp_removal_rate, exp_max_rate, is_random_attack, track_net=6)



    params1 = {"n_agents": 1, "map_type": "small", "logs_path": logs_path, "n_episodes": EPISODES, "n_steps": STEPS,
               "batch_size": BATCH_SIZE, "lr": 0.0005, "gamma": 0.99, "epsilon": 0.15, "epsilon_decay": 0.999,
               "log": True}

    processes = []

    for i, params in enumerate([params1.values(), params2.values()]):
        p = Process(target=train_agents, args=params, name="Exp%d" % i)
        p.start()
        processes.append(p)
        print(p.name)

    for p in processes:
        p.join()