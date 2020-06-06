

from utils.visualization import plot_comparisons_from_file_clustering, plot_comparisons_from_file_metrics
import utils.data_saver as ds

exp_removal_rate = 0.025
exp_max_rate = 0.5


def mu_experiment():
    mus = [2.00, 4.00, 8.00]
    n = 1000

    nets = ["poisson", "powerlaw"]
    types = ["attack", "failure"]
    net = nets[0]
    type = types[0]

    filename = "%s-incr-%s-mu=%.2f-n_nodes=%d"

    # poisson, attack
    filenames = []
    labels = []
    for mu in mus:
        filenames.append(filename % (net, type, mu, n))
        labels.append("mu={}".format(mu))

    file_name = "mu_%s-incr-%s-n_nodes=%d" % (net, type, n)
    title = "Mu Comparison for Incremental %s [%s network]" % (type, net)
    plot_comparisons_from_file_metrics(file_name, title, filenames, labels, exp_max_rate, exp_removal_rate)
    plot_comparisons_from_file_clustering(file_name, title, filenames, labels, exp_max_rate, exp_removal_rate)

def k_experiment():
    ks = [2.10, 2.50, 2.80]
    n = 1000

    nets = ["poisson", "powerlaw"]
    types = ["attack", "failure"]
    net = nets[1]
    type = types[0]

    filename = "%s-incr-%s-k=%.2f-n_nodes=%d"

    # poisson, attack
    filenames = []
    labels = []
    for k in ks:
        filenames.append(filename % (net, type, k, n))
        labels.append("k={}".format(k))

    file_name = "k_%s-incr-%s-n_nodes=%d" % (net, type, n)
    title = "K Comparison for Incremental %s [%s network]" % (type, net)
    plot_comparisons_from_file_metrics(file_name, title, filenames, labels, exp_max_rate, exp_removal_rate)
    plot_comparisons_from_file_clustering(file_name, title, filenames, labels, exp_max_rate, exp_removal_rate)

def poisson_vs_powerlaw_experiment():

    filenames = ["poisson-incr-attack-mu=2.00-n_nodes=1000",
                 "powerlaw-incr-attack-k=2.80-n_nodes=1000"]
    labels = ["poisson", "powerlaw"]


    file_name = "incr-attack-poisson_vs_powerlaw-n_nodes=1000"
    title = "Poisson vs Powerlaw [incremental attack, N=1000]"
    plot_comparisons_from_file_metrics(file_name, title, filenames, labels, exp_max_rate, exp_removal_rate)
    plot_comparisons_from_file_clustering(file_name, title, filenames, labels, exp_max_rate, exp_removal_rate)

def attack_vs_fail_experiment():

    filenames = ["poisson-incr-attack-mu=2.00-n_nodes=1000",
                 "poisson-incr-failure-mu=2.00-n_nodes=1000"]
    labels = ["attack", "failure"]


    file_name = "incr-poisson-attack_vs_failure-n_nodes=1000"
    title = "Attack vs Failure [incremental Poisson, N=1000]"
    plot_comparisons_from_file_metrics(file_name, title, filenames, labels, exp_max_rate, exp_removal_rate)
    plot_comparisons_from_file_clustering(file_name, title, filenames, labels, exp_max_rate, exp_removal_rate)


#mu_experiment()
#k_experiment()
#poisson_vs_powerlaw_experiment()
#attack_vs_fail_experiment()

