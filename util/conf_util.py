import yaml
import numpy as np

def get_default_conf(conf_path=None):
    if conf_path is None:
        conf_path = "./config/default.yaml"
    with open(conf_path, "r") as f_conf:
        conf = yaml.load(f_conf.read(), Loader=yaml.FullLoader)

    data_type = conf["para"]["data_type"]

    conf["data"]["H_path"] = conf["data"]["H_path"].format(
        data_type, data_type)
    conf["data"]["G_path"] = conf["data"]["G_path"].format(
        data_type, data_type)
    conf["data"]["perma_path"] = conf["data"]["perma_path"].format(data_type)
    conf["para"]["test_model_path"] = conf["para"]["test_model_path"].format(
        data_type, data_type)
    conf["para"]["logger_name"] = conf["para"]["logger_name"].format(data_type)
    H = np.loadtxt(conf["data"]["H_path"])
    G = np.loadtxt(conf["data"]["G_path"])
    perma = np.loadtxt(conf["data"]["perma_path"])

    k = len(G)
    n = len(G[0])
    rate = 1.0 * k / n
    v_size = len(H[0])

    pos = []
    for i in range(len(H)):
        for j in range(len(H[0])):
            if H[i][j] == 1:
                pos.append([i, j])

    e_size = len(pos)

    conf["data"]["v_size"] = v_size
    conf["data"]["e_size"] = e_size
    conf["data"]["l_size"] = e_size//v_size
    conf["data"]["rate"] = rate
    conf["data"]["pos"] = pos
    conf["data"]["H"] = H
    conf["data"]["G"] = G
    conf["data"]["perma"] = perma
    return conf
