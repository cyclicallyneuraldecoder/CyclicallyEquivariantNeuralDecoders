import random
import os
import sys

import numpy as np
import torch
import torch.optim as optim

from util.log_util import get_logger
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR

from util.data_util import SignalDataset
from util.data_util import SignalTestset
from util.conf_util import get_default_conf

from model.cycnet import cycnet

if len(sys.argv) == 2:
    conf_name = sys.argv[1]
    print("test conf_name:", conf_name)
    conf = get_default_conf(f"./config/{conf_name}.yaml")
else:
    print("default")
    conf = get_default_conf()

if torch.cuda.is_available():
    device = torch.device("cuda")
    os.environ["CUDA_VISIBLE_DEVICES"] = conf["para"]["CUDA_VISIBLE_DEVICES"]
else:
    device = torch.device("cpu")

logger = get_logger(conf["para"]["logger_name"])

# Validate part


def test(model, device, test_loader, para, Boosting_number):
    model = model.to(device)
    model.eval()
    BER_total = []
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data = data.to(device)
            target = target.to(device).to(torch.float)
            output = data
            for i in range(int(Boosting_number)):
                output = model(output, False)
            results = 1 - torch.sigmoid(output * 1000000)
            bool_equal = (results == target).to(torch.float)
            word_target = conf["data"]["v_size"] * \
                torch.ones(1, conf["para"]["test_batch_size"])
            word_target = word_target.cuda()
            codeword_equal = (torch.sum(bool_equal, -1).cuda()
                              == word_target).to(torch.float)
            BER = 1 - (torch.sum(bool_equal) /
                       (results.shape[0] * results.shape[1]))
            BER_total.append(BER)
        BER = torch.mean(torch.tensor(BER_total))
        snr = para["snr"]
        logger.warning(f"SNR={snr},Boosting_num={int(Boosting_number)-1},BER={BER:.7f}")


if __name__ == "__main__":
    para = conf["para"]
    seed = para["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    for snr in conf["para"]["snr_list"].split(","):
        para["snr"] = int(snr)
        for Boosting_number in conf["para"]["Boosting_number_list"].split(","):
            model = cycnet(conf, device).to(device)
            model.load_state_dict(torch.load(conf["para"]["test_model_path"]))
            testset = SignalTestset(conf)
            test_loader = DataLoader(testset, batch_size=conf["para"]["test_batch_size"])
            test(model, device, test_loader, para, Boosting_number)
