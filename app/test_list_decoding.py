import random
import os
import sys

import numpy as np
import torch
import torch.optim as optim

from torch.nn import BCEWithLogitsLoss
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR

from util.data_util import SignalDataset
from util.data_util import SignalTestset
from util.conf_util import get_default_conf
from util.log_util import get_logger
from model.cycnet import cycnet

if len(sys.argv) == 2:
    conf_name = sys.argv[1]
    print(conf_name)
    conf = get_default_conf(f"./config/{conf_name}.yaml")
else:
    print("default")
    conf = get_default_conf()

os.environ["CUDA_VISIBLE_DEVICES"] = str(conf["para"]["CUDA_VISIBLE_DEVICES"])
logger = get_logger(conf["para"]["logger_name"])
device = torch.device("cuda")

perma = conf["data"]["perma"]
H = conf["data"]["H"]
H = torch.from_numpy(H).to(device).to(torch.float)
Boosting_number = conf["para"]["Boosting_number"]
if type(conf['para']['list_size']) != int:
    list_size = conf['para']['list_size'].split(',')
else:
    list_size = [str(conf['para']['list_size'])]


def boosting(output, Boosting_number):
    for i in range(Boosting_number):
        output = model(output, False)
    return output


def add_list_size(list_size):
    list_n = []
    for i in list_size:
        if "len" in i:
            list_n.append(eval(i))
        else:
            list_n.append(int(i))
    return list_n


def test(model, device, test_loader, para,Boosting_number,conf):
    model.eval()
    zero = torch.zeros(para["test_batch_size"], 1).to(device).to(torch.float)
    list_decoding_num = add_list_size(list_size)
    with torch.no_grad():
        for i_perm_int in list_decoding_num:
            FER_total = []
            ML_lower_bound_total = []
            for batch_idx, (data, target) in enumerate(test_loader):
                data = data.to(device).to(torch.float)
                target = target.to(device).to(torch.float)
                data1 = torch.cat([zero, data], 1)
                Best_sum_LLR = -100000 * \
                    torch.ones(para['test_batch_size']).to(device)
                Best_codeword = torch.zeros(
                    para['test_batch_size'], len(H[0])).to(device)
                for i_perm in range(i_perm_int):
                    count = 0
                    permutation = np.zeros([len(H[0])+1, len(H[0])+1])
                    for j_perm in perma[i_perm]:
                        permutation[np.int(j_perm)][count] = 1
                        count = count + 1
                    per = torch.from_numpy(permutation).cuda().to(torch.float)
                    data2 = torch.matmul(data1, per)
                    data3 = data2[:, 1:]

                    output = boosting(data3, Boosting_number)

                    results = 1 - torch.sigmoid(output * 1000000)
                    count = torch.sum(torch.transpose(torch.matmul(
                        H, torch.transpose(results, 1, 0)), 1, 0) % 2, 1)
                    results[count != 0] = 0

                    results1 = torch.sum(results, 1) % 2
                    results2 = torch.cat((results1.unsqueeze(1), results), -1)
                    results3 = 1 - 2 * results2  # BPSK
                    temp_sum = torch.sum(results3 * data2, 1).to(torch.float)

                    inverse_per = per.inverse().to(torch.float)

                    for i in range(len(temp_sum)):
                        if temp_sum[i] > Best_sum_LLR[i] and count[i] == 0:
                            Best_sum_LLR[i] = temp_sum[i]
                            Extended_word = torch.matmul(
                                results2[i].to(torch.float), inverse_per)
                            Best_codeword[i] = Extended_word[1:]

                codeword_error = torch.zeros(para["test_batch_size"])
                ML_error = torch.zeros(para["test_batch_size"])

                for i in range(para["test_batch_size"]):
                    bool_equal = (Best_codeword[i] == target[i]).to(
                        torch.float)
                    if torch.sum(bool_equal) != len(H[0]):
                        codeword_error[i] = 1
                        PMT = 1 - 2 * target[i]
                        true_sum = torch.matmul(
                            PMT.to(torch.float), data[i].to(torch.float)).to(device)
                        if true_sum < Best_sum_LLR[i]:
                            ML_error[i] = 1
                snr = conf["para"]["snr"]
                FER = torch.sum(codeword_error)/conf["para"]["test_batch_size"]
                ML_lower_bound = torch.sum(ML_error) / conf["para"]["test_batch_size"]
                ML_lower_bound_total.append(ML_lower_bound)
                FER_total.append(FER)
            FER = torch.mean(torch.tensor(FER_total))
            ML_lower_bound = torch.mean(torch.tensor(ML_lower_bound_total))
            logger.warning(f"list_num={i_perm_int},Boosting_number={Boosting_number},snr={snr},FER={FER:.8f},ML_lower_bound={ML_lower_bound:.8f}")


if __name__ == "__main__":
    para = conf["para"]
    seed = para["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = cycnet(conf, device).to(device)
    model.load_state_dict(torch.load(conf["para"]["test_model_path"]))
    param_size = sum([param.nelement() for param in model.parameters()])
    print(f"parameters size: {param_size}\n")

    for snr in conf["para"]["snr_list"].split(","):
        para["snr"] = int(snr)
        testset = SignalTestset(conf)
        test_loader = DataLoader(testset, batch_size=conf["para"]["test_batch_size"])
        test(model, device, test_loader, para, Boosting_number,conf)
