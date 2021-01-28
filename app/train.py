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
    print("train conf_name:", conf_name)
    conf = get_default_conf(f"./config/{conf_name}.yaml")
else:
    print("default")
    conf = get_default_conf()

if torch.cuda.is_available():
    device = torch.device("cuda")
    os.environ["CUDA_VISIBLE_DEVICES"] = conf["para"]["CUDA_VISIBLE_DEVICES"]
    print(device,os.environ["CUDA_VISIBLE_DEVICES"])
else:
    device = torch.device("cpu")
    print(device)


logger = get_logger(conf["para"]["logger_name"])

def train(model, device, train_loader, para, conf, epoch, criterion):
    model.train()
    model = model.to(device)
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        data = data.to(device).to(torch.float)
        target = target.to(device).to(torch.float)
        output = model(data)
        loss = criterion(-output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % para["log_interval"] == 0:
            print(f"epoch: {epoch} bz: {batch_idx} loss: {loss:.4f}")

def test(model, device, test_loader, para,epoch):
    model = model.to(device)
    model.eval()
    BER_total = torch.zeros(len(test_loader))
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data = data.to(device)
            target = target.to(device).to(torch.float)
            output = model(data, False)
            results = 1 - torch.sigmoid(output * 1000000)
            bool_equal = (results == target).to(torch.float)
            word_target = conf["data"]["v_size"] * \
                torch.ones(1, conf["para"]["test_batch_size"])
            word_target = word_target.cuda()
            codeword_equal = (torch.sum(bool_equal, -1).cuda()
                              == word_target).to(torch.float)
            BER = 1 - (torch.sum(bool_equal) /
                       (results.shape[0] * results.shape[1]))
            BER_total[batch_idx] = BER
        BER = torch.mean(BER_total)
        snr = para["snr"]
        logger.warning(f"Epoch={epoch},SNR={snr},BER={BER:.7f}")


if __name__ == "__main__":
    para = conf["para"]
    seed = para["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    dataset = SignalDataset(conf)
    testset = SignalTestset(conf)
    train_loader = DataLoader(dataset, batch_size=para["train_batch_size"])
    test_loader = DataLoader(testset, batch_size=para["test_batch_size"])

    model = cycnet(conf, device).to(device)
    criterion = BCEWithLogitsLoss()
    optimizer = optim.RMSprop(model.parameters(), lr=para["lr"])
    param_size = sum([param.nelement() for param in model.parameters()])
    print(f"parameters size: {param_size}\n")

    for epoch in range(1, para['epoch_szie'] + 1):
        model.train()
        train(model, device, train_loader, para, conf, epoch, criterion)
        test(model, device, test_loader, para,epoch)
        data_type = conf["para"]["data_type"]
        train_save_path = "/".join(conf["para"]["train_save_path"].split("/")[:-1]).format(data_type)
        if not os.path.exists(train_save_path):
            os.makedirs(train_save_path)
        torch.save(model.state_dict(), conf["para"]["train_save_path"].format(data_type, data_type, epoch))
