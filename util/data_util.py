# data_util
import numpy as np
import torch
from torch.utils.data import Dataset

class SignalDataset(Dataset):
    def __init__(self, conf):
        print("loading train_data...")
        para = conf["para"]
        seed = para["seed"]
        np.random.seed(seed)
        self.v_size = conf["data"]["v_size"]
        self.e_size = conf["data"]["e_size"]
        self.rate = conf["data"]["rate"]
        self.train_dataset_length = conf["para"]["train_dataset_length"]
        self.batch_size = conf["para"]["train_batch_size"]
        self.input_data = np.zeros([self.train_dataset_length,self.v_size])
        self.label_data = np.zeros([self.train_dataset_length,self.v_size])
        self.snr_vali = conf["para"]["snr"] 

        m = 0
        for i in range(self.train_dataset_length//self.batch_size):
            for snr in [1,2,3,4,5,6,7,8]:
                for j in range(20):
                    xn, npower = self.awgn(snr)
                    self.input_data[m] = 2 * xn / npower
                    self.label_data[m] = np.zeros(self.v_size)
                    m = m + 1

    def __len__(self):
        return len(self.input_data)

    # Make up some real data
    def awgn(self, snr):
        x = np.zeros(self.v_size)
        x = 1 - 2 * x  # BPSK
        snr = 10 ** (snr / 10.0)
        xpower = np.sum(x ** 2) / self.v_size
        npower = xpower / (2 * snr * self.rate)
        return x + np.random.randn(self.v_size) * np.sqrt(npower), npower

    def __getitem__(self, idx):
        _input = torch.from_numpy(self.input_data[idx])
        label = torch.from_numpy(self.label_data[idx])
        return _input,label
       
class SignalTestset(Dataset):
    def __init__(self,conf):
        print("loading test_data...")
        para = conf["para"]
        seed = para["seed"]
        np.random.seed(seed)
        self.v_size = conf["data"]["v_size"]
        self.e_size = conf["data"]["e_size"]
        self.test_dataset_length = conf["para"]["test_dataset_length"]
        self.test_batch_size = conf["para"]["test_batch_size"]
        self._test = np.zeros([self.test_dataset_length , self.v_size])
        self.label1 = np.zeros([self.test_dataset_length, self.v_size])
        self.snr_vali = conf["para"]["snr"]
        self.G = conf["data"]["G"]

    def __len__(self):
        return len(self._test)

    def __getitem__(self,idx):
        # BCH Encode
        G = self.G
        k = len(G)
        n = len(G[0])
        r = n - k
        rate = 1.0 * k / n
        label1 = np.random.randint(0, 2, size=len(G))
        label1 = np.mod(np.matmul(label1, G), 2)
        modSignal = 1 - 2 * label1
        Es = 1
        snr = 10 ** (self.snr_vali / 10.0)
        npower = Es / (2 * snr * rate)  
        # npower = sigma2
        # N = np.zeros(n])
        N = np.sqrt(npower) * np.random.randn(n)
        receivedSignal = modSignal + N
        _test = 2 * receivedSignal / npower
        return _test, label1