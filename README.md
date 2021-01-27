# Cyclically Equivariant Neural Decoder

An implementation of Cyclically Equivariant Neural Decoders described in "Cyclically Equivariant Neural Decoders for Cyclic Codes" (Pytorch implementation)

## Abstract

Neural decoders were introduced as a generalization of the classic Belief Propagation (BP) decoding algorithms, where the Trellis graph in the BP algorithm is viewed as a neural network, and the weights in the Trellis graph are optimized by training the neural network. In this work,we propose a novel neural decoder for cyclic codes by exploiting their cyclically invariant property.More precisely, we impose a shift invariant structure on the weights of our neural decoder so that any cyclic shift of inputs results in the same cyclic shift of outputs. Extensive simulations with BCH codes and punctured Reed-Muller(RM) codes show that our new decoder consistently outperforms previous neural decoders when decoding cyclic codes. Finally, we propose a list decoding procedure that can significantly reduce the decoding error probability for BCH codes and punctured RM codes.


## Install

- Pytorch == 1.6.0

- Python3 (Recommend Anaconda)

- Matlab 2018b with Communications Toolbox 7.0


```bash
pip install -r requirements.txt
```

## Produce parity check matrix and generator matrix
For BCH code, run ```GenAndPar.m```   
For punctured RM code, run ```GenPolyRM.m```  
This gives you the coefficients of generator polynomial and parity check polynomial. For example, running ```GenAndPar.m``` with parameters n=63 and k=45 gives you

```
 n = 63 
 k = 45 
 Generator matrix row: 
1 1 1 1 0 0 1 1 0 1 0 0 0 0 0 1 1 1 1 
 Parity matrix row: 
1 1 0 0 1 1 0 0 1 0 0 0 0 0 1 1 0 0 1 0 0 1 1 1 1 1 0 0 1 1 0 1 0 0 1 0 1 0 1 1 1 1 0 0 1 1 
```

Another example: running ```GenPolyRM.m``` with parameters m=6,r=3 gives you

```
Punctured Reed-Muller codes parameters are 
 m = 6 
 r = 3 
 n = 63 
 k = 42 
 Generator matrix row: 
1 1 0 1 0 0 0 1 1 1 1 1 1 0 0 1 1 0 1 0 0 1 
 Parity matrix row: 
1 0 0 1 0 1 0 0 0 0 0 0 0 0 1 0 0 1 0 0 1 0 1 1 0 0 1 1 0 0 1 0 0 1 0 0 0 0 1 0 1 1 1 
```

After obtaining these coefficients, we use ```util.translation.py``` to produce the parity check matrix and the generator matrix. See ```BCH_H_63_45.txt``` for an example of parity check matrix of BCH(63,45) and ```BCH_G_63_45.txt``` for an example of generator matrix of BCH(63,45).

## Produce the permutations used in list decoding

- Set the parameter m in ```GFpermutation.m``` (Example: for BCH(63,45), pick m=6 because 2^6=64).  
- Run ```GFpermutation.m``` in Matlab with Communications Toolbox and we get the ```GFpermutation.txt```
- Put the path in config to get the permutation matrix used in ```test_list_decoding.py```. Example: For BCH(63,45), put the path of ```GFpermutation.txt``` in ```config.bch6345.yaml```


## Hyperparameters

In config file

## Results reproduction

To reproduce the performance of the model in our paper for BCH(63,45):

1. Run ```python -m app.train bch6345``` (bch6345 is the config name for BCH(63,45). Change it to other names for other codes)

2. Train with GPU. For 30 minutes of training with 1080Ti 11GB GPU (or for 10 minutes of training with 3090Ti 32GB), around epoch 10 you should get the model in save

3. Run ```python -m app.test bch6345``` to get test ber results

    SNR range in [dB] - [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]  
    Bit Error rate - [8.47E-02, 5.12E-02, 2.19E-02, 5.95E-03, 9.44E-04, 7.78E-05]

4. Run ```python -m app.test_list_decoding bch6345``` to get the Frame Error Rate with list_decoding.
| List size\SNR | 1        | 2        | 3        | 4        | 5        | 6        |
| ------------- | -------- | -------- | -------- | -------- | -------- | -------- |
| 1             | 7.64E-01 | 4.86E-01 | 2.03E-01 | 4.94E-02 | 6.27E-03 | 2.90E-04 |
| 2             | 7.15E-01 | 4.22E-01 | 1.55E-01 | 3.09E-02 | 3.16E-03 | 2.20E-04 |
| 4             | 6.57E-01 | 3.53E-01 | 1.13E-01 | 1.81E-02 | 1.26E-03 | 5.00E-05 |
| 8             | 5.92E-01 | 2.84E-01 | 7.75E-02 | 9.58E-03 | 5.60E-04 | 3.00E-05 |
| 16            | 5.31E-01 | 2.30E-01 | 5.31E-02 | 5.58E-03 | 2.30E-04 | 1.00E-05 |
| 64            | 4.48E-01 | 1.64E-01 | 3.11E-02 | 2.34E-03 | 7.00E-05 | 0.00E+00 |