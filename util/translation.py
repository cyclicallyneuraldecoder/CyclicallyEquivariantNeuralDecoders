import numpy as np
import os
np.set_printoptions(suppress=True)
# BCH translation
path = '../GenAndPar.txt'
with open(path,"r") as f: 
    data = f.readlines()
    if data[0][1] == "n":
        # BCH GenAndPar use
        exec(data[0].strip())
        exec(data[1].strip())
        g = list(map(int, data[3].strip().split()))            
        h = list(map(int, data[5].strip().split()))

        # Produce parity check matrix 
        zeros1 = [0] * (n - len(h))
        h = h + zeros1
        H = []
        for j in range(n):
            h1 = h[n-j:]
            h2 = h[:n-j]
            h3 = np.hstack((h1,h2))
            H.append(h3)
        np.savetxt(f"../BCH_H_{n}_{k}.txt",H,fmt='%i')
        
        # Produce generator matrix
        G = []
        zeros2 = [0] * (n - len(g))
        g = g + zeros2
        for j in range(k):
            g1 = g[n-j:]
            g2 = g[:n-j]
            g3 = np.hstack((g1,g2))
            G.append(g3)
        np.savetxt(f"../BCH_G_{n}_{k}.txt",G,fmt='%i')
    else:
        # RM GenAndPar use
        exec(data[3].strip())
        exec(data[4].strip())
        g = list(map(int, data[6].strip().split()))            
        h = list(map(int, data[8].strip().split()))

        # Produce parity check matrix 
        zeros1 = [0] * (n - len(h))
        h = h + zeros1
        H = []
        for j in range(n):
            h1 = h[n-j:]
            h2 = h[:n-j]
            h3 = np.hstack((h1,h2))
            H.append(h3)
        np.savetxt(f"../RM_H_{n}_{k}.txt",H,fmt='%i')
        
        # Produce generator matrix
        G = []
        zeros2 = [0] * (n - len(g))
        g = g + zeros2
        for j in range(k):
            g1 = g[n-j:]
            g2 = g[:n-j]
            g3 = np.hstack((g1,g2))
            G.append(g3)
        np.savetxt(f"../RM_G_{n}_{k}.txt",G,fmt='%i')
