import numpy as np

def padded_args(s: str):
    t = 0
    T = ""
    for x in s.split(" "):
        if x != "":
            T += x.strip().ljust(20)
            t += 1
    return T

def sample_spectrum():
        nurot_env_max = np.random.choice([0.2,0.4],p=[0.80,0.20])
        if (nurot_env_max==0.2):
            nurot_env = np.random.uniform(0.005,0.2)
        elif (nurot_env_max==0.4):
            nurot_env = np.random.uniform(0.2,0.4)

        nurot_core= np.random.uniform(0.005,2.8)
        a3 = 0.0
        snr = np.random.uniform(10.0,160.0)
        tobs = np.random.uniform(1300.0,1472.0)
        N0 = 10.**np.random.uniform(-1.0,3.0)
        qmax = 0.50
        dnu_min = np.random.choice([4.,9.],p=[0.60,0.40])
        if (dnu_min==4.):
            dnu = np.random.uniform(1.0,9.0)
            Dp_range = np.random.choice([40.,150.],p=[0.5,0.5])
            if Dp_range==40.:
                Dp_min=40.
                Dp_max=150.
            elif Dp_range==150.:
                Dp_min=150.
                Dp_max=500.
                qmax = 0.65
        elif (dnu_min==9.):
            dnu = np.random.uniform(9.0,19.0)
            Dp_range = np.random.choice([40.,150.],p=[0.9,0.1])
            if Dp_range==40.:
                Dp_min=40.
                Dp_max=150.
            elif Dp_range==150.:
                Dp_min=150.
                Dp_max=500.
        
        T = ""
        T += padded_args("""nurot_env     nurot_core      a2_l1_core    a2_l1_env    a2_l2_env   a2_l3_env   a3_l2_env     a3_l3_env     a4_l2_env    a4_l3_env    a5_l3_env    a6_l3_env    Dnu   epsilon    delta0l_percent  beta_p_star  nmax_spread   DP1    alpha    q      SNR    maxGamma   numax_spread    Vl1    Vl2    Vl3   H0_spread     A_Pgran   B_Pgran  C_Pgran    A_taugran   B_taugran   C_taugran   P      N0      Hfactor   Wfactor""")
        T += "\n"

        T += padded_args("""%.4f      %.4f           0           0           0           -9999        %.4f         -9999          0.0         -9999         0.0          0.0      %.3f     0.       0.2              0.000         5           %.3f     0.     0.     %f     0.05        10             0.3    0.15   0.0     10         0.01     -2.5       1.0        1.0        -1.6         1.0        1.0      %f       0.4       0.4    #val_min"""%(nurot_env,nurot_core,a3,dnu,Dp_min,snr,N0))
        T += "\n"

        T += padded_args("""0.4000      2.8000      0           0.          0.1          0.1        0.05          0.05          0.05         0.05         0.05       0.05       19.0      1.       5.0              0.100         5           %.3f    1.0    %.2f   160.0   0.14        10             2.5   0.80   0.1     10           3.5     -1.5       1.0        3.5       -0.7          1.0       4.0     31000     1.0       1.0    #val_max"""%(Dp_max,qmax))
        T += "\n"

        T += padded_args("""0.              0.                   0           0           0            0           0             0             0            0            0.           0.        0.       1.       1.                1.          0.           1.     1.       1.     0.     1.          0.             1.    1.     1.      0            0.       0.        0.        0.          1            0         1       0          1       1   #1=Variable_OR_0=Constant""")
        print(T)

sample_spectrum()