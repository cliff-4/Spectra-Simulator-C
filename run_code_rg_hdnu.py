import os
import numpy as np
import time
import random

np.random.seed(os.getpid() + os.getppid() + int((time.time())))

snr_max = np.load("snr_max_rg.npy")


def sample_generation(bins, parameter_prob, num_samples):
    bins_size = np.mean(np.diff(bins))
    temp_bins = bins[:-1] + bins_size
    num_bins = temp_bins.shape[0]
    sample_indices = np.random.choice(
        np.arange(num_bins), num_samples, p=parameter_prob
    )
    random_weights = np.random.random(size=sample_indices.shape)
    samples = (
        random_weights * bins[sample_indices]
        + (1.0 - random_weights) * bins[sample_indices + 1]
    )
    return samples


hist, bins = np.histogram(
    snr_max[(snr_max < 155) & (snr_max > 8)], bins=100, density=True
)
bins_size = np.mean(np.diff(bins))

tobs_non_rg = np.load("tobs_non_rg.npy")

hist_tobs, bins_tobs = np.histogram(tobs_non_rg, bins=15, density=True)
bins_size_tobs = np.mean(np.diff(bins_tobs))


def sample_generation_dnu():
    sample_index = np.random.choice(np.arange(3), 1, p=[6.0 / 15, 5.0 / 15, 4.0 / 15])
    random_weights = np.random.random()
    if sample_index == 0:
        sample = random_weights * 1.0 + (1.0 - random_weights) * 5.5
    elif sample_index == 1:
        sample = random_weights * 5.5 + (1.0 - random_weights) * 11.5
    else:
        sample = random_weights * 11.5 + (1.0 - random_weights) * 18.7

    return sample


def sample_spectrum(snr, tobs):
    # Dnu = sample_generation_dnu()
    nurot_env = np.random.uniform(0.005, 0.4)  # np.minimum(Dnu/2.,0.4)
    nurot_core = np.random.uniform(0.005, 2.8)  # np.minimum(Dnu/2.,2.8)
    N0 = 10.0 ** np.random.uniform(-1, 4.47)
    Dp_min = np.random.choice([40.0, 150.0], p=[0.9, 0.1])
    # Dp_max = 10.**np.random.uniform(-1,4.47)
    # N0_max = 2000.#np.random.choice([2000.,40000.])
    if Dp_min == 40.0:
        Dp_max = 150.0
    elif Dp_min == 150.0:
        Dp_max = 500.0
    f = open("Configurations/main.cfg", "w")
    f.write(
        """# Configuration in order to generate an ensemble of spectra automatically using [val_min] and [val_max] as the limits of the parameter space
        # Seven inputs are required:
        #               - model_name : Name of the model, as defined in models_database.pro
        #               - The name of the parameters as defined in models_database.pro, for the subfunction [model_name]
        #               - val_min : vector listing the initial parameters of the model [model_name]. See models_database.pro for more information about those parameters
        #               - val_max:  vector of same size as [val_min] with the final parameters of the model.
        #               - Tobs and Cadence: Observation duration (in days) and the Cadence of observation (in seconds)
        #               - forest_type: either grid or random. Currently, only random (uniform) is implemented.
        #               - erase_old_file: If 1, then (1) the combination file is overwritten and (2) the model number (identifier) is reset to 0.
        #                                 If 0, then (1) append the combination file and (2) model number = last model number + 1
        random 1 # forest_type, followed by the forest_params. If forest_type=random, then forest_params is a single value that corresponds to the Number of samples
        asymptotic_mm_freeDp_numaxspread_curvepmodes_v3                 # Name of the model. asymptotic_mm_freeDp_curvepmodes_v1 is a model iterating on what was learnt from asymptotic_v1, v2, v3 to generate a star with mixed modes
        all                     # Used template(s) name(s). If several, randomly select one/iteration. If set to 'all', will use all *.template files in Configuration/templates
        nurot_env nurot_core   Dnu  epsilon    delta0l_percent  beta_p_star  nmax_spread  DP1    alpha    q      SNR    maxGamma   numax_spread    Vl1    Vl2    Vl3   H0_spread     A_Pgran   B_Pgran  C_Pgran     A_taugran   B_taugran   C_taugran   P      N0      Hfactor    Wfactor\n"""
    )

    f.write(
        """%.4f      %.4f     9.    0.       0.2              0.000         5           %.3f     0.     0.    %.3f     0.05        10              0.3      0.15   0.0   10               0.01     -2.5       1.0        1.0           -1.6         1.0      1.0    %f     0.0     0.4    #val_min"""
        % (nurot_env, nurot_core, Dp_min, snr, N0)
    )

    f.write(
        """\n0.4000      2.8000      19.0   1.       3.5                0.100         5           %.3f    1.0     0.5    160.   0.35        10              2.5   0.80   0.1   10               3.5     -1.5      1.0        3.5           -0.7        1.0      4.0    31000     1.0     1.0    #val_max
        0.           0.        1.     1.       1.                       1.            1.          1.     1.       1.     0.     1.         0.              1.     1.     1.    0                 0.      0.        0.        0.             1           0        1      0      1      1   #If forest_type="random" ==> 1=Variable OR 0=Constant. If forest_type="grid" then must be the stepsize of the grid
        Tobs   Cadence  Naverage    Nrealisation
        %f     1754.38       1            1"""
        % (Dp_max, tobs)
    )
    f.write(
        """
        0     # It is erase_old_files. If set to 1, will remove old Combination.txt and restart counting from 1. Otherwise append Combination.txt
        0     # Do you want plots ? 0 = No, 1 = Yes - Recommended 0
        1     # Do you list values of the input model in the output ascii file?"""
    )
    f.close()

    cmd = "./sim.out > temp_sim.out"
    k = os.system(cmd)
    return k


i = 0
while i < 10000:
    if i >= 10000:
        break
    try:
        now = time.time()
        future = now + 10
        while time.time() < future:
            tobs = sample_generation(bins_tobs, hist_tobs * bins_size_tobs, 1)[-1]
            snr = sample_generation(bins, hist * bins_size, 1)[-1]
            k = sample_spectrum(snr, tobs)
            if k >= 0:
                break

        print(k)
        if k == 0:
            print("completed example")
            print(i)
            myfile = open("Data/Combinations.txt", "r")
            lines = myfile.readlines()
            myfile1 = open("Data/Spectra_info/%07d.txt" % (i + 1), "w")
            myfile1.writelines(lines[-1])
            myfile.close()
            myfile1.close()
            i += 1
        elif i != 0:
            myfile = open("Data/Combinations.txt", "r")
            # myfile = open('/scratch/siddharth.dhanpal/data_3/ver_'+str(i)+'/Spectra-Simulator-C/models_database.cpp', 'r')
            lines = myfile.readlines()
            myfile1 = open("Data/Combinations.txt", "w")
            # myfile1 = open('/scratch/siddharth.dhanpal/data_3/ver_'+str(i)+'/Spectra-Simulator-C/models_database.cpp', 'w')
            lines = lines[:-1]
            myfile1.writelines(lines)
            print("incomplete")
            myfile.close()
            myfile1.close()
    except:
        pass
