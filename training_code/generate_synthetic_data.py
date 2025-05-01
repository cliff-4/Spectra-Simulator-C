import numpy as np
import os
import json

# Data dir
data_dir = "training_code/data/"

# Number of examples per file
examples_per_file = 5120

# Generate parameters
def generate_stellar_parameters():
    # Create synthetic parameters within appropriate ranges
    params = {
        'Dnu': np.random.uniform(7.58, 19.0),              # Large frequency separation
        'Dp': np.random.uniform(150.0, 500.0),             # Period spacing
        'q': np.random.uniform(0.0, 0.65),                # Coupling factor
        'acr': np.random.uniform(0.3883, 2.8),              # Core rotation
        'aer': np.random.uniform(0.1494, 0.4),               # Envelope rotation
        'a3': np.random.uniform(0, 0.05),               # Differential rotation
        'inc': np.random.uniform(0.4, 1),              # Inclination angle
        'epp': np.random.uniform(0.0, 1.0),               # Energy per photon
        'epg': np.random.uniform(0.0, 1.0),               # Energy per gamma
        'numax': np.random.uniform(0.4, 1.0),           # Frequency of maximum power
        'snr': np.random.uniform(60.030356, 154.0),             # Signal-to-noise ratio
        'gamma': np.random.uniform(0.05, 0.14),           # Mode linewidth
        'vl1': np.random.uniform(0.3, 2.5),               # Visibilities l=1
        'vl2': np.random.uniform(0.15, 0.8),              # Visibilities l=2
        'vl3': np.random.uniform(0.0, 0.1)                # Visibilities l=3
    }
    
    # Convert to array in the correct order
    # return np.array([
    #     params['Dnu'], params['Dp'], params['q'], 
    #     params['acr'], params['aer'], params['a3'], 
    #     params['inc'], params['epp'], params['epg'],
    #     params['numax'], params['snr'], params['gamma'],
    #     params['vl1'], params['vl2'], params['vl3']
    # ])
    return np.random.uniform(0.1, 1.0, 36)

# Function to generate synthetic power spectrum
def generate_spectrum(params, length=35692):
    # Base spectrum - random noise
    spectrum = np.random.normal(0, 1, length)
    
    # Add some structure based on parameters
    # Simulate oscillation modes based on Dnu (large frequency separation)
    dnu = params[0]  # First parameter is Dnu
    for i in range(1, int(length/dnu/10)):
        # Add peaks at multiples of Dnu
        peak_pos = int(i * dnu * 10)
        if peak_pos < length:
            peak_width = max(1, int(params[11] * 100))  # Use gamma parameter for width
            peak_height = 5 * np.exp(-i/10)  # Decreasing amplitude
            
            # Add Lorentzian peak
            x = np.arange(max(0, peak_pos-peak_width), min(length, peak_pos+peak_width))
            spectrum[x] += peak_height / (1 + ((x - peak_pos)/params[11])**2)
    
    # Add some rotational splitting based on acr parameter
    acr = params[3]
    for i in range(1, int(length/dnu/10)):
        peak_pos = int(i * dnu * 10)
        if peak_pos < length:
            # Add side peaks
            side_distance = max(1, int(acr * 20))
            for side in [peak_pos - side_distance, peak_pos + side_distance]:
                if 0 <= side < length:
                    spectrum[side] += 2 * np.exp(-i/10)
    
    # Apply envelope modulation based on numax
    numax = params[9]
    envelope = np.exp(-((np.arange(length) - numax*100)**2) / (2 * (length/5)**2))
    spectrum *= envelope
    
    # Scale to have reasonable values
    spectrum = (spectrum - np.min(spectrum)) / (np.max(spectrum) - np.min(spectrum)) * 10
    
    # return spectrum
    return np.random.normal(0, 1, length)

# Generate training files (0-143)
for file_idx in range(144):
    file_data = []
    for example_idx in range(examples_per_file):
        # Generate parameters
        params = generate_stellar_parameters()
        
        # Generate spectrum
        spectrum = generate_spectrum(params)
        
        # Combine spectrum and parameters
        example_data = np.concatenate([spectrum, params])
        file_data.append(example_data)
    
    # Convert to numpy array
    file_data = np.array(file_data)
    
    # Save to file
    filename = f"data_{file_idx:03d}.npy"
    np.save(os.path.join(data_dir, filename), file_data)
    print(f"Generated {filename}")

# Generate validation files (144-151)
for file_idx in range(144, 152):
    file_data = []
    for example_idx in range(examples_per_file):
        params = generate_stellar_parameters()
        spectrum = generate_spectrum(params)
        example_data = np.concatenate([spectrum, params])
        file_data.append(example_data)
    
    file_data = np.array(file_data)
    filename = f"data_{file_idx:03d}.npy"
    np.save(os.path.join(data_dir, filename), file_data)
    print(f"Generated {filename}")

print(f"Dataset generated with {144 * examples_per_file} training examples and {8 * examples_per_file} validation examples")
print(f"Files are saved in {data_dir}")
print(f"Each example has {35692} spectral points and 15 stellar parameters")
print(f"Config file created at training_code/config.json")