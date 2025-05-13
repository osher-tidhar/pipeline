import pandas as pd
from multiprocessing import Pool, set_start_method
import time
import cProfile
import pstats
import torch
import os
from functools import wraps


os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['CUDA_MPS_ACTIVE_THREAD_PERCENTAGE'] = '100'
try:
    set_start_method('spawn')
except RuntimeError:
    pass  # Ignore if already set.
torch.set_grad_enabled(False)
torch.inference_mode(True)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

"""## Default parameters for Jenna's alg"""
# AMI
l = 25
# FNN
MaxDim = 14
Rtol = 15
Atol = 2
speed = 1
# RQA
target_rec = 0.025
NUM_WORKERS = 1


def profile_function(func):
    """Profiles the given function and prints the results."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        profiler = cProfile.Profile()
        profiler.enable()
        result = func(*args, **kwargs)
        profiler.disable()
        with open("profiler_output.txt", "w") as f:
            stats = pstats.Stats(profiler, stream=f).sort_stats('tottime') # Sort by total time spent in the function can use 'cumulative'.
            stats.print_stats()
        return result
    return wrapper


def get_shifts(activity_len):
    if activity_len < 1:
        return [int(el * activity_len * 60 // 3) for el in range(3)]
    elif activity_len == 1:
        return [0, 20, 40]
    else:
        return [int(el * (activity_len / 6)) for el in [0, 150, 240]]

def jalg(input_agrs):
    pat, tstamp, acc_signal, ecg_signal, min_activity_len, sampling_rate = input_agrs

    # Convert signals to PyTorch tensors and move to GPU
    acc_signal = torch.tensor(acc_signal).to(device)
    ecg_signal = torch.tensor(ecg_signal).to(device)

    shifts = get_shifts(activity_len=min_activity_len)
    row = {'pat': pat, 'Record Time': tstamp}
    
    # given min_Activity_len more that 1 minute, adjust the duration of cuts!!!!!!
    # Create time window slices of both signals. For example, with min_activity_len=1: shifts = [0, 20, 40], sampling_rate = 128Hz first slice would be [0:2560] (0 to 20 seconds).
    cuts = [
            [acc_signal[shifts[0] * sampling_rate : shifts[1] * sampling_rate], 
             ecg_signal[shifts[0] * sampling_rate : shifts[1] * sampling_rate]],
            [acc_signal[shifts[1] * sampling_rate : shifts[2] * sampling_rate], 
             ecg_signal[shifts[1] * sampling_rate : shifts[2] * sampling_rate]],
            [acc_signal[shifts[2] * sampling_rate : ], 
             ecg_signal[shifts[2] * sampling_rate : ]],
        ]

    res_df = pd.DataFrame()
    num_features = 15  # Shift, EMB, DEL, RADIUS, Size, REC, DET, MeanL, MaxL, EntrL, LAM, MeanV, MaxV, EntrV, EntrW
    # Initialize output tensor on GPU
    output_tensor = torch.zeros((len(cuts), num_features), device=device)

    for i, (cut, shift) in enumerate(zip(cuts, shifts)):
        output_tensor[i, 1:] = full_RQA_run(cut[0], cut[1])
        output_tensor[i, 0] = shift
        
    # Transfer data to CPU and convert to DataFrame
    output_data = output_tensor.cpu().numpy()
    columns = ['Shift', 'EMB', 'DEL', 'RADIUS', 'Size', 'REC', 'DET', 'MeanL', 'MaxL', 'EntrL', 'LAM', 'MeanV', 'MaxV', 'EntrV', 'EntrW']
    res_df = pd.DataFrame(output_data, columns=columns)
    for i, shift in enumerate(shifts):
        row[f"shift_{shift}"] = output_tensor[i].cpu().numpy()

    avg_dict = res_df[['DET', 'MeanL', 'MaxL', 'EntrL','LAM', 'MeanV', 'MaxV', 'EntrV', 'EntrW']].mean().to_dict()
    std_dict = res_df[['DET', 'MeanL', 'MaxL', 'EntrL','LAM', 'MeanV', 'MaxV', 'EntrV', 'EntrW']].std().to_dict()

    avg_dict = {f'avg3points.{key}' : value for key, value in avg_dict.items()}
    std_dict = {f'std3points.{key}' : value for key, value in std_dict.items()}
    return {**row, **avg_dict, **std_dict}


def FNN(data, tau, MaxDim, Rtol, Atol, speed=0):
    """
    GPU optimized False Nearest Neighbors calculation.
    Estimates the embedding dimension of a time series using the False Nearest Neighbors (FNN) method.

    Args:
        data: A one-dimensional numpy array representing the time series.
        tau: Time delay for embedding.
        MaxDim: Maximum embedding dimension to consider.
        Rtol: Threshold for the relative distance change criterion.
        Atol: Threshold for the absolute distance criterion.
        speed: If 1, stop calculation once a minimum FNN is found. Otherwise, calculate up to MaxDim.

    Returns:
        dE: A numpy array containing the percentage of false nearest neighbors for each dimension.
        dim: The estimated optimal embedding dimension.
    """
    with torch.no_grad():
        # Process in smaller chunks
        CHUNK_SIZE = 500  # Adjust based on GPU memory

        # Convert tau to integer if it's a tensor
        tau_int = int(tau.item()) if torch.is_tensor(tau) else int(tau)
        if tau_int == 0:
            tau_int = 1  # Set minimum step size to 1
        
        n = len(data) - tau_int * MaxDim
        if n <= 0:
            return torch.zeros((MaxDim-1, 2), device=device), MaxDim
        
        rA = torch.std(data)
        x = data[:n]
        dE = torch.zeros((MaxDim-1, 2), device=device)

        for dim in range(2, MaxDim + 1):
            seq_len_int = tau_int * (dim - 1) + 1
            valid_len = len(x) - seq_len_int + 1
            
            if valid_len < 2:
                dE[dim-2] = torch.tensor([dim, float('inf')], device=device)
                continue

            # Convert to integers before using in arange
            valid_len_int = int(valid_len)

            # Create embedding matrix in chunks
            indices = (torch.arange(valid_len_int, device=device).unsqueeze(1) + 
                        torch.arange(0, seq_len_int, tau_int, device=device).unsqueeze(0))
            
            if indices.numel() == 0 or indices.max() >= len(x):
                dE[dim-2] = torch.tensor([dim, float('inf')], device=device)
                continue

            # Process sequences in chunks
            total_fnn = 0
            total_valid = 0
            sequences = x[indices]
            
            for i in range(0, len(sequences), CHUNK_SIZE):
                chunk = sequences[i:i + CHUNK_SIZE]
                
                # Calculate distances for chunk
                distances = torch.cdist(chunk.unsqueeze(0), sequences.unsqueeze(0), p=2)[0]
                torch.cuda.empty_cache()
                
                # Find nearest neighbors
                dists, neighs = torch.topk(distances, k=2, dim=1, largest=False)
                
                # Calculate FNN criteria for chunk
                future_idx = neighs[:, 1] + tau*dim
                valid_mask = future_idx < len(x)
                
                if valid_mask.any():
                    valid_indices = torch.arange(len(future_idx), device=device)[valid_mask]
                    deltas = torch.abs(x[future_idx[valid_mask]] - x[valid_indices])
                    criteria = ((deltas / dists[valid_mask, 1] > Rtol) | 
                                (torch.sqrt(dists[valid_mask, 1]**2 + deltas**2) / rA > Atol))
                    
                    total_fnn += criteria.float().sum()
                    total_valid += valid_mask.sum()
                
                torch.cuda.empty_cache()

            # Calculate percentage for entire set
            if total_valid > 0:
                fnn_percentage = (total_fnn / total_valid).item()
            else:
                fnn_percentage = float('inf')
            
            dE[dim-2] = torch.tensor([dim, fnn_percentage], device=device)

            if speed and (fnn_percentage == 0 or 
                (dim > 3 and dim < MaxDim and dE[dim-3, 1] > dE[dim-2, 1] < dE[dim-1, 1])):
                return dE, dim

        # Find optimal dimension
        mask = (dE[:-1, 1] == 0) | (torch.abs(dE[1:, 1] - dE[:-1, 1]) <= 0.001)
        best_dim = int(dE[mask.nonzero()[0][0], 0]) if mask.any() else MaxDim
        return dE, best_dim


def ami_thomas(x, max_lag):
    # Ensure input is a 1D tensor with sufficient length
    if x.ndim != 1:
        raise ValueError("Input time series must be a 1D array")
    if len(x) > 2000:
        x = x[:2000]

    # Calculate AMI for each lag
    n = len(x)
    ami = torch.zeros((max_lag, 2), device=device)

    for lag in range(max_lag):
        ami[lag, 0] = lag+1
        x1, x2 = x[:-(lag+1)], x[lag+1:]
        ami[lag, 1] = average_mutual_information(torch.column_stack((x1, x2)))

    # Find potential tau values (minimums in AMI)
    tau = []
    for i in range(1, len(ami)-1):
        if ami[i-1, 1] >= ami[i, 1] <= ami[i+1, 1]:
            tau.append(ami[i, :])

    # Find lag at 20% of initial AMI as additional potential tau
    condition = (ami[:, 1] <= 0.2 * ami[0, 1]).long()
    threshold_index = torch.argmax(condition) #.item()
    if threshold_index > 0:
        tau.append(ami[threshold_index, :])

    # Handle cases with no definitive minimum or insufficient max_lag
    if not tau:
        if max_lag * 1.5 > len(x) / 3:
            tau = [[torch.tensor(9999, device=device), torch.tensor(0, device=device)]]  # Return as list to maintain consistency
        else:
            tau, ami = ami_thomas(x, int(max_lag * 1.5))
    return tau, ami


def average_mutual_information(data):
    """GPU-optimized AMI calculation with proper resource management."""
    with torch.no_grad():  # Prevent gradient computation
        n = len(data)
        if n <= 1:
            return torch.tensor(0.0, device=device)
        
        # Extract x, y without creating new tensors
        x = data[:, 0].contiguous()  # Make memory contiguous
        y = data[:, 1].contiguous()
        
        # Calculate bandwidths using Scott's rule - reuse memory
        std_x = torch.std(x, unbiased=False)
        std_y = torch.std(y, unbiased=False)
        n_factor = n ** (-1/6)
        hx = std_x * n_factor
        hy = std_y * n_factor

        # Compute marginal and joint densities in batches to limit memory
        batch_size = min(1000, n)
        p_x = torch.zeros_like(x)
        p_y = torch.zeros_like(y)
        
        for i in range(0, n, batch_size):
            end = min(i + batch_size, n)
            p_x[i:end] = univariate_kernel_density(x[i:end], x, hx)
            p_y[i:end] = univariate_kernel_density(y[i:end], y, hy)
            torch.cuda.empty_cache()  # Clear unused memory

        joint_p_xy = bivariate_kernel_density(data, data, hx, hy)
        
        # Calculate AMI in place
        valid_mask = (joint_p_xy * p_x * p_y) > 0
        ratios = torch.zeros_like(joint_p_xy)
        ratios[valid_mask] = joint_p_xy[valid_mask] / (p_x[valid_mask] * p_y[valid_mask])
        log_values = torch.where(valid_mask, torch.log2(ratios), torch.zeros_like(ratios))
        return torch.mean(log_values)


def multivariate_normal_pdf(x, mean, cov):
    """GPU-optimized multivariate normal PDF calculation."""
    with torch.no_grad():
        d = x.shape[1]
        device = x.device

        # Add regularization to ensure positive definiteness
        min_eig = 1e-6
        diag_reg = torch.eye(d, device=device) * min_eig
        cov = cov + diag_reg

        try:
            # Try Cholesky decomposition first
            L = torch.linalg.cholesky(cov, upper=True)
            det_cov = torch.prod(torch.square(L.diagonal()))
            cov_inv = torch.cholesky_inverse(L)

        except RuntimeError:
            # If Cholesky fails, try eigendecomposition
            try:
                eigvals, eigvecs = torch.linalg.eigh(cov)
                
                # Ensure eigenvalues are positive
                eigvals = torch.clamp(eigvals, min=min_eig)
                
                # Reconstruct positive definite matrix
                cov = eigvecs @ torch.diag(eigvals) @ eigvecs.T
                det_cov = torch.prod(eigvals)
                cov_inv = eigvecs @ torch.diag(1.0 / eigvals) @ eigvecs.T

            except RuntimeError:
                # If all else fails, return zeros
                return torch.zeros(x.shape[0], device=device)

        # Compute PDF with stable computation
        try:
            norm_const = torch.rsqrt((2 * torch.pi) ** d * det_cov)
            x_diff = x - mean
            exp_term = -0.5 * torch.sum((x_diff @ cov_inv) * x_diff, dim=1)
            return torch.exp(torch.log(norm_const) + exp_term)  # More numerically stable
        except RuntimeError:
            return torch.zeros(x.shape[0], device=device)


def bivariate_kernel_density(values, data, bandwidth_x, bandwidth_y):
    """GPU-optimized bivariate kernel density estimation."""
    with torch.no_grad():
        # Get shapes and device
        n, _ = data.shape
        m, _ = values.shape

        # Early exit for empty tensors
        if n == 0 or m == 0:
            return torch.zeros(m, device=device)

        # Calculate correlation coefficient efficiently
        if n > 1:
            # Compute correlation without creating intermediate tensors
            data_centered = data - data.mean(dim=0, keepdim=True)
            data_std = torch.sqrt((data_centered ** 2).sum(dim=0))
            rho = (data_centered[:, 0] * data_centered[:, 1]).sum() / (data_std[0] * data_std[1])
        else:
            rho = torch.tensor(0.0, device=device)

        # Construct covariance matrix in-place
        cov_matrix = torch.zeros(2, 2, device=device)
        cov_matrix[0, 0] = bandwidth_x ** 2
        cov_matrix[1, 1] = bandwidth_y ** 2
        cov_matrix[0, 1] = cov_matrix[1, 0] = rho * bandwidth_x * bandwidth_y

        # Calculate differences efficiently
        differences = -linear_depth(values, -data)

        # Calculate density
        prob = multivariate_normal_pdf(differences, torch.zeros(2, device=device), cov_matrix)
        
        # Compute cumulative sum and density in-place
        cum_prob = torch.cumsum(prob, dim=0)
        density = torch.zeros(m, device=device)
        
        # First element
        density[0] = cum_prob[n-1] / n
        
        # Remaining elements using vectorized operations
        if m > 1:
            indices = torch.arange(1, m, device=device) * n
            density[1:] = (cum_prob[indices] - cum_prob[indices - n]) / n
        return density


def univariate_kernel_density(values, data, bandwidth, batch_size=100):
    values = values.float().to(device)
    data = data.float().to(device)

    n = len(data)
    m = len(values)

    density = torch.zeros(m, device=device)

    # Process in batches
    for i in range(0, m, batch_size):
        end = min(i + batch_size, m)
        batch_values = values[i:end]

        # Vectorized computation for each batch
        kernel_values = torch.exp(-0.5 * ((batch_values[:, None] - data[None, :]) / bandwidth) ** 2) / (bandwidth * torch.sqrt(torch.tensor(2 * torch.pi, device=device)))
        batch_density = torch.sum(kernel_values, dim=1) / n

        density[i:end] = batch_density
    return density


def linear_depth(a, b):
    m, d = a.shape
    n, _ = b.shape

    # Compute pairwise differences using broadcasting
    differences = a[:, None, :] + b[None, :, :]

    # Reshape to match the original output shape
    differences = differences.reshape(m * n, d)
    return differences


def full_RQA_run(acc_sig_tensor, ecg_sig_tensor):
    row = {}

    # AMI acc
    ttau, acc_ami = ami_thomas(acc_sig_tensor, l)
    acc_tau = ttau[0][0]
    acc_tau = acc_tau.long()

    # AMI ecg
    ttau, ami = ami_thomas(ecg_sig_tensor, l)

    # Find the index where the condition is met using pure PyTorch operations.
    threshold = 0.2 * ami[0, 1]
    condition_mask = ami[:, 1] <= threshold
    if condition_mask.any():
        ecg_tau_index = torch.nonzero(condition_mask)[0]
    else:
        # Use default value if no condition is met
        ecg_tau_index = torch.tensor(0, device=device)

    # Get the corresponding tau value
    ecg_tau = ami[ecg_tau_index, 0]
    ecg_tau = ecg_tau.long()

    # FNN acc
    _, acc_dim = FNN(data=acc_sig_tensor, tau=acc_tau, MaxDim=MaxDim, Rtol=Rtol, Atol=Atol, speed=speed)
    _, ecg_dim = FNN(data=ecg_sig_tensor, tau=ecg_tau, MaxDim=MaxDim, Rtol=Rtol, Atol=Atol, speed=speed)

    # RQA
    delay = torch.minimum(acc_tau, ecg_tau)
    emb = torch.maximum(torch.tensor(acc_dim, device=device), torch.tensor(ecg_dim, device=device))
    row['del'], row['emb'] = delay, emb
    rrow = RQA_calculation(acc_sig_tensor, ecg_sig_tensor, delay, emb, target_rec=target_rec)
    return torch.tensor([
        row['emb'],
        row['del'],
        rrow[1],  # radius
        rrow[0],  # size
        rrow[2],  # rec
        rrow[3],  # det
        rrow[4],  # mean_L
        rrow[5],  # max_L
        rrow[6],  # entr_L
        rrow[7],  # lam
        rrow[8],  # mean_V
        rrow[9],  # max_V
        rrow[10],  # entr_V
        rrow[11]  # entrW
    ], device=device)


def embed(x, tau, dim):
    '''embed = lambda x, tau, dim: np.array([x[i:i + tau*(dim - 1) + 1:tau] for i in range(len(x)) if len(x[i:i + tau*(dim - 1) + 1:tau]) == dim])
    '''
    # Initialize an empty list to store the embedded vectors
    embedded_vectors = []
    
    # Convert tau and dim to scalars if they are tensors
    if isinstance(tau, torch.Tensor):
        tau = tau.item()
    if isinstance(dim, torch.Tensor):
        dim = dim.item()

    # Iterate over the possible start indices
    for i in range(len(x) - tau * (dim - 1)):
        # Extract the slice with the specified delay and dimension
        indices = [i + j * tau for j in range(dim)]
        slice_vec = x[indices]
        
        # Check if the slice is not empty and has the correct length
        if len(slice_vec) == dim:
            # Append the slice to the list of embedded vectors
            embedded_vectors.append(slice_vec)
    
    # Check if embedded_vectors is not empty before stacking
    if embedded_vectors:
        embedded_tensor = torch.stack(embedded_vectors)
    else:
        # Handle the case where embedded_vectors is empty
        embedded_tensor = torch.empty((0, dim), device=x.device)
    return embedded_tensor


def custom_root_scalar(f, bracket, tol=1e-4, max_iter=100):
    """
    Find a value within the range [a, b] that satisfies a certain condition.
    
    Parameters:
    - f: The function to evaluate.
    - a, b: The range within which to search.
    - tol: The tolerance for convergence.
    - max_iter: The maximum number of iterations.
    
    Returns:
    - The value found by binary search.
    """
    a, b = bracket
    for _ in range(max_iter):
        c = (a + b) / 2
        fc = f(c)
        
        # Check if the condition is met
        if abs(fc) < tol:
            return c
        
        # Adjust the range based on the condition
        if fc > 0:
            b = c
        else:
            a = c
    
    # If max_iter is reached without convergence, return the midpoint
    return (a + b) / 2

def construct_func(distMat):
    """Constructs a function to calculate recurrence rate for a given radius."""
    distMat = -distMat
    n = len(distMat)  # Store the size for efficiency
    def find_rec_function(radius):
        """Calculates the recurrence rate for the given radius."""
        return torch.sum(distMat + radius >= 0) / (n * n)
    return find_rec_function

def calculate_metrics(counts):
    det = torch.sum(counts[counts > 1]) / torch.sum(counts) * 100
    mean = torch.mean(counts[counts > 1])
    mmax = torch.max(counts)
    
    # Calculate histogram and entropy
    #hist, _ = torch.histogram(counts[counts > 1].float(), bins=int(torch.max(counts[counts > 1])) + 1)
    #hist = torch.histc(counts[counts > 1], bins=int(torch.max(counts[counts > 1])) + 1)
    #entr = -1 * torch.sum(hist * torch.log2(hist) * (hist > 0))
    valid_counts = counts[counts > 1]
    if len(valid_counts) > 0:
        max_count = int(torch.max(valid_counts))
        hist = torch.bincount(valid_counts.long(), minlength=max_count + 1)
        hist = hist[hist > 0]  # Remove zero counts
        entr = -torch.sum(hist * torch.log2(hist))
    else:
        hist = torch.tensor([0.0], device=device)  # Handle empty case
        entr = torch.tensor([0.0], device=device)    
    return det, mean, mmax, entr

def extract_diagonals(bin_mat, d_ind):
    if d_ind >= 0:
        diag_len = len(bin_mat) - d_ind
        diag = bin_mat[torch.arange(diag_len), torch.arange(diag_len) + d_ind]
    else:
        diag_len = len(bin_mat) + d_ind
        diag = bin_mat[torch.arange(diag_len) + abs(d_ind), torch.arange(diag_len)]
    return diag

def process_diagonal(bin_mat):
    max_len = len(bin_mat)
    
    # Initialize counts on the correct device
    counts = torch.tensor([], dtype=torch.int64, device=device)
    
    # Loop through diagonals
    for d_ind in range(-max_len, max_len):
        diag = extract_diagonals(bin_mat, d_ind)
        # Count consecutive ones directly on the tensor
        unique, counts = torch.unique_consecutive(diag, return_counts=True)
        consecutive_ones = counts[unique == 1]
        counts = torch.cat((counts, consecutive_ones))
    
    # Calculate metrics using vectorized operations
    det = torch.sum(counts[counts > 1]) / torch.sum(counts) * 100
    mean = torch.mean(counts[counts > 1].float())
    mmax = torch.max(counts)
    
    # Calculate histogram and entropy
    valid_counts = counts[counts > 1]
    if len(valid_counts) > 0:
        max_count = int(torch.max(valid_counts))
        hist = torch.bincount(valid_counts.long(), minlength=max_count + 1)
        hist = hist[hist > 0]  # Remove zero counts
        entr = -torch.sum(hist * torch.log2(hist))
    else:
        hist = torch.tensor([0.0], device=device)  # Handle empty case
        entr = torch.tensor([0.0], device=device)
    return det, mean, mmax, entr


def process_vertical(bin_mat):
    counts = []
    for v_ind in range(len(bin_mat)):
        tensor = bin_mat[:, v_ind]
        counts.extend(count_consecutive_ones(tensor))
    counts = torch.tensor(counts, dtype=torch.float32, device=device)
    return calculate_metrics(counts)


def RQA_calculation(acc_sig, ecg_sig, delay, emb, target_rec=0.025):
    """GPU-optimized RQA calculation."""
    with torch.no_grad():
        # Calculate stats
        acc_mean, acc_std = acc_sig.mean(), acc_sig.std()
        ecg_mean, ecg_std = ecg_sig.mean(), ecg_sig.std()
        
        # Normalize in-place
        acc_sig.sub_(acc_mean).div_(acc_std + 1e-8)
        ecg_sig.sub_(ecg_mean).div_(ecg_std + 1e-8)

        # Embedding with minimal allocation
        acc_emb = embed(acc_sig, delay, emb)
        ecg_emb = embed(ecg_sig, delay, emb)

        size = acc_emb.size(0)

        # Calculate distance matrix efficiently
        distMat = torch.cdist(acc_emb, ecg_emb)
        max_dist = distMat.max(dim=0)[0] if distMat.numel() > 0 else torch.tensor(float('-inf'))

        # Binary search for radius
        rec_function = lambda r: (distMat <= r).float().mean()
        radius = custom_root_scalar(
            lambda radius: rec_function(radius) - target_rec,
            bracket=[0, torch.max(distMat)],
            tol=10**(-4)
        )
        
        # Calculate metrics in-place
        bin_mat = (distMat <= radius).float()
        distMat = None  # Free memory

        # Store basic metrics
        rec = bin_mat.mean().item() * 100

        # Process diagonal lines
        det, mean_L, max_L, entr_L = process_diagonal(bin_mat)
        
        # Process vertical lines
        lam, mean_V, max_V, entr_V = process_vertical(bin_mat)
        
        # Calculate weighted entropy (reuse bin_mat)
        entrW = rqa_weighted_entropy(-bin_mat)

    variables = [size, radius, rec, det, mean_L, max_L, entr_L, lam, mean_V, max_V, entr_V, entrW]
    return torch.tensor(variables, device=device)

def rqa_weighted_entropy(wrp):
    """Calculates the weighted entropy of the weighted recurrence plot GPU-optimized."""
    with torch.no_grad():
        # Sum along axis in one operation
        si = torch.sum(wrp, dim=0)
        
        # Get min/max in one operation
        mi, ma = torch.aminmax(si)

        # Early exit if invalid values
        if torch.isnan(mi) or torch.isnan(ma):
            return 0.0
        
        # Calculate bin width
        m = torch.clamp((ma - mi) / 49, min=1.0)
        
        # Create bins efficiently
        bins = torch.arange(mi, ma + m/2, m, device=device)
        
        # Calculate probabilities using vectorized operations
        S = si.sum()
        counts = torch.stack([(si >= s) & (si < (s + m)) for s in bins])
        P = torch.sum(si.unsqueeze(0) * counts, dim=1) / S
        
        # Calculate entropy directly
        return -torch.sum(P * torch.log(P) * (P > 0))


def count_consecutive_ones(tensor):
    unique, counts = torch.unique_consecutive(tensor, return_counts=True)
    return counts[unique == 1]


"""# Run RQA analysis in Parallel"""
def get_day_RQA(signal_cuts, min_activity_len, ecg_sampling_rate):
    alg_ins = [[pat, tstamp, acc, ecg, min_activity_len, ecg_sampling_rate] 
               for pat, tstamp, acc, ecg in signal_cuts[['pat', 'Record Time', 'acc', 'ecg']].values]
    
    # Limit number of workers to CPU count
    n_workers = NUM_WORKERS #min(len(alg_ins), os.cpu_count())
    with Pool(processes=n_workers) as pool:
        try:
            outs = pool.map(jalg, alg_ins[:])
        except Exception as e:
            print(f"Pool execution failed: {str(e)}")
            raise  
    return signal_cuts.merge(pd.DataFrame(outs), on=['pat', 'Record Time'])
