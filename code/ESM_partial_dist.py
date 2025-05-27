import sys, os, pickle, torch, math
import numpy as np
from scipy.spatial.distance import cdist, cosine

# -------------------- Parse inputs --------------------
if len(sys.argv) != 3:
    print("Usage: python compute_partial_matrix.py <distance_type> <array_index>")
    sys.exit(1)

exp_type = sys.argv[1]         # one of 'ha_based', 'mean', 'max', 'cls'
array_idx = int(sys.argv[2])   # row block index

# -------------------- Load input --------------------
with open('../data/family_prot.pkl', 'rb') as file:
    fam_prot = pickle.load(file)
with open('../data/protein_highAttnSite.pkl', 'rb') as file:
    pid_HA = pickle.load(file)

# Clean families
fam_prot_cleaned = {
    k: v for k, v in fam_prot.items()
    if k and isinstance(k, str) and k.strip() and k != 'nan' and len(v) >= 2
}

# List of all proteins
protein_set = {p for fam in fam_prot_cleaned.values() for p in fam}
protein_list = sorted([p for p in protein_set if p in pid_HA and os.path.exists(f"../data/heatmap/{p}.npy")])
N = len(protein_list)
print(f"Total proteins: {N}")

# -------------------- Partition block --------------------
block_size = 500
start = array_idx * block_size
end = min(N, (array_idx + 1) * block_size)
block_indices = range(start, end)

# -------------------- Helper functions --------------------
def get_HA_vectors(prot):
    ha_list = pid_HA.get(prot, [])
    heatmap_path = f"../data/heatmap/{prot}.npy"
    if not os.path.exists(heatmap_path): return []
    heatmap = np.load(heatmap_path)
    return [heatmap[:, s] for s in ha_list]

def get_repr_vec(prot, mode, llm_output_dir):
    ptfile = f"{llm_output_dir}/representation_matrices/{prot}.pt"
    if not os.path.exists(ptfile): return None
    x = torch.load(ptfile)
    if mode == 'cls': return x[0, 1, :].numpy()
    emb = x[0, 1:-1, :].numpy()
    return np.mean(emb, axis=0) if mode == 'mean' else np.max(emb, axis=0)

# -------------------- Compute block --------------------
llm_output_dir = '../data/llm_data'
block = np.zeros((len(block_indices), N))
for idx_i, i in enumerate(block_indices):
    prot_i = protein_list[i]
    if exp_type == 'ha_based':
        vecs_i = get_HA_vectors(prot_i)
        for j, prot_j in enumerate(protein_list):
            vecs_j = get_HA_vectors(prot_j)
            if not vecs_i or not vecs_j:
                block[idx_i, j] = np.nan
                continue
            if i == j:
                block[idx_i, j] = 0.0
            else:
                dist = cdist(vecs_i, vecs_j, metric='cosine')
                
    else:
        vec_i = get_repr_vec(prot_i, exp_type, llm_output_dir)
        for j, prot_j in enumerate(protein_list):

            vec_j = get_repr_vec(prot_j, exp_type, llm_output_dir)
            if vec_i is None or vec_j is None:
                block[idx_i, j] = np.nan
                continue
            if i == j:
                dist = 0.0
            else:
                dist = cosine(vec_i, vec_j)
            block[idx_i, j] = dist
    
# -------------------- Verify diagonal --------------------
row_indices = list(block_indices)
diagonal_values = [block[idx_i, i] for idx_i, i in enumerate(row_indices) if i < N]

nonzero_diag = [(i, val) for i, val in zip(row_indices, diagonal_values) if not np.isclose(val, 0.0, atol=1e-6)]

if nonzero_diag:
    print(f"[WARNING] Non-zero diagonal values in block {array_idx}:")
    for i, val in nonzero_diag:
        print(f"  distance_matrix[{i}, {i}] = {val}")

# -------------------- Save block --------------------
os.makedirs(f"../data/distance_blocks/{exp_type}", exist_ok=True)
np.save(f"../data/distance_blocks/{exp_type}/block_{array_idx}.npy", block)
print(f"Saved: distance_blocks/{exp_type}/block_{array_idx}.npy")