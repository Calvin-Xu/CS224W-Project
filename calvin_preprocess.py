import os
import pickle

if os.path.exists("graph_w_covid_genes_treatments_string_cif_coalesced_6.pkl"):
    G = pickle.load(
        open("graph_w_covid_genes_treatments_string_cif_coalesced_6.pkl", "rb")
    )
else:
    print("Graph file not found!")

import numpy as np

# Identify the treatment relation edges and collect all treat_type values
treat_type_set = set()
for u, v, data in G.edges(data=True):
    rel = data.get("relation", None)
    if rel == "DRKG::Treats::Compound:Disease":
        tt = data.get("treat_type", None)
        if tt is not None:
            treat_type_set.add(tt)

# Create a one-hot encoding mapping for treat_type
treat_type_list = list(treat_type_set)
treat_type_to_idx = {tt: i for i, tt in enumerate(treat_type_list)}
num_treat_types = len(treat_type_list)

# Apply one-hot encoding to treatment edges
for u, v, data in G.edges(data=True):
    rel = data.get("relation", None)
    if rel == "DRKG::Treats::Compound:Disease":
        tt = data.get("treat_type", None)
        one_hot = np.zeros(num_treat_types, dtype=np.float32)
        if tt is not None:
            one_hot[treat_type_to_idx[tt]] = 1.0
        data["treat_type_onehot"] = one_hot
    else:
        # For non-treatment edges, no treat_type_onehot
        if "treat_type_onehot" in data:
            del data["treat_type_onehot"]

# Identify gene-gene edges and their numeric attributes
gene_gene_attributes = [
    "coexpression",
    "experimentally_determined_interaction",
    "automated_textmining",
    "combined_score",
    "rmsd_score",
]
numeric_values = {attr: [] for attr in gene_gene_attributes}

# Collect numeric values for normalization (ignoring 'N/A')
for u, v, data in G.edges(data=True):
    # Check if edge is gene-gene
    if "Gene::" in u and "Gene::" in v:
        for attr in gene_gene_attributes:
            val = data.get(attr, "N/A")
            if val != "N/A":
                try:
                    numeric_values[attr].append(float(val))
                except ValueError:
                    # In case non-numeric sneaks in
                    pass

# Compute mean and std for each attribute for z-score normalization
attr_stats = {}
for attr in gene_gene_attributes:
    arr = np.array(numeric_values[attr], dtype=np.float32)
    if len(arr) > 0:
        mean_val = np.mean(arr)
        std_val = np.std(arr)
        # If std is zero, avoid division by zero
        if std_val == 0:
            std_val = 1e-9
    else:
        # If no valid values, use neutral stats
        mean_val = 0.0
        std_val = 1.0
    attr_stats[attr] = (mean_val, std_val)

# Normalize and mask gene-gene attributes
for u, v, data in G.edges(data=True):
    if "Gene::" in u and "Gene::" in v:
        feature_vector = []
        mask_vector = []
        for attr in gene_gene_attributes:
            val = data.get(attr, "N/A")
            if val == "N/A":
                # Missing value: use 0 and mask 1
                feature_vector.append(0.0)
                mask_vector.append(1.0)
            else:
                # Convert to float and z-score normalize
                fval = float(val)
                mean_val, std_val = attr_stats[attr]
                norm_val = (fval - mean_val) / std_val
                feature_vector.append(norm_val)
                mask_vector.append(0.0)

        data["gene_gene_features"] = np.array(feature_vector, dtype=np.float32)
        data["gene_gene_mask"] = np.array(mask_vector, dtype=np.float32)
    else:
        # If not gene-gene, remove if previously set
        if "gene_gene_features" in data:
            del data["gene_gene_features"]
        if "gene_gene_mask" in data:
            del data["gene_gene_mask"]

# At this point, G has edges with:
# - 'treat_type_onehot' for compound-disease treatment edges
# - 'gene_gene_features' and 'gene_gene_mask' for gene-gene edges

pickle.dump(G, open("ml_graph.pkl", "wb"))
