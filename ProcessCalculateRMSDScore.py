import pandas as pd
from itertools import combinations
import os
import tempfile
import shutil
from multiprocessing import Pool
import time
import requests
from Bio.PDB import MMCIFParser, Superimposer
from Bio.PDB.Atom import Atom


# Parse an mmCIF file and extract a list of atoms
def parse_mmcif_as_atoms(file_path):
    try:
        parser = MMCIFParser(QUIET=True)
        structure = parser.get_structure("structure", file_path)
        atoms = [atom for model in structure for chain in model for residue in chain for atom in residue]
        return atoms
    except Exception as e:
        print(f"Error parsing mmCIF file {file_path}: {e}")
        return None


# Parse two mmCIF files, align structures, and calculate RMSD
def align_and_calculate_rmsd(mmcif_file1, mmcif_file2):
    atoms1 = parse_mmcif_as_atoms(mmcif_file1)
    atoms2 = parse_mmcif_as_atoms(mmcif_file2)

    if atoms1 is None or atoms2 is None:
        return None

    # Truncate larger set of atoms
    min_length = min(len(atoms1), len(atoms2))
    atoms1 = atoms1[:min_length]
    atoms2 = atoms2[:min_length]

    super_imposer = Superimposer()
    super_imposer.set_atoms(atoms1, atoms2)  # Works with lists of Atom objects
    return super_imposer.rms


# Download mmCIF file for a given UniProt ID
def download_mmCIF(uniprot_id, gene_name, output_dir="mmcif_files"):
    base_url = f"https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-model_v4.cif"
    output_path = os.path.join(output_dir, f"{gene_name}.cif")
    try:
        os.makedirs(output_dir, exist_ok=True)
        response = requests.get(base_url, stream=True)
        if response.status_code == 200:
            with open(output_path, "wb") as file:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)
            return output_path
        else:
            raise Exception(f"Download failed: {response.status_code} - {response.reason}")
    except Exception as e:
        print(f"Error downloading {uniprot_id}: {e}")
        return None


import pandas as pd
from itertools import combinations
import os
from multiprocessing import Pool
import time

def download_all_mmcif(uniprot_ids, mmcif_dir):
    """
    Download all mmCIF files for the given UniProt IDs into a specified directory.
    """
    os.makedirs(mmcif_dir, exist_ok=True)
    for gene_name, uniprot_id in uniprot_ids.items():
        mmcif_file = os.path.join(mmcif_dir, f"{gene_name}.cif")
        if not os.path.exists(mmcif_file):
            try:
                # Replace this with the actual function to download mmCIF files
                download_mmCIF(uniprot_id, gene_name, output_dir=mmcif_dir)
            except Exception as e:
                print(f"Failed to download mmCIF for {gene_name}: {e}")

def process_batch(batch_idx, batch_pairs, mmcif_dir, output_dir):
    """
    Process a single batch of gene pairs.
    """
    print(f"Processing batch {batch_idx}...")
    result_rows = []

    for file1, file2 in batch_pairs:
        try:
            mmcif_file1 = os.path.join(mmcif_dir, file1)
            mmcif_file2 = os.path.join(mmcif_dir, file2)

            # Extract gene names by removing the .cif extension
            gene1 = file1.rsplit('.', 1)[0]
            gene2 = file2.rsplit('.', 1)[0]

            # Calculate RMSD
            rmsd_score = align_and_calculate_rmsd(mmcif_file1, mmcif_file2)

            # Append results
            result_rows.append({
                "gene_name_1": gene1,
                "gene_name_2": gene2,
                "rmsd_score": rmsd_score
            })
        except Exception as e:
            result_rows.append({
                "gene_name_1": gene1,
                "gene_name_2": gene2,
                "rmsd_score": None
            })

    # Save batch results to CSV
    batch_file = os.path.join(output_dir, f"batch_{batch_idx}.csv")
    pd.DataFrame(result_rows).to_csv(batch_file, index=False)
    print(f"Finished processing batch {batch_idx}.")
    return batch_file


def process_batches_parallel(mmcif_dir, output_dir, batch_size=1000, num_workers=4):
    """
    Process gene pairs in batches using multiprocessing.
    """
    # Get the list of .cif files in the directory
    cif_files = [f for f in os.listdir(mmcif_dir) if f.endswith(".cif")]
    if not cif_files:
        raise FileNotFoundError("No .cif files found in the specified mmcif_dir.")

    # Generate all pairwise combinations
    file_pairs = list(combinations(cif_files, 2))
    num_batches = len(file_pairs) // batch_size + (len(file_pairs) % batch_size > 0)

    # Split file pairs into batches
    batches = [
        (i + 1, file_pairs[i * batch_size: (i + 1) * batch_size])
        for i in range(num_batches)
    ]
    print(f"Total number of batches: {len(batches)}")

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Parallel processing of batches
    start_time = time.time()
    with Pool(num_workers) as pool:
        results = pool.starmap(
            process_batch,
            [(batch_idx, batch_pairs, mmcif_dir, output_dir) for batch_idx, batch_pairs in batches]
        )

    total_time = time.time() - start_time
    print(f"Total processing time: {total_time / 60:.2f} minutes")

    # Return paths of saved batch files
    return results, total_time


def combine_batches(output_dir, final_output_file):
    """
    Combine all batch CSV files into a single final output CSV.
    """
    batch_files = [os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.startswith("batch_")]
    combined_df = pd.concat([pd.read_csv(batch_file) for batch_file in batch_files])
    combined_df.to_csv(final_output_file, index=False)
    print(f"Final results saved to {final_output_file}")



if __name__ == '__main__':
    # Example usage
    csv_path = "./final_genes.csv"
    uniprot_file = "./uniprot_ids.csv"
    mmcif_dir = "./mmcif_files"
    output_dir = "./batches"
    final_output_file = "./similarity_results.csv"

    #step 1: Load UniProt IDs
    if os.path.exists(uniprot_file):
        print("Loading UniProt IDs from saved file...")
        uniprot_ids = pd.read_csv(uniprot_file, nrows=12).set_index('gene_name')['uniprot_id'].to_dict()
    else:
        print("Error: UniProt file not found.")
        exit(1)

    # Step 2: Download all mmCIF files
    print("Downloading all mmCIF files...")
    download_all_mmcif(uniprot_ids, mmcif_dir)

    # Step 3: Process batches in parallel
    print("Starting parallel processing...")
    batch_results, total_time = process_batches_parallel(
        mmcif_dir, output_dir, batch_size=1000, num_workers=8
    )

    # Step 4: Combine batch results
    combine_batches(output_dir, final_output_file)


## Code to detect missed genes from mmcif download

# import pandas as pd
# import os

# # Load the first 333 rows of the CSV file
# csv_path = "final_genes.csv"  # Path to your final_gene.csv
# df = pd.read_csv(csv_path, nrows=333)  # Load only the first 333 rows
# gene_names = set(df['gene_name'])  # Extract gene names as a set

# # List all .cif files in the mmcif_files folder
# folder_path = "mmcif_files"  # Path to your mmcif_files folder
# cif_files = {os.path.splitext(file)[0] for file in os.listdir(folder_path) if file.endswith('.cif')}

# # Find gene names in the first 333 rows of the CSV but not in the folder
# missing_genes = gene_names - cif_files

# # Write the missing genes to a .txt file
# output_path = "missing_genes.txt"  # Output file path
# with open(output_path, "w") as f:
#     for gene in sorted(missing_genes):  # Sort for better readability
#         f.write(gene + "\n")

# print(f"Missing genes have been saved to {output_path}.")
