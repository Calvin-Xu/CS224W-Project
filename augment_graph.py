import os
import pickle
import networkx as nx
import openpyxl
import pandas as pd
from tqdm import tqdm


def load_drkg():
    # Load the DRKG relationships file
    drkg_df = pd.read_csv(
        "drkg.tar/drkg.tsv",
        sep="\t",
        header=None,
        names=["source", "relation", "target"],
    )

    G = nx.DiGraph()

    # Add edges from the dataframe
    # Each row represents a relationship between source and target
    for _, row in tqdm(drkg_df.iterrows(), total=len(drkg_df), desc="Building graph"):
        G.add_edge(row["source"], row["target"], relation=row["relation"])

    return G


# Load and process host genes files
def load_host_genes():
    # Load COVID-19 host genes
    covid_genes_df = pd.read_csv("covid19-host-genes.tsv", sep="\t", header=None)
    host_genes = covid_genes_df[2].tolist()  # Get third column

    # Load coronavirus-related host genes
    related_genes_df = pd.read_csv(
        "coronavirus-related-host-genes.tsv", sep="\t", header=None
    )
    related_host_genes = related_genes_df[2].tolist()  # Get third column

    return host_genes, related_host_genes


def get_gene_neighbors(G, genes, max_hops=2):
    first_hop = set()
    second_hop = set()

    for gene in genes:
        if gene in G:
            # Get first hop neighbors
            neighbors_1 = set(
                n
                for n in G.neighbors(gene)
                #    if n.startswith("Gene::")
            )
            first_hop.update(neighbors_1)

            # Get second hop neighbors
            for neighbor in neighbors_1:
                second_hop.update(
                    n
                    for n in G.neighbors(neighbor)
                    #   if n.startswith("Gene::")
                )

    # Remove original genes and first hop genes from second hop
    second_hop = second_hop - set(genes) - first_hop
    first_hop = first_hop - set(genes)  # Remove original genes from first hop

    return list(first_hop), list(second_hop)


def prepare_gene_neighbors_csv(
    graph, host_genes, related_host_genes, update_graph=False
):
    """
    Prepare gene neighbors data and optionally update graph node attributes.

    Args:
        graph: NetworkX graph containing DRKG data
        host_genes: List of COVID-19 host genes
        related_host_genes: List of coronavirus-related host genes
        update_graph: Boolean, whether to update graph node attributes (default: False)

    Returns:
        DataFrame containing gene neighbor information
    """
    # Convert gene lists to sets for faster lookup
    host_genes_set = set(host_genes)
    related_host_genes_set = set(related_host_genes)

    # Get gene-only neighbors for both gene sets
    covid_first_hop, covid_second_hop = get_gene_neighbors(graph, host_genes)
    related_first_hop, related_second_hop = get_gene_neighbors(
        graph, related_host_genes
    )

    # Convert to sets for faster lookup and remove any overlap with seed genes
    covid_first_hop_set = set(covid_first_hop) - host_genes_set - related_host_genes_set
    covid_second_hop_set = (
        set(covid_second_hop)
        - host_genes_set
        - related_host_genes_set
        - covid_first_hop_set
    )
    related_first_hop_set = (
        set(related_first_hop) - host_genes_set - related_host_genes_set
    )
    related_second_hop_set = (
        set(related_second_hop)
        - host_genes_set
        - related_host_genes_set
        - related_first_hop_set
    )

    # Prepare data for CSV and update graph attributes
    data = []

    # Add COVID-19 host genes (seeds)
    for gene in host_genes_set:
        data.append(
            {"source_gene": gene, "gene_set": "COVID-19", "neighbor_type": "seed"}
        )
        if update_graph and gene in graph:
            graph.nodes[gene].update({"gene_set": "COVID-19", "neighbor_type": "seed"})

    # Add coronavirus-related host genes (seeds)
    for gene in related_host_genes_set:
        data.append(
            {
                "source_gene": gene,
                "gene_set": "coronavirus-related",
                "neighbor_type": "seed",
            }
        )
        if update_graph and gene in graph:
            graph.nodes[gene].update(
                {"gene_set": "coronavirus-related", "neighbor_type": "seed"}
            )

    # Add first-hop neighbors
    for gene in covid_first_hop_set:
        data.append(
            {"source_gene": gene, "gene_set": "COVID-19", "neighbor_type": "first_hop"}
        )
        if update_graph and gene in graph:
            graph.nodes[gene].update(
                {"gene_set": "COVID-19", "neighbor_type": "first_hop"}
            )

    for gene in related_first_hop_set:
        data.append(
            {
                "source_gene": gene,
                "gene_set": "coronavirus-related",
                "neighbor_type": "first_hop",
            }
        )
        if update_graph and gene in graph:
            graph.nodes[gene].update(
                {"gene_set": "coronavirus-related", "neighbor_type": "first_hop"}
            )

    # Add second-hop neighbors
    for gene in covid_second_hop_set:
        data.append(
            {"source_gene": gene, "gene_set": "COVID-19", "neighbor_type": "second_hop"}
        )
        if update_graph and gene in graph:
            graph.nodes[gene].update(
                {"gene_set": "COVID-19", "neighbor_type": "second_hop"}
            )

    for gene in related_second_hop_set:
        data.append(
            {
                "source_gene": gene,
                "gene_set": "coronavirus-related",
                "neighbor_type": "second_hop",
            }
        )
        if update_graph and gene in graph:
            graph.nodes[gene].update(
                {"gene_set": "coronavirus-related", "neighbor_type": "second_hop"}
            )

    # Convert to DataFrame and save to CSV
    df = pd.DataFrame(data)
    df.to_csv("gene_neighbors.csv", index=False)

    return df


def process_treatment_db(excel_path):
    # Load Excel file with multiple sheets
    excel_file = pd.ExcelFile(excel_path, engine="openpyxl")
    wb = openpyxl.load_workbook(excel_path)

    results = {}

    # Load and process each sheet
    for sheet_name in excel_file.sheet_names:
        df = pd.read_excel(excel_file, sheet_name=sheet_name)
        ws = wb[sheet_name]

        # Create a dictionary to map row numbers to hyperlinks
        row_hyperlinks = {}
        for row in ws.iter_rows(min_row=2):  # Start from second row (after header)
            for cell in row:
                if cell.hyperlink:
                    # Excel rows are 1-based, pandas is 0-based, and we skip header
                    row_idx = (
                        cell.row - 2
                    )  # Subtract 2 to account for header and 0-based indexing
                    row_hyperlinks[row_idx] = cell.hyperlink.target
                    break

        # Add hyperlinks column to DataFrame
        df["DB_ID"] = df.index.map(lambda x: row_hyperlinks.get(x, "").split("/")[-1])

        # Save to CSV
        df.to_csv(f"covid-treatments-db/{sheet_name}.csv", index=False)
        results[sheet_name] = df

    wb.close()
    return results


def add_treatment_attributes_to_graph(G, treatment_dfs):
    """
    Add treatment database information as node attributes to the graph.

    Args:
        G: NetworkX graph containing DRKG data
        treatment_dfs: Dictionary of DataFrames from process_treatment_db()

    Returns:
        List of drugs found in both treatment DB and DRKG
    """
    drugs_in_drkg = set()

    for sheet_name, df in treatment_dfs.items():
        # Skip rows with empty DB_ID
        df = df.dropna(subset=["DB_ID"])

        for _, row in df.iterrows():
            # Construct DRKG drug ID format
            drug_id = f"Compound::{row['DB_ID']}"

            if drug_id in G:
                drugs_in_drkg.add(drug_id)

                # Add all fields except DB_ID as node attributes
                attributes = row.drop("DB_ID").to_dict()
                # Clean up attributes (remove NaN values)
                attributes = {k: v for k, v in attributes.items() if pd.notna(v)}

                # Initialize or update nested dictionary structure
                if "treatment_data" not in G.nodes[drug_id]:
                    G.nodes[drug_id]["treatment_data"] = {}
                G.nodes[drug_id]["treatment_data"][sheet_name] = attributes

    # Write found drugs to file
    drugs_list = sorted(list(drugs_in_drkg))
    with open("drugs_in_drkg.txt", "w") as f:
        for drug in drugs_list:
            f.write(f"{drug}\n")

    print(f"Found {len(drugs_in_drkg)} drugs in DRKG")
    return drugs_list


def add_edge_attributes(
    G, gene_mapping_path="../string-data-full/final_genes.csv", edge_data_paths=None
):
    """
    Add edge attributes from CSV/TSV files to the graph.

    Args:
        G: NetworkX graph to augment
        gene_mapping_path: Path to CSV with source_gene and gene_name columns for ID mapping
        edge_data_paths: List of paths to CSV/TSV files or directories. For directories,
                        all CSV and TSV files within will be processed recursively.

    Returns:
        The modified graph with new edge attributes
    """
    # Load gene mapping
    gene_mapping = pd.read_csv(gene_mapping_path)
    # Create a dictionary for quick lookup
    name_to_id = dict(zip(gene_mapping["gene_name"], gene_mapping["source_gene"]))

    print(f"Loaded {len(name_to_id)} gene mappings")

    if edge_data_paths is None:
        return G

    # Valid file extensions and their corresponding separators
    valid_extensions = {".csv": ",", ".tsv": "\t"}

    # Collect all CSV/TSV files from the provided paths
    data_files = []
    for path in edge_data_paths:
        if os.path.isfile(path):
            ext = os.path.splitext(path.lower())[1]
            if ext in valid_extensions:
                data_files.append(path)
        elif os.path.isdir(path):
            # Walk through directory recursively
            for root, _, files in os.walk(path):
                for file in files:
                    ext = os.path.splitext(file.lower())[1]
                    if ext in valid_extensions:
                        data_files.append(os.path.join(root, file))

    total_interactions = 0

    # Process each data file
    for path in data_files:
        try:
            # Determine the separator based on file extension
            ext = os.path.splitext(path.lower())[1]
            separator = valid_extensions[ext]

            edge_data = pd.read_csv(path, sep=separator)

            # Check if file has required columns
            required_cols = {"#node1", "node2"}
            if not required_cols.issubset(edge_data.columns):
                print(
                    f"Skipping {path}: Missing required columns {required_cols - set(edge_data.columns)}"
                )
                continue

            # Add edges and their attributes to the graph
            for _, row in tqdm(
                edge_data.iterrows(), total=len(edge_data), desc=f"Processing {path}"
            ):
                # Get the corresponding gene names from the mapping
                source = name_to_id.get(row["#node1"])
                target = name_to_id.get(row["node2"])

                if source is not None and target is not None:
                    # Add or update edge attributes
                    edge_attrs = {
                        "coexpression": row.get("coexpression", "N/A"),
                        "experimentally_determined_interaction": row.get(
                            "experimentally_determined_interaction", "N/A"
                        ),
                        "automated_textmining": row.get("automated_textmining", "N/A"),
                        "combined_score": row.get("combined_score", "N/A"),
                        "rmsd_score": row.get("rmsd_score", "N/A"),
                    }

                    # Add edge if it doesn't exist, or update attributes if it does
                    if G.has_edge(source, target):
                        G[source][target].update(edge_attrs)
                        total_interactions += 1
                    else:
                        G.add_edge(source, target, **edge_attrs)
                        total_interactions += 1
        except Exception as e:
            print(f"Error processing {path}: {str(e)}")
            continue

    print(f"Added {total_interactions} interactions")
    return G


def coalesce_relations(graph, relations_to_coalesce, new_relation_type):
    """
    Coalesce specified relation types into a new relation type and add the old relation type as an edge attribute.

    Args:
        graph: NetworkX graph containing DRKG data
        relations_to_coalesce: List of relation types to coalesce
        new_relation_type: The new relation type to assign to coalesced edges (default: "DRKG::Treats::Compound:Disease")
    """
    total_coalesced = 0
    for u, v, data in graph.edges(data=True):
        if "relation" in data and data["relation"] in relations_to_coalesce:
            # Store the old relation type as an edge attribute
            old_relation = data["relation"]
            data["treat_type"] = old_relation

            # Update the relation type to the new one
            data["relation"] = new_relation_type
            total_coalesced += 1

    print(f"Coalesced {total_coalesced} edges")
    return graph


if __name__ == "__main__":
    # Load or create graph
    if os.path.exists("graph.pkl"):
        graph = pickle.load(open("graph.pkl", "rb"))
    else:
        graph = load_drkg()
        print(
            f"Graph has {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges"
        )
        pickle.dump(graph, open("graph.pkl", "wb"))

    # # Load and print host genes
    host_genes, related_host_genes = load_host_genes()
    print(f"\nNumber of COVID-19 host genes: {len(host_genes)}")
    print(f"First few host genes: {host_genes[:5]}")
    print(f"\nNumber of coronavirus-related host genes: {len(related_host_genes)}")
    print(f"First few related host genes: {related_host_genes[:5]}")

    # Process gene neighbors
    neighbors_df = prepare_gene_neighbors_csv(
        graph, host_genes, related_host_genes, update_graph=True
    )
    print("\nResults written to gene_neighbors.csv")
    print("\nSummary statistics:")
    print(neighbors_df.groupby(["gene_set", "neighbor_type"]).size())

    # Process treatment database
    treatment_dfs = process_treatment_db(
        "covid-treatments-db/COVID19-Treatment-DB.xlsx"
    )

    # Add treatment information to graph
    drugs_found = add_treatment_attributes_to_graph(graph, treatment_dfs)
    print("\nFirst 5 drugs found in DRKG:")
    print("\n".join(f"{drug}: {graph.nodes[drug]}" for drug in drugs_found[:5]))

    # In main:
    print("\nChecking gene attributes:")
    print("\n".join(f"{gene}: {graph.nodes[gene]}" for gene in host_genes[:3]))

    if not os.path.exists("graph_w_covid_genes_treatments.pkl"):
        pickle.dump(graph, open("graph_w_covid_genes_treatments.pkl", "wb"))

    # Add STRING edge data
    graph = add_edge_attributes(
        graph,
        edge_data_paths=[
            "../string-data-full/string-interaction-data",
            "../string-data-full/similarity_results.csv",
        ],
    )

    print("\nChecking example edge attributes:\n")
    print(f"Gene::6434 -> Gene::27429: {graph['Gene::6434']['Gene::27429']}")
    print(f"Gene::27429 -> Gene::6434: {graph['Gene::27429']['Gene::6434']}")
    print(f"Gene::26092 -> Gene::5577: {graph['Gene::26092']['Gene::5577']}")

    COV_disease_list = [
        "Disease::SARS-CoV2 E",
        "Disease::SARS-CoV2 M",
        "Disease::SARS-CoV2 N",
        "Disease::SARS-CoV2 Spike",
        "Disease::SARS-CoV2 nsp1",
        "Disease::SARS-CoV2 nsp10",
        "Disease::SARS-CoV2 nsp11",
        "Disease::SARS-CoV2 nsp12",
        "Disease::SARS-CoV2 nsp13",
        "Disease::SARS-CoV2 nsp14",
        "Disease::SARS-CoV2 nsp15",
        "Disease::SARS-CoV2 nsp2",
        "Disease::SARS-CoV2 nsp4",
        "Disease::SARS-CoV2 nsp5",
        "Disease::SARS-CoV2 nsp5_C145A",
        "Disease::SARS-CoV2 nsp6",
        "Disease::SARS-CoV2 nsp7",
        "Disease::SARS-CoV2 nsp8",
        "Disease::SARS-CoV2 nsp9",
        "Disease::SARS-CoV2 orf10",
        "Disease::SARS-CoV2 orf3a",
        "Disease::SARS-CoV2 orf3b",
        "Disease::SARS-CoV2 orf6",
        "Disease::SARS-CoV2 orf7a",
        "Disease::SARS-CoV2 orf8",
        "Disease::SARS-CoV2 orf9b",
        "Disease::SARS-CoV2 orf9c",
        "Disease::MESH:D045169",
        "Disease::MESH:D045473",
        "Disease::MESH:D001351",
        "Disease::MESH:D065207",
        "Disease::MESH:D028941",
        "Disease::MESH:D058957",
        "Disease::MESH:D006517",
    ]

    treatment_relations_list = [
        "GNBR::T::Compound:Disease",  # Treatment/therapy relationship (including investigatory use)
        "Hetionet::CtD::Compound:Disease",  # Direct treatment relationship between compound and disease
        "GNBR::Pa::Compound:Disease",  # Compound alleviates or reduces disease symptoms
        "GNBR::Pr::Compound:Disease",  # Compound prevents or suppresses disease
        "DRUGBANK::treats::Compound:Disease",  # Established treatment relationship from DrugBank database
        "GNBR::C::Compound:Disease",  # Compound inhibits disease-related cell growth (especially in cancers)
        "Hetionet::CpD::Compound:Disease",  # Palliative treatment relationship (focuses on symptom relief)
    ]

    graph = coalesce_relations(
        graph, treatment_relations_list, "DRKG::Treats::Compound:Disease"
    )

    print(f"Checking example edge attributes:\n")
    print(
        f"Edge from DB00004 to MESH:C063419: {graph.get_edge_data('Compound::DB00004', 'Disease::MESH:C063419')}"
    )
    print(
        f"Edge from MESH:C004656 to MESH:C537014: {graph.get_edge_data('Compound::MESH:C004656', 'Disease::MESH:C537014')}"
    )
    print(
        f"Edge from DB00997 to DOID:363: {graph.get_edge_data('Compound::DB00997', 'Disease::DOID:363')}\n"
    )

    pickle.dump(
        graph,
        open("graph_w_covid_genes_treatments_string_cif_coalesced_6.pkl", "wb"),
    )
