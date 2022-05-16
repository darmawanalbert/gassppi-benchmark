"""**Import Libraries, Global Variables**"""

import subprocess
import re
import numpy as np
import random
import json
from Bio.PDB import *
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score, matthews_corrcoef, confusion_matrix
from Bio.PDB.SASA import ShrakeRupley
from Bio.PDB.ResidueDepth import ResidueDepth
import warnings
import time
import math

start_time = time.time()

random.seed(10)

# List of directory paths
repository_path = "/root/gassppi-benchmark/"
dbd5_path = repository_path + "dbd5/"
tmalign_path = repository_path + "TMalign"
templates_path = repository_path + "templates/"
pymol_path = repository_path + "pymol/"

# Biopython-related objects
pdb_parser = PDBParser()

# Last Heavy Atom mapping
lha_dict = {
    "GLY": "CA",
    "ALA": "CB",
    "GLN": "CD",
    "GLU": "CD",
    "ILE": "CD1",
    "LEU": "CD1",
    "MET": "CE",
    "HIS": "CE1",
    "ASN": "CG",
    "ASP": "CG",
    "PRO": "CG",
    "VAL": "CG1",
    "THR": "CG2",
    "TRP": "CH2",
    "ARG": "CZ",
    "PHE": "CZ",
    "LYS": "NZ",
    "SER": "OG",
    "TYR": "OH",
    "CYS": "SG",
}

# Kyte-Doolittle Hydrophobicity scale mapping
# Taken from: https://resources.qiagenbioinformatics.com/manuals/clcgenomicsworkbench/650/Hydrophobicity_scales.html
# Reference: https://home.hiroshima-u.ac.jp/kei/IdentityX/picts/BE-hydrophobicity.pdf#page4
# Positive values -> hydrophobic
# Negative values -> hydrophilic (more likely to be interact with solvent -> surface)
kyte_doolittle_dict = {
    "ALA": 1.8,
    "CYS": 2.5,
    "ASP": -3.5,
    "GLU": -3.5,
    "PHE": 2.8,
    "GLY": -0.4,
    "HIS": -3.2,
    "ILE": 4.5,
    "LYS": -3.9,
    "LEU": 3.8,
    "MET": 1.9,
    "ASN": -3.5,
    "PRO": -1.6,
    "GLN": -3.5,
    "ARG": -4.5,
    "SER": -0.8,
    "THR": -0.7,
    "VAL": 4.2,
    "TRP": -0.9,
    "TYR": -1.3,
}

amino_acid_list = list(lha_dict.keys())

warnings.filterwarnings("ignore")

"""**Residue Definition**"""

class Residue:
    """
    A class to represent a Residue.
    In this case, a Residue is represented by a reference atom (either LHA, CA, etc).
    Residue is the "gene" in genetic algorithms (which constitutes an "individual").
    In other words, an individual is a list of Residue.
    A residue consists of 2 residue information, 1 chain information, and 2 reference atom information

    Attributes
    ----------
    residue_name : str
        The 3-letter amino acid name
        Example: "LYS"

    residue_sequence_position: int
        The position number of this residue inside the whole protein sequence
        Example: 67

    chain_name : str
        The 1-letter name of the corresponding chain
        Example: "A"

    atom_name : str
        The abbreviation of the reference atom name
        Example: "CA"

    atom_coordinates : list[float]
        The 3-dimensional coordinates of the reference atom
        Example: [-7.555, 8.098, 13.093]

    residue_sasa : float
        The Solvent Accessibility Surface Areas (SASA) associated for each residue
        Calculated using Shrake-Rupley algorithm. Unit in Angstrom^2
        Example: 69.11805430792289

    residue_depth: float
        The ResidueDepth associated for each residue
        Calculated using MSMS (calling from Bio.PDB.ResidueDepth). Unit in Angstrom
        Example: 1.485757

    """
    def __init__(self, residue_name, residue_sequence_position, chain_name, atom_name, atom_coordinates, residue_sasa, residue_depth):
        self.residue_name = residue_name
        self.residue_sequence_position = residue_sequence_position
        self.chain_name = chain_name
        self.atom_name = atom_name
        self.atom_coordinates = atom_coordinates
        self.residue_sasa = residue_sasa
        self.residue_depth = residue_depth

def is_same_residue(residue_1, residue_2):
    """Is Same Residue
    Given two Residue objects, determine whether it is the same residue or not

    Parameters:
    residue_1 (Residue): The first residue to be compared
    residue_2 (Residue): The second residue to be compared

    Returns:
    bool: A value which indicates whether these two Residue objects are the same or not
    """

    # For now, two residues are the same if the residue_name, residue_sequence_position,
    # and the chain_name are the same.
    # Regardless whether it is the same instance or not
    is_same_residue_name = residue_1.residue_name == residue_2.residue_name
    is_same_sequence_position = residue_1.residue_sequence_position == residue_2.residue_sequence_position
    is_same_chain_name = residue_1.chain_name == residue_2.chain_name
    
    return (is_same_residue_name and is_same_sequence_position and is_same_chain_name)

def is_residue_in_interface(residue, interface):
    """Is Residue In Interface
    Given a residue object and an interface, check whether this residue object exists inside the interface

    Parameters:
    residue (Residue): A Residue object to be checked
    interface (list[Residue]): List of Residue objects which represents a protein interface

    Returns:
    bool: A value which indicates whether the residue exists inside the interface

    """
    return any([True if is_same_residue(residue, current_residue) else False for current_residue in interface])

def print_interface_info(interface):
    """Print Interface Info
    Given an interface, print relevant information
    For debugging purpose only
    
    Parameters:
    interface (list[Residue]): The list of Residue object which constitutes the interface
    
    Returns:
    None
    
    """
    print("Number of Residues: ", len(interface))
    for i in range(len(interface)):
        if i > 0:
            print(" - ", end="")
        print(interface[i].residue_name, interface[i].residue_sequence_position, interface[i].chain_name, end="")
    print("")

"""**Utility Functions**"""

def euclidean_distance(coordinate_1, coordinate_2):
    """Euclidean Distance
    Given 3-dimensional coordinates of 2 atoms, calculate its Euclidean distance
    
    Parameters:
    coordinate_1 (list[float]): x,y,z coordinates of the first atom
    coordinate_2 (list[float]): x,y,z coordinates of the second atom
    
    Returns:
    float: The euclidean distance
    
    """
    return float(np.sqrt(((coordinate_1[0] - coordinate_2[0]) ** 2) +
                   ((coordinate_1[1] - coordinate_2[1]) ** 2) +
                   ((coordinate_1[2] - coordinate_2[2]) ** 2)))

def load_pdb(pdb_id, pdb_directory_path, pdb_parser, lha_dict, reference_atom="lha"):
    """Load PDB
    Given a PDB ID and its directory, load the .pdb file using Bio.PDB module and generate a list of residue
    
    Parameters:
    pdb_id (str): The PDB ID for the protein structure
    pdb_directory_path (str): Absolute path to access the PDB file
    pdb_parser (Bio.PDB.PDBParser.PDBParser): Bio.PDB Parser
    lha_dict (dict{residue_name: atom_name}): Corresponding Last Heavy Atom for each amino acids 
    reference_atom (str): Reference atom used ("lha" or "ca")
    
    Returns:
    list[Residue]: List of Residue object which constitutes the protein structure
    
    """
    residue_list = []
    amino_acid_list = list(lha_dict.keys())
    pdb_file_path = pdb_directory_path + pdb_id + ".pdb"
    pdb_structure = pdb_parser.get_structure(pdb_id, pdb_file_path)

    # Calculate the SASA for each residue using Shrake-Rupley algorithm from Bio.PDB
    sr = ShrakeRupley()
    sr.compute(pdb_structure, level="R")

    # Calculate ResidueDepth for each residue using MSMS from Bio.PDB.ResidueDepth
    # Each value can be accessed through residue_depth_dict[chain_name][residue_sequence_position][residue_name]
    residue_depth_dict = {}
    rd = ResidueDepth(pdb_structure)
    for item in rd:
        chain_name = item[0].get_parent().id
        residue_sequence_position = item[0].get_full_id()[3][1]
        residue_name = item[0].get_resname()
        residue_depth = item[1][0]
        # Create empty sub-dictionary if a key has never been encountered before
        if chain_name not in residue_depth_dict:
            residue_depth_dict[chain_name] = {}
        if residue_sequence_position not in residue_depth_dict[chain_name]:
            residue_depth_dict[chain_name][residue_sequence_position] = {}
        # Put the depth information inside the dictionary
        residue_depth_dict[chain_name][residue_sequence_position][residue_name] = residue_depth

    # Only take the ATOM keyword (exclude the HETATM, hetero atom that is not inside standard amino acids)
    biopdb_residue_list = [residue for residue in pdb_structure.get_residues() if residue.get_resname() in amino_acid_list]
    if reference_atom == "lha":
        biopdb_atom_list = [atom for residue in biopdb_residue_list for atom in residue if atom.get_name() == lha_dict[residue.get_resname()]]
    else:
        biopdb_atom_list = [atom for residue in biopdb_residue_list for atom in residue if atom.get_name() == "CA"]
        
    for atom in biopdb_atom_list:
        # Create a new Residue instance
        current_residue_name = atom.get_parent().get_resname()
        current_residue_sequence_position = atom.get_parent().get_full_id()[3][1]
        current_chain_name = atom.get_parent().get_parent().id
        current_atom_name = atom.get_name()
        current_atom_coordinates = atom.get_coord().tolist()
        current_residue_sasa = float(atom.get_parent().sasa)
        current_residue_depth = float(residue_depth_dict[current_chain_name][current_residue_sequence_position][current_residue_name])
        current_atom = Residue(current_residue_name,
                               current_residue_sequence_position,
                               current_chain_name,
                               current_atom_name,
                               current_atom_coordinates,
                               current_residue_sasa,
                               current_residue_depth)
        residue_list.append(current_atom)
    return residue_list

def get_actual_interface(residue_list_1, residue_list_2, distance_threshold=6.0):
    """Get Actual Interface
    Given one receptor structure and one ligand structure, infer its interfaces based on certain distance threshold
    
    Parameters:
    residue_list_1 (list[Residue]): List of Residue object from the first protein structure
    residue_list_2 (list[Residue]): List of Residue object from the second protein structure
    distance_threshold (float): The acceptable distance between a receptor's atom and a ligand's atom (in Angstrom unit)
    
    Returns:
    list[Residue]: List of Residue object which constitutes the protein interface
    
    """
    interface_list = []
    for residue_1 in residue_list_1:
        for residue_2 in residue_list_2:
            current_distance = euclidean_distance(residue_1.atom_coordinates, residue_2.atom_coordinates)
            if current_distance < distance_threshold:
                interface_list.append(residue_1)
                break

    return interface_list

def remove_hetatm_row(file_path):
	"""Remove HETATM Row
	Given a path to a PDB file, remove its HETATM rows

	Parameters:
	file_path (str): File path to a PDB file

	Returns:
	None

	"""
	line_list = []
	with open(file_path, "r") as fp:
		line_list = fp.readlines()

	with open(file_path, "w") as fp:
		for line in line_list:
			if "HETATM" not in line:
				fp.write(line)

"""**Dataset Preprocessing**"""

# Docking Benchmark 5.0 (abbreviated as DBD5)
# https://zlab.umassmed.edu/benchmark/benchmark5.0.html
# List of PDB ID of protein complexes based on Complex Category Labels
# Modified PDB ID due to SASA (remove the HETATM rows): 3SZK, 1WEJ, 2PCC, 2YVJ
# EI = Enzyme-Inhibitor
dbd5_ei_list = [ # Rigid-body
                "1AVX", "1AY7", "1BUH", "1BVN", "1CLV", "1D6R", "1DFJ", "1EAW",
                "1EZU", "1F34", "1FLE", "1GL1", "1GXD", "1HIA", "1JTD", "1JTG",
                "1MAH", "1OPH", "1OYV", "1PPE", "1R0R", "1TMQ", "1UDI", "1YVB",
                "2ABZ", "2B42", "2J0T", "2OUL", "2SIC", "2SNI", "2UUY", "3A4S",
                "3SGQ", "3VLB", "4CPA", "4HX3", "7CEI",
                 # Medium Difficulty
                "1CGI", "1JIW", "4IZ7",
                 # Difficult
                "1ACB", "1PXV", "1ZLI", "2O3B"]

# ES = Enzyme-Substrate
dbd5_es_list = [ # Rigid-body
                "1E6E", "1EWY", "1Z5Y", "2A1A", "2A9K", "2MTA", "2O8V", "2OOB",
                "2PCC", "4H03",
                 # Medium Difficulty
                "1KKL", "1ZM4", "4LW4",
                 # Difficult
                "1F6M", "1FQ1", "1JK9", "2IDO"]

# ER = Enzyme complex with a regulatory or accessory chain
dbd5_er_list = [ # Rigid-body
                "1F51", "1GLA", "1JWH", "1OC0", "1US7", "1WDW", "2AYO", "2GAF",
                "2OOR", "2YVJ", "3K75", "3LVK", "3PC8",
                 # Medium Difficulty
                "1IJK", "1M10", "1NW9", "1R6Q", "2NZ8", "2Z0E" , "4FZA",
                 # Difficult
                "1JMO", "1JZD", "2OT3", "3FN1", "3H11", "4GAM"]

# A = Antibody-Antigen (receptor-ligand)
dbd5_a_list = [ # Rigid-body
               "1AHW", "1BVK", "1DQJ", "1E6J", "1JPS", "1MLC", "1VFB", "1WEJ",
               "2FD6", "2I25", "2VIS", "2VXT", "2W9E", "3EOA", "3HMX", "3MXW",
               "3RVW", "4DN4", "4FQI", "4G6J", "4G6M", "4GXU",
                # Medium Difficulty
               "3EO1", "3G6D", "3HI6", "3L5W", "3V6Z",
                # Difficult
               "1BGX"]

# AB = Antigen - Bound Antibody
dbd5_ab_list = [ # Rigid-body
                "1BJ1", "1FSK", "1I9R", "1IQD", "1K4C", "1KXQ", "1NCA", "1NSN",
                "1QFW", "2JEL",
                 # Difficult
                "2HMI"]

# OG = Others, G-protein containing
dbd5_og_list = [ # Rigid-body
                "1A2K", "1AZS", "1E96", "1FQJ", "1HE1", "1I4D", "1J2J", "1Z0K",
                "2FJU", "2G77", "2GTP",
                 # Medium Difficulty
                "1GP2", "1GRN", "1HE8", "1I2M", "1K5D", "1LFD", "1WQ1", "2H7V",
                "3CPH",
                 # Difficult
                "1BKD", "1IBR", "1R8S"]

# OR = Others, Receptor containing
dbd5_or_list = [ # Rigid-body
                "1GHQ", "1HCF", "1K74", "1KAC", "1KTZ", "1ML0", "1PVH", "1RV6",
                "1SBB", "1T6B", "1XU1", "1ZHH", "2AJF", "2HLE", "2X9A", "4M76",
                 # Medium Difficulty
                "3R9A", "3S9D",
                 # Difficult
                "1E4K", "1EER", "1FAK", "1IRA", "2I9B", "3L89"]

# OX = Others, miscellaneous
dbd5_ox_list = [ # Rigid-body
                "1AK4", "1AKJ", "1EFN", "1EXB", "1FCC", "1FFW", "1GCQ", "1GPW",
                "1H9D", "1KLU", "1KXP", "1M27", "1OFU", "1QA9", "1RLB", "1S1Q",
                "1XD3", "1ZHI", "2A5T", "2B4J", "2BTF", "2HQS", "2VDB", "3BIW",
                "3BP8", "3D5S", "3H2V", "3P57",
                 # Medium Difficulty
                "1B6C", "1FC2", "1IB1", "1MQ8", "1N2C", "1SYX", "1XQS", "2CFH",
                "2HRK", "2OZA", "3AAA", "3AAD", "3BX7", "3DAW", "3SZK", "4JCV",
                 # Difficult
                "1ATN", "1DE4", "1H1V", "1RKE", "1Y64", "2C0L", "2J7P", "3F1P"]

dbd5_all_list = dbd5_ei_list + dbd5_es_list + dbd5_er_list + dbd5_a_list + dbd5_ab_list + dbd5_og_list + dbd5_or_list + dbd5_ox_list

# Docking Benchmark 3.0 (abbreviated as DBD3)
# https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2726780/
# List of PDB ID of protein complexes based on Complex Category Labels
# E = Enzyme-Inhibitor or Enzyme-Substrate
dbd3_e_list = [ # Rigid-body
               "1AVX", "1AY7", "1BVN", "1CGI", "1D6R", "1DFJ", "1E6E", "1EAW",
               "1EWY", "1EZU", "1F34", "1HIA", "1MAH", "1N8O", "1OPH", "1PPE",
               "1R0R", "1TMQ", "1UDI", "1YVB", "2B42", "2MTA", "2O8V", "2PCC",
               "2SIC", "2SNI", "2UUY", "7CEI",
                # Medium Difficulty
               "1ACB", "1IJK", "1KKL", "1M10", "1NW9",
                # Difficult
               "1FQ1", "1PXV"]

# A = Antibody-Antigen
dbd3_a_list = [ # Rigid-body
               "1AHW", "1BVK", "1DQJ", "1E6J", "1JPS", "1MLC", "1VFB", "1WEJ",
               "2FD6", "2I25", "2VIS",
                # Medium Difficulty
               "1BGX",
                # Difficult
               "1E4K"]

# AB = Antigen-Bound Antibody
dbd3_ab_list = [ # Rigid-body
                "1BJ1", "1FSK", "1I9R", "1IQD", "1K4C", "1KXQ", "1NCA", "1NSN",
                "1QFW", "2JEL",
                 # Difficult
                "2HMI"]

# O = Others
dbd3_o_list = [ # Rigid-body
                "1A2K", "1AK4", "1AKJ", "1AZS", "1B6C", "1BUH", "1E96", "1EFN",
                "1F51", "1FC2", "1FQJ", "1GCQ", "1GHQ", "1GLA", "1GPW", "1HE1",
                "1I4D", "1J2J", "1K74", "1KAC", "1KLU", "1KTZ", "1KXP", "1ML0",
                "1QA9", "1RLB", "1S1Q", "1SBB", "1T6B", "1XD3", "1Z0K", "1Z5Y",
                "1ZHI", "2AJF", "2BTF", "2HLE", "2HQS", "2OOB",
                # Medium Difficulty
                "1GP2", "1GRN", "1HE8", "1I2M", "1IB1", "1K5D", "1N2C", "1WQ1",
                "1XQS", "2CFH", "2H7V", "2HRK", "2NZ8",
                # Difficult
                "1ATN", "1BKD", "1DE4", "1EER", "1FAK", "1H1V", "1IBR", "1IRA", "1JMO",
                "1R8S", "1Y64", "2C0L", "2OT3"]

dbd3_all_list = dbd3_e_list + dbd3_a_list + dbd3_ab_list + dbd3_o_list

def infer_ppi_templates(pdb_id_list, pdb_directory_path, pdb_parser, lha_dict, reference_atom, distance_threshold):
    """Infer PPI Templates
    For each protein complex in the dataset, calculate and infer its PPI templates

    Parameters:
    pdb_id_list (list[str]): List of Protein Complex PDB ID to be processed
    pdb_directory_path (str): Absolute path to access the PDB file
    pdb_parser (Bio.PDB.PDBParser.PDBParser): Bio.PDB Parser
    lha_dict (dict{residue_name: atom_name}): Corresponding Last Heavy Atom for each amino acids 
    reference_atom (str): Reference atom used ("lha" or "ca")
    distance_threshold (float): The acceptable distance between a receptor's atom and a ligand's atom (in Angstrom unit)

    Returns:
    dict{pdb_id: list[Residue]}: Dictionary of PPI templates for each PDB ID

    """
    ppi_templates_dict = {}
    for pdb_id in pdb_id_list:
        current_ligand = load_pdb(pdb_id + "_l_u", pdb_directory_path, pdb_parser, lha_dict, reference_atom)
        current_receptor = load_pdb(pdb_id + "_r_u", pdb_directory_path, pdb_parser, lha_dict, reference_atom)
        ppi_templates_dict[pdb_id + "_l_u"] = get_actual_interface(current_ligand, current_receptor, distance_threshold)
        ppi_templates_dict[pdb_id + "_r_u"] = get_actual_interface(current_receptor, current_ligand, distance_threshold)

    return ppi_templates_dict

def save_ppi_templates(ppi_templates_dict, save_directory_path, file_name):
    """Save PPI Templates
    Using Python JSON library, save the PPI templates dictionary as a JSON file

    Parameters:
    ppi_templates_dict (dict{pdb_id: list[Residue]}): Dictionary of PPI templates for each PDB ID
    save_directory_path (str): Absolute directory path to save the file
    file_name (str): The saved file name with its .json extension

    Returns:
    None

    """
    # Convert the ppi_templates_dict into JSON-friendly dictionary
    json_friendly_dict = { pdb_id: [residue.__dict__ for residue in residue_list] for pdb_id, residue_list in ppi_templates_dict.items() }

    # Save the file
    with open(save_directory_path + file_name, "w") as fp:
        json.dump(json_friendly_dict, fp)

def load_ppi_templates(load_directory_path, file_name):
    """Load PPI Templates
    Using Python JSON library, load a JSON file which contains PPI templates dictionary

    Parameters:
    load_directory_path (str): Absolute directory path to load the file
    file_name (str): The file name with its .json extension to be loaded

    Returns:
    dict{pdb_id: list[Residue]}: Dictionary of PPI templates for each PDB ID

    """
    # In case the file is empty, initialize an empty dictionary to be returned
    deserialized_ppi_templates = {}

    # Load the file
    with open(load_directory_path + file_name, "r") as fp:
        loaded_ppi_templates = json.load(fp)

    # Deserialized the loaded_ppi_templates
    deserialized_ppi_templates = { pdb_id: [Residue(x['residue_name'], x['residue_sequence_position'], x['chain_name'], x['atom_name'], x['atom_coordinates'], x['residue_sasa'], x['residue_depth']) for x in residue_list] 
                                    for pdb_id, residue_list in loaded_ppi_templates.items() }
    return deserialized_ppi_templates

# Docking Benchmark 5: Load PPI Templates (5A)
# Load DBD5 PPI template dictionary from .json files (to be used in the main program)
dbd5_a_templates_5a_dict = load_ppi_templates(templates_path, "dbd5_a_templates_5a.json")
dbd5_ei_templates_5a_dict = load_ppi_templates(templates_path, "dbd5_ei_templates_5a.json")
dbd5_er_templates_5a_dict = load_ppi_templates(templates_path, "dbd5_er_templates_5a.json")
dbd5_es_templates_5a_dict = load_ppi_templates(templates_path, "dbd5_es_templates_5a.json")
dbd5_ab_templates_5a_dict = load_ppi_templates(templates_path, "dbd5_ab_templates_5a.json")
dbd5_og_templates_5a_dict = load_ppi_templates(templates_path, "dbd5_og_templates_5a.json")
dbd5_ox_templates_5a_dict = load_ppi_templates(templates_path, "dbd5_ox_templates_5a.json")
dbd5_or_templates_5a_dict = load_ppi_templates(templates_path, "dbd5_or_templates_5a.json")

dbd5_all_templates_5a_dict = {**dbd5_a_templates_5a_dict, **dbd5_ei_templates_5a_dict, **dbd5_er_templates_5a_dict,
                      **dbd5_es_templates_5a_dict, **dbd5_ab_templates_5a_dict, **dbd5_og_templates_5a_dict,
                      **dbd5_ox_templates_5a_dict, **dbd5_or_templates_5a_dict}
print(len(dbd5_all_templates_5a_dict))

# Docking Benchmark 3: Load PPI Templates (5A)
# Load DBD3 PPI template dictionary from .json files (to be used in the main program)
dbd3_e_templates_5a_dict = load_ppi_templates(templates_path, "dbd3_e_templates_5a.json")
dbd3_a_templates_5a_dict = load_ppi_templates(templates_path, "dbd3_a_templates_5a.json")
dbd3_ab_templates_5a_dict = load_ppi_templates(templates_path, "dbd3_ab_templates_5a.json")
dbd3_o_templates_5a_dict = load_ppi_templates(templates_path, "dbd3_o_templates_5a.json")

dbd3_all_templates_5a_dict = {**dbd3_e_templates_5a_dict, **dbd3_a_templates_5a_dict, **dbd3_ab_templates_5a_dict,
                      **dbd3_o_templates_5a_dict}
print(len(dbd3_all_templates_5a_dict))

"""**GASS-PPI Core Functions**
"""

def calculate_fitness_score(individual, interface_template):
    """Calculate Fitness Score
    Adhering to the original GASS, calculate the distance between an individual and the interface template
    Essentially, it's the modified version of RMSD (with no normalising factor)
    
    Parameters:
    individual (list[Residue]): The individual that needs to be evaluated
    interface_template (list[Residue]): The interface template, used as a reference
    
    Returns:
    float: The fitness score of an individual
    
    """
    n = len(individual)

    # Fitness Score 1: Spatial Distance Score
    individual_coordinates_list = [residue.atom_coordinates for residue in individual]
    individual_distance_list = [euclidean_distance(individual_coordinates_list[i], individual_coordinates_list[j]) for i in range(n-1) for j in range(i+1, n)]

    template_coordinates_list = [residue.atom_coordinates for residue in interface_template]
    template_distance_list = [euclidean_distance(template_coordinates_list[i], template_coordinates_list[j]) for i in range(n-1) for j in range(i+1, n)]

    spatial_distance_score = float(np.sqrt(np.sum([(abs(x[0] - x[1]) ** 2) for x in zip(individual_distance_list, template_distance_list)])))

    # Fitness Score 2: Depth Distance Score
    individual_depth_list = [residue.residue_depth for residue in individual]
    individual_depth_distance_list = [abs(individual_depth_list[i] - individual_depth_list[j]) for i in range(n-1) for j in range(i+1, n)]

    template_depth_list = [residue.residue_depth for residue in interface_template]
    template_depth_distance_list = [abs(template_depth_list[i] - template_depth_list[j]) for i in range(n-1) for j in range(i+1, n)]

    depth_distance_score = float(np.sqrt(np.sum([(abs(x[0] - x[1]) ** 2) for x in zip(individual_depth_distance_list, template_depth_distance_list)])))

    # Fitness Score 3 (experiment): SASA score
    individual_sasa_list = [residue.residue_sasa for residue in individual]
    individual_sasa_distance_list = [abs(individual_sasa_list[i] - individual_sasa_list[j]) for i in range(n-1) for j in range(i+1, n)]

    template_sasa_list = [residue.residue_sasa for residue in interface_template]
    template_sasa_distance_list = [abs(template_sasa_list[i] - template_sasa_list[j]) for i in range(n-1) for j in range(i+1, n)]

    sasa_distance_score = float(np.sqrt(np.sum([(abs(x[0] - x[1]) ** 2) for x in zip(individual_sasa_distance_list, template_sasa_distance_list)])))

    # Total Fitness Score
    fitness_score = spatial_distance_score + depth_distance_score + sasa_distance_score
    return fitness_score

def calculate_normalised_fitness_score(individual, interface_template):
    """Calculate Normalised Fitness Score
    Adhering to the original GASS, calculate the distance between an individual and the interface template
    Essentially, it's the standard version of RMSD
    
    Parameters:
    individual (list[Residue]): The individual that needs to be evaluated
    interface_template (list[Residue]): The interface template, used as a reference
    
    Returns:
    float: The fitness score of an individual
    
    """
    n = len(individual)

    # Fitness Score 1: Spatial Distance Score
    individual_coordinates_list = [residue.atom_coordinates for residue in individual]
    individual_distance_list = [euclidean_distance(individual_coordinates_list[i], individual_coordinates_list[j]) for i in range(n-1) for j in range(i+1, n)]

    template_coordinates_list = [residue.atom_coordinates for residue in interface_template]
    template_distance_list = [euclidean_distance(template_coordinates_list[i], template_coordinates_list[j]) for i in range(n-1) for j in range(i+1, n)]

    # use this for other fitness scoring as well
    # becoming typical RMSD
    distance_size = len(individual_distance_list)

    spatial_distance_score = float(np.sqrt(np.sum([(abs(x[0] - x[1]) ** 2) for x in zip(individual_distance_list, template_distance_list)]) / distance_size))

    return spatial_distance_score

def generate_initial_population(input_protein_structure, interface_template, population_size):
    """Generate Initial Population
    Given the input protein structure and its interface template, generate list
    of random individuals which constitutes the initial population.

    Each residue inside the individual needs to have the same residue_name compared to
    the interface_template, but can have different residue_sequence_position or chain_name. 

    Parameters:
    input_protein_structure (list[Residue]): List of Residues object which constitutes the input protein structure
    interface_template (list[Residue]): List of Residue object which constitutes the interface template
    population_size (int): The number of random individuals to be generated

    Returns:
    list[list[Residue]]: The initial population (list of individuals)

    """

    # Split the input_protein_structure into 20 list (based on amino_acid_list)
    input_protein_dict= { x: [] for x in amino_acid_list }
    for residue in input_protein_structure:
        input_protein_dict[residue.residue_name].append(residue)

    initial_population_list = []
    # Generate the initial population
    for _ in range(population_size):
        # Generate a random individual
        random_individual = []
        for residue in interface_template:
            amino_acid_type = residue.residue_name
            # Generate a random residue that aligns with the amino_acid_type
            if len(input_protein_dict[amino_acid_type]) > 0:
                random_residue_index = random.randrange(len(input_protein_dict[amino_acid_type]))
                random_residue = input_protein_dict[amino_acid_type][random_residue_index]
                # Ensure that the randomly picked residue is distinct inside the same individual
                while is_residue_in_interface(random_residue, random_individual):
                    random_residue_index = random.randrange(len(input_protein_dict[amino_acid_type]))
                    random_residue = input_protein_dict[amino_acid_type][random_residue_index]
            else:
                # In case the protein structure doesn't contain the required amino_acid_type, select any random residue with any type
                random_residue_index = random.randrange(len(input_protein_structure))
                random_residue = input_protein_structure[random_residue_index]

            random_individual.append(random_residue)
        initial_population_list.append(random_individual)

    return initial_population_list

def deterministic_tournament_selection(population_list, tournament_size, number_of_tournament):
    """Deterministic Tournament Selection
    Perform deterministic tournament selection towards current population list
    to select parents (for generating new generations)

    Parameters:
    population_list (list[(list[Residue], int)]): Current list of population
    tournament_size (int): Hyperparameter to determine how many random individuals in each tournament
    number_of_tournament (int): Hyperparameter to determine how many tournaments to be performed

    Returns:
    list[list[Residue]]: List of parents

    """
    random.shuffle(population_list)

    parent_list = []
    # Perform n tournaments (where n is the number_of_tournament)
    for _ in range(number_of_tournament):
        # Choose random individuals to participate in the tournament
        tournament_participant_list = []
        for _ in range(tournament_size):
            random_idx = random.randrange(len(population_list))
            tournament_participant_list.append(population_list[random_idx])
        # Find the fittest individual from the tournament_participant_list
        fittest_individual = tournament_participant_list[0]
        for i in range(1, len(tournament_participant_list)):
            if tournament_participant_list[i][1] < fittest_individual[1]:
                fittest_individual = tournament_participant_list[i]

        parent_list.append(fittest_individual)

    return parent_list

def crossover(individual_1, individual_2, crossover_probability):
    """Crossover
    Given two individuals, perform a single point crossover based on crossover probability
    
    Parameters:
    individual_1 (list[Residue]): First parent
    individual_2 (list[Residue]): Second parent
    crossover_probability (float): The probability that a crossover occurs (between 0 and 1.0)
    
    Returns:
    list[Residue]: Individual 1 after crossover
    list[Residue]: Individual 2 after crossover
    
    """
    # Only apply crossover if a random value is within the crossover probability
    # Otherwise, simply return these individuals without performing any operation
    random_percentage = random.random()
    if random_percentage < crossover_probability:
        # Randomly determine the crossover point
        individual_size = len(individual_1)
        crossover_point = random.randrange(individual_size)
        
        # Swap residues between two individuals, starting from the crossover point
        for i in range(crossover_point, individual_size):
            individual_1[i], individual_2[i] = individual_2[i], individual_1[i]
        
    return (individual_1, individual_2)

def mutation(input_protein_structure, individual, mutation_probability):
    """Mutation
    Given an individual, perform a single point mutations based on mutation probability.
    In this case, no conservative mutation is employed
    
    Parameters:
    input_protein_structure (list[Residue]): List of Residues object which constitutes the input protein structure
    individual (list[Residue]): Individual to be mutated
    mutation_probability (float): The probability that a mutation occurs (between 0 and 1.0)
    
    Returns:
    list[Residue]: Mutated individuals
    
    """
    # Only apply mutation if a random value is within the mutation probability
    # Otherwise, simply return the individual without performing any operation
    random_percentage = random.random()
    if random_percentage < mutation_probability:
        # Randomly determine the mutation point
        individual_size = len(individual)
        mutation_point = random.randrange(individual_size)
        mutated_residue = individual[mutation_point]
        
        # Mutate the residue at the mutation point into the same amino acid types 
        possible_mutation = [residue for residue in input_protein_structure 
                            if residue.residue_name == mutated_residue.residue_name 
                            and residue.residue_sequence_position != mutated_residue.residue_sequence_position
                            and not is_residue_in_interface(residue, individual)]
        
        if len(possible_mutation) > 0:
            new_residue_index = random.randrange(len(possible_mutation))
            new_residue = possible_mutation[new_residue_index]
            individual[mutation_point] = new_residue
    
    return individual

def can_run_gass_ppi(input_protein_structure, interface_template):
    """Can Run GASS PPI?
    Defensive Programming effort. Perform pre-check before running GASS-PPI.
    Edge Case: 
    - The template size should be larger than 0 (sample case: 2O3B has 0 template with distance threshold 4A)
    - GASS-PPI shouldn't run if the protein structure does not contain
      all necessary residues to form an individual which has the same
      residue types with the interface template
      e.g. The template has 4 CYS, but the input protein structure only contains 3 CYS
                

    Parameters:
    input_protein_structure (list[Residue]): List of Residue which constitutes the entire protein structure
    interface_template (list[Residue]): List of Residue which is used as a template for the genetic algorithms

    Returns:
    bool: True if it is possible to run GASS-PPI, False otherwise

    """
    # Check whether the template size is valid (>0)
    if len(interface_template) <= 0:
        return False

    filtered_protein_structure = [residue for residue in input_protein_structure]
    # Count the number of each residue type in the filtered_protein_structure
    number_of_residues_in_structure = { x: 0 for x in amino_acid_list }
    for residue in filtered_protein_structure:
        number_of_residues_in_structure[residue.residue_name] += 1

    # Count the number of each residue type in the interface_template
    number_of_residues_in_template = { x: 0 for x in amino_acid_list}
    for residue in interface_template:
        number_of_residues_in_template[residue.residue_name] += 1
    
    for amino_acid in amino_acid_list:
        if number_of_residues_in_template[amino_acid] > number_of_residues_in_structure[amino_acid]:
            return False
    
    return True

def gass_ppi(input_protein_structure, interface_template, population_size=300, number_of_generations=300, crossover_probability=0.5, mutation_probability=0.7, tournament_size=3, number_of_tournament=50, verbose=False):
    """GASS-PPI Method
    Given the input protein structure and the interface template, perform genetic algorithms
    to search the most likely interface
    
    Parameters:
    input_protein_structure (list[Residue]): List of Residue which constitutes the entire protein structure
    interface_template (list[Residue]): List of Residue which is used as a template for the genetic algorithms
    population_size (int): Total number of individuals inside the population
    number_of_generations (int): Number of iterations run in the genetic algorithm
    crossover_probability (float): Probability value between 0-1 which governs the likelihood that crossover is performed
    mutation_probability (float): Probability value between 0-1 which governs the likelihood that mutation is performed
    tournament_size (int): The number of individuals inside a particular tournament
    number_of_tournament (int): Number of tournaments to be performed
    verbose (bool): Draw plot and additional statistics (for analysis only). Default is False.
    
    Returns:
    list[(list[Residue], float)]: The final population generated from the genetic algorithms.
                                  Each tuple consists of the individual and its correspondings fitness score
    
    """
    # Track the lowest fitness score (for plot of convergence, if verbose=True)
    lowest_fitness_list = []
    sasa_threshold = 0.0
    eps = 0.000001
    protein_structure = [residue for residue in input_protein_structure]
    
    # Initial Population
    population_list_no_fitness = generate_initial_population(protein_structure, interface_template, population_size)
    population_list = [(individual, calculate_normalised_fitness_score(individual, interface_template)) for individual in population_list_no_fitness]
    
    # Evolutionary Steps
    for i in range(number_of_generations):
        # Selection
        parent_list = deterministic_tournament_selection(population_list, tournament_size, number_of_tournament)

        for j in range(0, len(parent_list), 2):
            # Crossover
            new_individual_1, new_individual_2 = crossover(list(population_list[j][0]), list(population_list[j+1][0]), crossover_probability)

            # Mutation
            mutated_new_individual_1 = mutation(protein_structure, list(new_individual_1), mutation_probability)
            mutated_new_individual_2 = mutation(protein_structure, list(new_individual_2), mutation_probability)

            # Fitness Evaluation
            fitness_score_1 = calculate_normalised_fitness_score(mutated_new_individual_1, interface_template)
            fitness_score_2 = calculate_normalised_fitness_score(mutated_new_individual_2, interface_template)

            # Add 2 new individuals into the population_list
            population_list.append((mutated_new_individual_1, fitness_score_1))
            population_list.append((mutated_new_individual_2, fitness_score_2))
        
        # Population Management (steady-state)
        population_list.sort(key = lambda x: x[1])
        population_list = population_list[:population_size]
        # Record the current lowest fitness score
        if verbose:
            lowest_fitness_list.append(population_list[0][1])
        # If the fitness score already 0, stop the evolutionary steps as it already converges towards the most optimal solution
        if population_list[0][1] <= eps:
            break

    return population_list

def evaluate_ppi(actual_interface, predicted_interface, protein_structure):
    """Evaluate PPI
    Given the actual interface and its predicted interface, calculate some performance metrics.
    Calculated using scikit-learn
    
    Parameters:
    actual_interface (list[Residue]): List of Residue objects which represents the experimentally proven interface
    predicted_inteface (list[Residue]): List of Residue objects which represents the predicted interface from GASS-PPI
    protein_structure (list[Residue]): List of Residue objects which represents the entire protein structure
    
    Returns:
    float: Precision (Positive Predictive Value) score
    float: Recall (Sensitivity/True Positive Rate) score 
    float: AUC-ROC score
    float: AUC-PR score
    float: Matthew's Correlation Coefficient score
    float: Specificity (True Negative Rate) score
    float: NPV (Negative Predictive Value) score
    
    """
    # Additional reference: https://en.m.wikipedia.org/wiki/Evaluation_of_binary_classifiers
    
    # From predicted interface, convert it into 0/1 label (y_pred)
    y_pred = [1 if is_residue_in_interface(residue, predicted_interface) else 0 for residue in protein_structure]

    # From actual interface, convert it into 0/1 label (y_actual)
    y_actual = [1 if is_residue_in_interface(residue, actual_interface) else 0 for residue in protein_structure]

    # Based on the actual and predicted labels, calculate the metrics using sklearn.metrics functions
    # https://scikit-learn.org/stable/modules/model_evaluation.html
    tn, fp, fn, tp = confusion_matrix(y_actual, y_pred).ravel()

    precision = precision_score(y_actual, y_pred)
    recall = recall_score(y_actual, y_pred)
    auc_roc = roc_auc_score(y_actual, y_pred)
    auc_pr = average_precision_score(y_actual, y_pred)
    mcc = matthews_corrcoef(y_actual, y_pred)
    specificity = tn / (tn + fp)
    npv = tn / (tn + fn)
    
    return (precision, recall, auc_roc, auc_pr, mcc, specificity, npv)

def evaluate_ppi_population(actual_interface, population_list, protein_structure, ranking_size=1):
    """Evaluate PPI Population
    Given a population of PPI produced by GASS-PPI method, evaluate n individual 
    (where n is the ranking_size) using evaluate_ppi method and return the best individual.
    For now, the "best" is indicated by the highest AUC-ROC

    Parameters:
    actual_interface (list[Residue]): List of Residue objects which represents the experimentally proven interface
    population_list (list[(list[Residue], float)]): The population list to be evaluated (already sorted in GASS-PPI)
    protein_structure (list[Residue]): List of Residue objects which represents the entire protein structure
    ranking_size (int): Number of individuals to be evaluated (from individual #0 to #ranking_size-1)


    Returns:
    int: Best Individual Ranking (from 0 to population_size - 1)
    float: Precision (Positive Predictive Value) score
    float: Recall (Sensitivity/True Positive Rate) score 
    float: AUC-ROC score
    float: AUC-PR score
    float: Matthew's Correlation Coefficient score
    float: Specificity (True Negative Rate) score
    float: NPV (Negative Predictive Value) score

    """
    max_ranking = 0
    max_precision = 0.0
    max_recall = 0.0
    max_auc_roc = 0.0
    max_auc_pr = 0.0
    max_mcc = 0.0
    max_specificity = 0.0
    max_npv = 0.0

    # Evaluate each individual until reaching the ranking_size
    for i in range(ranking_size):
        precision, recall, auc_roc, auc_pr, mcc, specificity, npv = evaluate_ppi(actual_interface, population_list[i][0], protein_structure)
        # Check whether the AUC-ROC is higher than the previous individual
        if auc_roc > max_auc_roc:
            max_ranking = i
            max_precision = precision
            max_recall = recall
            max_auc_roc = auc_roc
            max_auc_pr = auc_pr
            max_mcc = mcc
            max_specificity = specificity
            max_npv = npv
        
    return (max_ranking, max_precision, max_recall, max_auc_roc, max_auc_pr, max_mcc, max_specificity, max_npv)

"""**GASS-PPI Usage**"""

def dbd_sanity_test(pdb_id_list, templates_dict, ranking_size=100, verbose=False, iteration_per_protein=1, population_size=300, number_of_generations=300, crossover_probability=0.5, mutation_probability=0.7, tournament_size=3, number_of_tournament=50):
    """DBD Sanity Test
    Given a list of PDB ID available in Docking Benchmark Dataset and precomputed PPI templates,
    execute GASS-PPI on each protein complexes using its own template

    Parameters:
    pdb_id_list (list[str]): List of PDB ID available in Docking Benchmark 5 
    templates_dict (dict{pdb_id: list[Residue]}): Dictionary of PPI templates for each PDB ID
    ranking_size (int): Number of individuals to be evaluated (from individual #0 to #ranking_size-1)
    verbose (bool): True for additional logs, False otherwise
    iteration_per_protein (int): Number of iteration performed for each protein (1 by default, 30 for statistical confidence)
    population_size (int): Total number of individuals inside the population
    number_of_generations (int): Number of iterations run in the genetic algorithm
    crossover_probability (float): Probability value between 0-1 which governs the likelihood that crossover is performed
    mutation_probability (float): Probability value between 0-1 which governs the likelihood that mutation is performed
    tournament_size (int): The number of individuals inside a particular tournament
    number_of_tournament (int): Number of tournaments to be performed

    Returns:
    list[int]: Individual ranking list
    list[float]: List of precision score
    list[float]: List of recall score
    list[float]: List of AUC-ROC score
    list[float]: List of AUC-PR score
    list[float]: List of MCC score
    list[float]: List of Specificity score
    list[float]: List of NPV score

    """
    # Initialise returned list
    individual_ranking_list = []
    precision_list = []
    recall_list = []
    auc_roc_list = []
    auc_pr_list = []
    mcc_list = []
    specificity_list = []
    npv_list = []

    # For each PDB ID, evaluate both its corresponding ligand and receptor
    for pdb_id in pdb_id_list:
        for monomer_pdb_id in [pdb_id + "_l_u", pdb_id + "_r_u"]:
            # Step 1: Load the monomeric protein structure and PPI template
            monomer_pdb_structure = load_pdb(monomer_pdb_id, dbd5_path, pdb_parser, lha_dict, "lha")
            interface_template = templates_dict[monomer_pdb_id]

            for _ in range(iteration_per_protein):
                # Step 2: GASS-PPI
                if can_run_gass_ppi(monomer_pdb_structure, interface_template):
                    # if verbose:
                    #     print("Currently evaluating:", monomer_pdb_id, ", with template size:", len(interface_template))

                    predicted_population_list = gass_ppi(monomer_pdb_structure, interface_template, population_size=population_size, number_of_generations=number_of_generations, crossover_probability=crossover_probability, mutation_probability=mutation_probability, tournament_size=tournament_size, number_of_tournament=number_of_tournament, verbose=False)

                    # Step 3: Evaluation
                    individual_ranking, precision, recall, auc_roc, auc_pr, mcc, specificity, npv = evaluate_ppi_population(interface_template, predicted_population_list, monomer_pdb_structure, ranking_size)
                    # TODO: Remove this later, for debugging only
                    # if verbose:
                    #     print_interface_info(predicted_population_list[individual_ranking][0])

                    individual_ranking_list.append(individual_ranking)
                    precision_list.append(precision)
                    recall_list.append(recall)
                    auc_roc_list.append(auc_roc)
                    auc_pr_list.append(auc_pr)
                    mcc_list.append(mcc)
                    specificity_list.append(specificity)
                    npv_list.append(npv)
                else:
                    print("Cannot run GASS-PPI on ", monomer_pdb_id)
                    precision_list.append(0)
                    recall_list.append(0)
                    auc_roc_list.append(0)
                    auc_pr_list.append(0)
                    mcc_list.append(0)
                    specificity_list.append(0)
                    npv_list.append(0)

    # Additional logs for development purposes
    if verbose:
        print("Mean Precision: ", np.mean(precision_list))
        print("Mean Recall: ", np.mean(recall_list))
        print("Mean AUC-ROC Score: ", np.mean(auc_roc_list))
        print("Mean AUC-PR Score: ", np.mean(auc_pr_list))
        print("Mean MCC: ", np.mean(mcc_list))
        print("Mean Specificity: ", np.mean(specificity_list))
        print("Mean NPV: ", np.mean(npv_list))

    return (individual_ranking_list, precision_list, recall_list, auc_roc_list, auc_pr_list, mcc_list, specificity_list, npv_list)

# GA Parameter Tuning
def run_grid_search(pdb_id_list, templates_dict):
    """Run Grid Search

    Parameters:
    pdb_id_list (list[str]): List of PDB ID available in Docking Benchmark 5 
    templates_dict (dict{pdb_id: list[Residue]}): Dictionary of PPI templates for each PDB ID

    Returns:
    None

    """
    population_size_list = [300, 500]
    number_of_generations_list = [200, 300, 500]
    crossover_probability_list = [0.5, 0.7, 0.9]
    mutation_probability_list = [0.3, 0.7, 0.9]
    tournament_size_list = [2, 3]
    number_of_tournament_list = [50, 100]

    # Keep track of the current optimal value
    optimal_population_size = population_size_list[0]
    optimal_number_of_generations = number_of_generations_list[0]
    optimal_crossover_probability = crossover_probability_list[0]
    optimal_mutation_probability = mutation_probability_list[0]
    optimal_tournament_size = tournament_size_list[0]
    optimal_number_of_tournament = number_of_tournament_list[0]

    optimal_precision = 0.0
    optimal_recall = 0.0
    optimal_auc_roc = 0.0
    optimal_auc_pr = 0.0
    optimal_mcc = 0.0
    optimal_specificity = 0.0
    optimal_npv = 0.0

    for population_size in population_size_list:
        for number_of_generations in number_of_generations_list:
            for crossover_probability in crossover_probability_list:
                for mutation_probability in mutation_probability_list:
                    for tournament_size in tournament_size_list:
                        for number_of_tournament in number_of_tournament_list:
                            print("Running sanity test with population_size:", population_size, ", number_of_generations:", number_of_generations, ", crossover_probability:", crossover_probability, ", mutation_probability:", mutation_probability, ", tournament_size:", tournament_size, ", number_of_tournament:", number_of_tournament)
                            _, precision_list, recall_list, auc_roc_list, auc_pr_list, mcc_list, specificity_list, npv_list = dbd_sanity_test(pdb_id_list, templates_dict, ranking_size=100, verbose=False, iteration_per_protein=1, population_size=population_size, number_of_generations=number_of_generations, crossover_probability=crossover_probability, mutation_probability=mutation_probability, tournament_size=tournament_size, number_of_tournament=number_of_tournament)
                            avg_auc_roc = np.mean(auc_roc_list)
                            print("Mean AUC-ROC:", avg_auc_roc)
                            if avg_auc_roc > optimal_auc_roc:
                                optimal_population_size = population_size
                                optimal_number_of_generations = number_of_generations
                                optimal_crossover_probability = crossover_probability
                                optimal_mutation_probability = mutation_probability
                                optimal_tournament_size = tournament_size
                                optimal_number_of_tournament = number_of_tournament

                                optimal_precision = np.mean(precision_list)
                                optimal_recall = np.mean(recall_list)
                                optimal_auc_roc = avg_auc_roc
                                optimal_auc_pr = np.mean(auc_pr_list)
                                optimal_mcc = np.mean(mcc_list)
                                optimal_specificity = np.mean(specificity_list)
                                optimal_npv = np.mean(npv_list)
    print("")
    print("Optimal GA Parameters")                  
    print("Population Size:", optimal_population_size)
    print("Number of Generations:", optimal_number_of_generations)
    print("Crossover Probability:", optimal_crossover_probability)
    print("Mutation Probability:", optimal_mutation_probability)
    print("Tournament Size:", optimal_tournament_size)
    print("Number of Tournament:", optimal_number_of_tournament)

    print("which produced below metrics:")
    print("Mean precision:", optimal_precision)
    print("Mean recall:", optimal_recall)
    print("Mean AUC-ROC:", optimal_auc_roc)
    print("Mean AUC-PR:", optimal_auc_pr)
    print("Mean MCC:", optimal_mcc)
    print("Mean Specificity:", optimal_specificity)
    print("Mean NPV:", optimal_npv)

def dbd_all_templates(pdb_id_list, templates_dict, ranking_size=100, verbose=False):
    """DBD All Templates
    Given a list of PDB ID available in Docking Benchmark Dataset and precomputed PPI templates,
    execute GASS-PPI on each protein complexes using all other templates

    Parameters:
    pdb_id_list (list[str]): List of PDB ID available in Docking Benchmark Dataset
    templates_dict (dict{pdb_id: list[Residue]}): Dictionary of PPI templates for each PDB ID
    ranking_size (int): Number of individuals to be evaluated (from individual #0 to #ranking_size-1)
    verbose (bool): True for additional logs, False otherwise

    Returns:
    list[int]: Individual ranking list
    list[float]: List of precision score
    list[float]: List of recall score
    list[float]: List of AUC-ROC score
    list[float]: List of AUC-PR score
    list[float]: List of MCC score
    list[float]: List of Specificity score
    list[float]: List of NPV score

    """
    # Initialise returned list
    individual_ranking_list = []
    precision_list = []
    recall_list = []
    auc_roc_list = []
    auc_pr_list = []
    mcc_list = []
    specificity_list = []
    npv_list = []

    # For each PDB ID, evaluate both its corresponding ligand and receptor
    for pdb_id in pdb_id_list:
        for monomer_pdb_id in [pdb_id + "_l_u", pdb_id + "_r_u"]:
            # Step 1: Load the monomeric protein structure and PPI template
            monomer_pdb_structure = load_pdb(monomer_pdb_id, dbd5_path, pdb_parser, lha_dict, "lha")
            aggregated_population_list = []

            if verbose:
                print("Currently evaluating:", monomer_pdb_id)

            # Step 2: GASS-PPI with all templates except its own
            for template_pdb_id, interface_template in templates_dict.items():
                if (monomer_pdb_id != template_pdb_id):
                    if (can_run_gass_ppi(monomer_pdb_structure, interface_template) and len(interface_template) > 1):
                        template_size = len(interface_template)
                        if verbose:
                            print("use template of ", template_pdb_id, "with template size:", template_size)
                            print_interface_info(interface_template)
                        current_population_list = gass_ppi(monomer_pdb_structure, interface_template, verbose=False)
                        # normalised_population_list = [(individual[0], individual[1] / template_size) for individual in current_population_list]
                        # aggregated_population_list += normalised_population_list
                        aggregated_population_list += current_population_list

            aggregated_population_list.sort(key = lambda x: x[1])
            # aggregated_population_list = aggregated_population_list[:ranking_size]
            if verbose:
                for i in range(50):
                    print("Ranking", i+1, "with fitness score =", aggregated_population_list[i][1])
                    print_interface_info(aggregated_population_list[i][0])

            # Step 3: Evaluation
            actual_interface = templates_dict[monomer_pdb_id]
            individual_ranking, precision, recall, auc_roc, auc_pr, mcc, specificity, npv = evaluate_ppi_population(actual_interface, aggregated_population_list, monomer_pdb_structure, ranking_size)
            individual_ranking_list.append(individual_ranking)
            precision_list.append(precision)
            recall_list.append(recall)
            auc_roc_list.append(auc_roc)
            auc_pr_list.append(auc_pr)
            mcc_list.append(mcc)
            specificity_list.append(specificity)
            npv_list.append(npv)

    # Additional logs for development purposes
    if verbose:
        print("Mean Precision: ", np.mean(precision_list))
        print("Mean Recall: ", np.mean(recall_list))
        print("Mean AUC-ROC Score: ", np.mean(auc_roc_list))
        print("Mean AUC-PR Score: ", np.mean(auc_pr_list))
        print("Mean MCC: ", np.mean(mcc_list))
        print("Mean Specificity: ", np.mean(specificity_list))
        print("Mean NPV: ", np.mean(npv_list))

    return (individual_ranking_list, precision_list, recall_list, auc_roc_list, auc_pr_list, mcc_list, specificity_list, npv_list)

def run_gass_ppi_application(pdb_id, templates_dict):
    """Run GASS-PPI Application
    Given an input monomeric protein, use GASS-PPI to search for possible list of PPIs

    Parameters:
    pdb_id (str): Input PDB ID
    templates_dict (dict{pdb_id: list[Residue]}): Dictionary of PPI templates for each PDB ID

    Returns:
    list[(list[Residue], float)]: The final population generated from the genetic algorithms.
                                  Each tuple consists of the individual and its correspondings fitness score

    """
    # For now, it is assumed that the PDB ID is located at the Google Drive
    pdb_testcase_path = repository_path + "testcase/"
    pdb_structure = load_pdb(pdb_id, pdb_testcase_path, pdb_parser, lha_dict, "lha")
    aggregated_population_list = []

    for template_pdb_id, interface_template in templates_dict.items():
        if can_run_gass_ppi(pdb_structure, interface_template):
            template_size = len(interface_template)
            current_population_list = gass_ppi(pdb_structure, interface_template, verbose=False)
            normalised_population_list = [(individual[0], individual[1] / template_size) for individual in current_population_list]
            aggregated_population_list += normalised_population_list

    aggregated_population_list.sort(key = lambda x: x[1])
    aggregated_population_list = aggregated_population_list[:100]

    return aggregated_population_list

"""**Sanity Test Execution**

Below indicates the performance of the SOTA methods:

"""

# DBD5 Sanity Test for each complex category labels
print("DBD5: Enzyme-Inhibitor (EI)")
ei_ranking_list, ei_precision_list, ei_recall_list, ei_auc_roc_list, ei_auc_pr_list, ei_mcc_list, ei_specificity_list, ei_npv_list = dbd_sanity_test(dbd5_ei_list, dbd5_ei_templates_5a_dict, verbose=True, iteration_per_protein=30)
print("\n\n")

print("DBD5: Enzyme-Substrate (ES)")
es_ranking_list, es_precision_list, es_recall_list, es_auc_roc_list, es_auc_pr_list, es_mcc_list, es_specificity_list, es_npv_list = dbd_sanity_test(dbd5_es_list, dbd5_es_templates_5a_dict, verbose=True, iteration_per_protein=30)
print("\n\n")

print("DBD5: Enzyme complex with a regulatory or accessory chain (ER)")
er_ranking_list, er_precision_list, er_recall_list, er_auc_roc_list, er_auc_pr_list, er_mcc_list, er_specificity_list, er_npv_list = dbd_sanity_test(dbd5_er_list, dbd5_er_templates_5a_dict, verbose=True, iteration_per_protein=30)
print("\n\n")

print("DBD5: Antibody-Antigen (A)")
a_ranking_list, a_precision_list, a_recall_list, a_auc_roc_list, a_auc_pr_list, a_mcc_list, a_specificity_list, a_npv_list = dbd_sanity_test(dbd5_a_list, dbd5_a_templates_5a_dict, verbose=True, iteration_per_protein=30)
print("\n\n")

print("DBD5: Antigen-Bound Antibody (AB)")
ab_ranking_list, ab_precision_list, ab_recall_list, ab_auc_roc_list, ab_auc_pr_list, ab_mcc_list, ab_specificity_list, ab_npv_list = dbd_sanity_test(dbd5_ab_list, dbd5_ab_templates_5a_dict, verbose=True, iteration_per_protein=30)
print("\n\n")

print("DBD5: Others, G-protein containing (OG)")
og_ranking_list, og_precision_list, og_recall_list, og_auc_roc_list, og_auc_pr_list, og_mcc_list, og_specificity_list, og_npv_list = dbd_sanity_test(dbd5_og_list, dbd5_og_templates_5a_dict, verbose=True, iteration_per_protein=30)
print("\n\n")

print("DBD5: Others, Receptor containing (OR)")
or_ranking_list, or_precision_list, or_recall_list, or_auc_roc_list, or_auc_pr_list, or_mcc_list, or_specificity_list, or_npv_list = dbd_sanity_test(dbd5_or_list, dbd5_or_templates_5a_dict, verbose=True, iteration_per_protein=30)
print("\n\n")

print("DBD5: Others, miscellaneous (OX)")
ox_ranking_list, ox_precision_list, ox_recall_list, ox_auc_roc_list, ox_auc_pr_list, ox_mcc_list, ox_specificity_list, ox_npv_list = dbd_sanity_test(dbd5_ox_list, dbd5_ox_templates_5a_dict, verbose=True, iteration_per_protein=30)
print("\n\n")

print("DBD5: All Datasets")
dbd5_ranking_list = ei_ranking_list + es_ranking_list + er_ranking_list + a_ranking_list + ab_ranking_list + og_ranking_list + or_ranking_list + ox_ranking_list
dbd5_precision_list = ei_precision_list + es_precision_list + er_precision_list + a_precision_list + ab_precision_list + og_precision_list + or_precision_list + ox_precision_list
dbd5_recall_list = ei_recall_list + es_recall_list + er_recall_list + a_recall_list + ab_recall_list + og_recall_list + or_recall_list + ox_recall_list
dbd5_auc_roc_list = ei_auc_roc_list + es_auc_roc_list + er_auc_roc_list + a_auc_roc_list + ab_auc_roc_list + og_auc_roc_list + or_auc_roc_list + ox_auc_roc_list
dbd5_auc_pr_list = ei_auc_pr_list + es_auc_pr_list + er_auc_pr_list + a_auc_pr_list + ab_auc_pr_list + og_auc_pr_list + or_auc_pr_list + ox_auc_pr_list
dbd5_mcc_list = ei_mcc_list + es_mcc_list + er_mcc_list + a_mcc_list + ab_mcc_list + og_mcc_list + or_mcc_list + ox_mcc_list
dbd5_specificity_list = ei_specificity_list + es_specificity_list + er_specificity_list + a_specificity_list + ab_specificity_list + og_specificity_list + or_specificity_list + ox_specificity_list
dbd5_npv_list = ei_npv_list + es_npv_list + er_npv_list + a_npv_list + ab_npv_list + og_npv_list + or_npv_list + ox_npv_list

print("Mean Precision:", np.mean(dbd5_precision_list))
print("Mean Recall:", np.mean(dbd5_recall_list))
print("Mean AUC-ROC:", np.mean(dbd5_auc_roc_list))
print("Mean AUC-PR:", np.mean(dbd5_auc_pr_list))
print("Mean MCC:", np.mean(dbd5_mcc_list))
print("Mean Specificity:", np.mean(dbd5_specificity_list))
print("Mean NPV:", np.mean(dbd5_npv_list))

# DBD3 Sanity Test for each Complex Category Labels
print("DBD3: Enzyme-Inhibitor or Enzyme-Substrate (E)")
e_ranking_list, e_precision_list, e_recall_list, e_auc_roc_list, e_auc_pr_list, e_mcc_list, e_specificity_list, e_npv_list = dbd_sanity_test(dbd3_e_list, dbd3_e_templates_5a_dict, verbose=True, iteration_per_protein=30)
print("\n\n")

print("DBD3: Antibody-Antigen (A)")
a_ranking_list, a_precision_list, a_recall_list, a_auc_roc_list, a_auc_pr_list, a_mcc_list, a_specificity_list, a_npv_list = dbd_sanity_test(dbd3_a_list, dbd3_a_templates_5a_dict, verbose=True, iteration_per_protein=30)
print("\n\n")

print("DBD3: Antigen-Bound Antibody (AB)")
ab_ranking_list, ab_precision_list, ab_recall_list, ab_auc_roc_list, ab_auc_pr_list, ab_mcc_list, ab_specificity_list, ab_npv_list = dbd_sanity_test(dbd3_ab_list, dbd3_ab_templates_5a_dict, verbose=True, iteration_per_protein=30)
print("\n\n")

print("DBD3: Others (O)")
o_ranking_list, o_precision_list, o_recall_list, o_auc_roc_list, o_auc_pr_list, o_mcc_list, o_specificity_list, o_npv_list = dbd_sanity_test(dbd3_o_list, dbd3_o_templates_5a_dict, verbose=True, iteration_per_protein=30)
print("\n\n")

print("DBD3: All Datasets")
dbd3_ranking_list = e_ranking_list + a_ranking_list + ab_ranking_list + o_ranking_list
dbd3_precision_list = e_precision_list + a_precision_list + ab_precision_list + o_precision_list
dbd3_recall_list = e_recall_list + a_recall_list + ab_recall_list + o_recall_list
dbd3_auc_roc_list = e_auc_roc_list + a_auc_roc_list + ab_auc_roc_list + o_auc_roc_list
dbd3_auc_pr_list = e_auc_pr_list + a_auc_pr_list + ab_auc_pr_list + o_auc_pr_list
dbd3_mcc_list = e_mcc_list + a_mcc_list + ab_mcc_list + o_mcc_list
dbd3_specificity_list = e_specificity_list + a_specificity_list + ab_specificity_list + o_specificity_list
dbd3_npv_list = e_npv_list + a_npv_list + ab_npv_list + o_npv_list

print("Mean Precision:", np.mean(dbd3_precision_list))
print("Mean Recall:", np.mean(dbd3_recall_list))
print("Mean AUC-ROC:", np.mean(dbd3_auc_roc_list))
print("Mean AUC-PR:", np.mean(dbd3_auc_pr_list))
print("Mean MCC:", np.mean(dbd3_mcc_list))
print("Mean Specificity:", np.mean(dbd3_specificity_list))
print("Mean NPV:", np.mean(dbd3_npv_list))

"""**GA Parameter Tuning Execution**"""

# run_grid_search(["1E4K"], dbd5_all_templates_dict)

"""**Same Protein Family Execution**"""

# # DBD5 Same Protein Family for each complex category labels
# print("DBD5: Enzyme-Inhibitor (EI)")
# ei_ranking_list, ei_precision_list, ei_recall_list, ei_auc_roc_list, ei_auc_pr_list, ei_mcc_list, ei_specificity_list, ei_npv_list = dbd_all_templates(dbd5_ei_list, dbd5_ei_templates_dict, verbose=True)
# print("\n\n")

# print("DBD5: Enzyme-Substrate (ES)")
# es_ranking_list, es_precision_list, es_recall_list, es_auc_roc_list, es_auc_pr_list, es_mcc_list, es_specificity_list, es_npv_list = dbd_all_templates(dbd5_es_list, dbd5_es_templates_dict, verbose=True)
# print("\n\n")

# print("DBD5: Enzyme complex with a regulatory or accessory chain (ER)")
# er_ranking_list, er_precision_list, er_recall_list, er_auc_roc_list, er_auc_pr_list, er_mcc_list, er_specificity_list, er_npv_list = dbd_all_templates(dbd5_er_list, dbd5_er_templates_dict, verbose=True)
# print("\n\n")

# print("DBD5: Antibody-Antigen (A)")
# a_ranking_list, a_precision_list, a_recall_list, a_auc_roc_list, a_auc_pr_list, a_mcc_list, a_specificity_list, a_npv_list = dbd_all_templates(dbd5_a_list, dbd5_a_templates_dict, verbose=True)
# print("\n\n")

# print("DBD5: Antigen-Bound Antibody (AB)")
# ab_ranking_list, ab_precision_list, ab_recall_list, ab_auc_roc_list, ab_auc_pr_list, ab_mcc_list, ab_specificity_list, ab_npv_list = dbd_all_templates(dbd5_ab_list, dbd5_ab_templates_dict, verbose=True)
# print("\n\n")

# print("DBD5: Others, G-protein containing (OG)")
# og_ranking_list, og_precision_list, og_recall_list, og_auc_roc_list, og_auc_pr_list, og_mcc_list, og_specificity_list, og_npv_list = dbd_all_templates(dbd5_og_list, dbd5_og_templates_dict, verbose=True)
# print("\n\n")

# print("DBD5: Others, Receptor containing (OR)")
# or_ranking_list, or_precision_list, or_recall_list, or_auc_roc_list, or_auc_pr_list, or_mcc_list, or_specificity_list, or_npv_list = dbd_all_templates(dbd5_or_list, dbd5_or_templates_dict, verbose=True)
# print("\n\n")

# print("DBD5: Others, miscellaneous (OX)")
# ox_ranking_list, ox_precision_list, ox_recall_list, ox_auc_roc_list, ox_auc_pr_list, ox_mcc_list, ox_specificity_list, ox_npv_list = dbd_all_templates(dbd5_ox_list, dbd5_ox_templates_dict, verbose=True)
# print("\n\n")

# print("DBD5: All Datasets")
# dbd5_ranking_list = ei_ranking_list + es_ranking_list + er_ranking_list + a_ranking_list + ab_ranking_list + og_ranking_list + or_ranking_list + ox_ranking_list
# dbd5_precision_list = ei_precision_list + es_precision_list + er_precision_list + a_precision_list + ab_precision_list + og_precision_list + or_precision_list + ox_precision_list
# dbd5_recall_list = ei_recall_list + es_recall_list + er_recall_list + a_recall_list + ab_recall_list + og_recall_list + or_recall_list + ox_recall_list
# dbd5_auc_roc_list = ei_auc_roc_list + es_auc_roc_list + er_auc_roc_list + a_auc_roc_list + ab_auc_roc_list + og_auc_roc_list + or_auc_roc_list + ox_auc_roc_list
# dbd5_auc_pr_list = ei_auc_pr_list + es_auc_pr_list + er_auc_pr_list + a_auc_pr_list + ab_auc_pr_list + og_auc_pr_list + or_auc_pr_list + ox_auc_pr_list
# dbd5_mcc_list = ei_mcc_list + es_mcc_list + er_mcc_list + a_mcc_list + ab_mcc_list + og_mcc_list + or_mcc_list + ox_mcc_list
# dbd5_specificity_list = ei_specificity_list + es_specificity_list + er_specificity_list + a_specificity_list + ab_specificity_list + og_specificity_list + or_specificity_list + ox_specificity_list
# dbd5_npv_list = ei_npv_list + es_npv_list + er_npv_list + a_npv_list + ab_npv_list + og_npv_list + or_npv_list + ox_npv_list

# print("Mean Precision:", np.mean(dbd5_precision_list))
# print("Mean Recall:", np.mean(dbd5_recall_list))
# print("Mean AUC-ROC:", np.mean(dbd5_auc_roc_list))
# print("Mean AUC-PR:", np.mean(dbd5_auc_pr_list))
# print("Mean MCC:", np.mean(dbd5_mcc_list))
# print("Mean Specificity:", np.mean(dbd5_specificity_list))
# print("Mean NPV:", np.mean(dbd5_npv_list))

# # DBD3 Same Protein Family for each Complex Category Labels
# print("DBD3: Enzyme-Inhibitor or Enzyme-Substrate (E)")
# e_ranking_list, e_precision_list, e_recall_list, e_auc_roc_list, e_auc_pr_list, e_mcc_list, e_specificity_list, e_npv_list = dbd_all_templates(dbd3_e_list[:1], dbd3_e_templates_dict, verbose=True)
# print("\n\n")

# print("DBD3: Antibody-Antigen (A)")
# a_ranking_list, a_precision_list, a_recall_list, a_auc_roc_list, a_auc_pr_list, a_mcc_list, a_specificity_list, a_npv_list = dbd_all_templates(dbd3_a_list[:1], dbd3_a_templates_dict, verbose=True)
# print("\n\n")

# print("DBD3: Antigen-Bound Antibody (AB)")
# ab_ranking_list, ab_precision_list, ab_recall_list, ab_auc_roc_list, ab_auc_pr_list, ab_mcc_list, ab_specificity_list, ab_npv_list = dbd_all_templates(dbd3_ab_list, dbd3_ab_templates_dict, verbose=True)
# print("\n\n")

# print("DBD3: Others (O)")
# o_ranking_list, o_precision_list, o_recall_list, o_auc_roc_list, o_auc_pr_list, o_mcc_list, o_specificity_list, o_npv_list = dbd_all_templates(dbd3_o_list, dbd3_o_templates_dict, verbose=True)
# print("\n\n")

# print("DBD3: All Datasets")
# dbd3_ranking_list = e_ranking_list + a_ranking_list + ab_ranking_list + o_ranking_list
# dbd3_precision_list = e_precision_list + a_precision_list + ab_precision_list + o_ranking_list
# dbd3_recall_list = e_recall_list + a_recall_list + ab_recall_list + o_recall_list
# dbd3_auc_roc_list = e_auc_roc_list + a_auc_roc_list + ab_auc_roc_list + o_auc_roc_list
# dbd3_auc_pr_list = e_auc_pr_list + a_auc_pr_list + ab_auc_pr_list + o_auc_pr_list
# dbd3_mcc_list = e_mcc_list + a_mcc_list + ab_mcc_list + o_mcc_list
# dbd3_specificity_list = e_specificity_list + a_specificity_list + ab_specificity_list + o_specificity_list
# dbd3_npv_list = e_npv_list + a_npv_list + ab_npv_list + o_npv_list

# print("Mean Precision:", np.mean(dbd3_precision_list))
# print("Mean Recall:", np.mean(dbd3_recall_list))
# print("Mean AUC-ROC:", np.mean(dbd3_auc_roc_list))
# print("Mean AUC-PR:", np.mean(dbd3_auc_pr_list))
# print("Mean MCC:", np.mean(dbd3_mcc_list))
# print("Mean Specificity:", np.mean(dbd3_specificity_list))
# print("Mean NPV:", np.mean(dbd3_npv_list))

"""**All Templates Execution**"""

# _, _, _, _, _, _, _, _ = dbd_all_templates(dbd5_all_list, dbd5_all_templates_dict, verbose=True)

# _, _, _, _, _, _, _, _ = dbd_all_templates(dbd3_all_list, dbd3_all_templates_dict, verbose=True)

# _, _, _, _, _, _, _, _ = dbd_all_templates(dbd3_e_list[:1], dbd3_e_templates_dict, verbose=True)

num_sec = int(time.time() - start_time)
num_hours = math.floor(num_sec / (60 * 60))
num_sec = num_sec % (60 * 60)
num_minutes = math.floor(num_sec / 60)
num_sec = num_sec % 60
print("Executed in ", num_hours, "hours,", num_minutes, "minutes,", num_sec, "seconds")
