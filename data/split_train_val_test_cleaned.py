import os
import random
import argparse

def read_identifiers_from_file(file_path):
    with open(file_path, 'r') as file:
        return set(line.strip() for line in file)

def save_identifiers_to_file(identifiers, file_path):
    with open(file_path, 'w') as file:
        for identifier in sorted(identifiers):
            file.write(identifier + '\n')

def extract_and_randomize_identifiers(directory, output_file):
    """
    Extracts unique identifiers from .pdb files in a specified directory, randomizes them,
    and writes them to a text file.
    
    Parameters:
    - directory: str. The path to the directory containing the .pdb files.
    - output_file: str. The path to the text file where randomized identifiers will be written.

    Returns:
    - None. Writes randomized identifiers to a specified text file.
    """
    identifiers = set()

    # Ensure the directory exists
    if not os.path.exists(directory):
        print("The specified directory does not exist.")
        return

    # Iterate through each file in the directory
    for filename in os.listdir(directory):
        if filename.endswith('.pdb'):
            # Extract the identifier before the first underscore
            identifier = filename.split('_')[0]
            identifiers.add(identifier)
    
    # Convert set to list and shuffle
    identifier_list = list(identifiers)
    random.shuffle(identifier_list)
    
    # Write the randomized identifiers to a text file
    with open(output_file, 'w') as file:
        for identifier in identifier_list:
            file.write(identifier + '\n')

def main(args):
    # Process .pdb files to extract and randomize identifiers
    extract_and_randomize_identifiers(args.pdb_directory, args.test_identifiers_file)

    # Read test identifiers from the same file
    test_identifiers = read_identifiers_from_file(args.test_identifiers_file)

    # Collect all subfolder names from the specified directory
    all_identifiers = {folder for folder in os.listdir(args.base_dir) if os.path.isdir(os.path.join(args.base_dir, folder))}

    # Ensure all test identifiers are present in the directory
    if not test_identifiers.issubset(all_identifiers):
        print("Error: Some identifiers in the test file are missing in the directory.")
        return

    # Remove test identifiers from all_identifiers
    remaining_identifiers = list(all_identifiers - test_identifiers)

    # Randomly shuffle the remaining identifiers
    random.shuffle(remaining_identifiers)

    # Calculate the number of items for each set
    total_count = len(remaining_identifiers) + len(test_identifiers)
    test_count = max(len(test_identifiers), int(total_count * 0.1))
    val_count = int(total_count * 0.1)
    train_count = total_count - test_count - val_count

    # Ensure test set has enough identifiers, adjusting from the remaining identifiers if needed
    if len(test_identifiers) < test_count:
        extra_needed = test_count - len(test_identifiers)
        test_identifiers.update(set(remaining_identifiers[:extra_needed]))
        remaining_identifiers = remaining_identifiers[extra_needed:]

    # Update counts for training and validation sets
    train_identifiers = remaining_identifiers[:train_count]
    val_identifiers = remaining_identifiers[train_count:train_count + val_count]

    # Save identifiers to files
    save_identifiers_to_file(train_identifiers, args.train_file)
    save_identifiers_to_file(val_identifiers, args.val_file)
    save_identifiers_to_file(test_identifiers, args.test_file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process .pdb files and split identifiers into datasets.")
    parser.add_argument('--pdb_directory', default='zincbind_cleaned/zincbind_test', type=str,  help='Directory containing .pdb files to process.')
    parser.add_argument('--base_dir', default='zincbind_cleaned_processed', type=str,  help='Base directory containing subfolders for splitting.')
    parser.add_argument('--test_identifiers_file', default='splits/test_metal3d.txt', type=str, help='File for test identifiers and output from .pdb processing.')
    parser.add_argument('--train_file', type=str, default='splits/train_cleaned.txt', help='Output file for training identifiers.')
    parser.add_argument('--val_file', type=str, default='splits/val_cleaned.txt', help='Output file for validation identifiers.')
    parser.add_argument('--test_file', type=str, default='splits/test_cleaned.txt', help='Output file for test identifiers.')
    args = parser.parse_args()

    main(args)
