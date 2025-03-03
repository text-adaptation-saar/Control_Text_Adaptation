import os

# Define the paths to the major directories
major_directories = ["DiffWords", "Length", "MaxDepDepth", "MaxDepLength", "WordCount"]


# Function to merge the first line of CSV files from subdirectories of a major directory
def merge_csv_files(major_dir):
    # Create or open the output file for writing
    with open(f"{major_dir}_merged.csv", "w") as outfile:
        # Traverse through each subdirectory in the major directory
        for subdir, _, files in os.walk(major_dir):
            for file in files:
                if file == "success_rate.csv":
                    file_path = os.path.join(subdir, file)
                    # Open each success_rate.csv file for reading
                    with open(file_path, "r") as infile:
                        # Read the first line (header) and write it to the output file
                        first_line = infile.readline().strip()
                        outfile.write(first_line + "\n")

    print(f"Merged CSV saved for {major_dir}.")


# Iterate over each major directory and merge CSV files
for major_dir in major_directories:
    merge_csv_files(major_dir)