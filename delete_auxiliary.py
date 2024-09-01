import os
import shutil

# Define the path to the "Exercises" directory
exercises_dir = "Exercises"

# Define the range of week numbers
week_start = 35
week_end = 47

# List of auxiliary LaTeX file extensions
auxiliary_extensions = ['.aux', '.log', '.out', '.toc', '.nav', '.snm', '.vrb', '.synctex.gz', '.bbl', '.blg', '.fdb.latexmk', '.fls']

# Loop through each week
for week in range(week_start, week_end + 1):
    week_folder = f"Week {week}"
    week_path = os.path.join(exercises_dir, week_folder)

    # Check if the week folder exists
    if os.path.exists(week_path) and os.path.isdir(week_path):
        print(f"Processing {week_folder}...")

        # Loop through all files in the week folder
        for root, dirs, files in os.walk(week_path):
            for file in files:
                file_path = os.path.join(root, file)
                
                # Check if the file is an auxiliary LaTeX file
                if any(file.endswith(ext) for ext in auxiliary_extensions):
                    try:
                        os.remove(file_path)
                        print(f"Deleted: {file_path}")
                    except Exception as e:
                        print(f"Failed to delete {file_path}: {e}")
    else:
        print(f"{week_folder} not found. Skipping...")

print("Cleanup complete.")