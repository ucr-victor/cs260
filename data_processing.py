import os

# Specify the directory where router manuals are stored
data_dir = r"C:\Users\victor\router_data"
file_paths = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.txt')]

# Function to load and clean data from files
def load_and_clean_data(file_paths):
    data = []
    for path in file_paths:
        with open(path, 'r', encoding='utf-8') as file:
            content = file.read()
            # Clean data by removing newlines and unnecessary spaces
            cleaned_content = content.replace("\n", " ").strip()
            data.append(cleaned_content)
    return data

# Load and clean the data from the specified files
data = load_and_clean_data(file_paths)
print(f"Loaded data: {data[:3]}")  # Print the first 3 data entries
