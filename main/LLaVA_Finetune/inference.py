import os
import json

# Path to the directory containing .txt files
directory_path = "test_adobe/results_task2"

# Function to process a single .txt file
def process_txt_file(file_path, index):
    explanation = {}
    start_reading = False

    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line.startswith("Assistant:"):
                start_reading = True
                line = line.replace("Assistant:", "").strip()
            
            if start_reading and line:
                if ":" in line:  # If line has an artifact name and explanation
                    artefact, explanation_text = map(str.strip, line.split(":", 1))
                    explanation[artefact] = explanation_text
            elif line == "------Over-------":
                break  # Stop processing if end marker is found
    
    return {"index": index, "explanation": explanation}

# Function to process all .txt files in the directory
def process_directory(directory_path):
    json_data = []
    for file_name in sorted(os.listdir(directory_path)):
        if file_name.endswith(".txt"):
            file_index = int(os.path.splitext(file_name)[0])  # Assuming file names are integers
            file_path = os.path.join(directory_path, file_name)
            json_data.append(process_txt_file(file_path, file_index))
    
    return json_data

# Main execution
if __name__ == "__main__":
    json_result = process_directory(directory_path)

    # Save the result as a JSON file
    output_path = "../../results/73_task2.json"
    with open(output_path, 'w') as json_file:
        json.dump(json_result, json_file, indent=4)
    
    print(f"Processed data has been saved to {output_path}")
