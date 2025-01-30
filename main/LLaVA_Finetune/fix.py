import json
import os
import shutil

def organize_fake_images(json_file, source_folder, output_folder):

    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Load JSON data
    with open(json_file, 'r') as f:
        predictions = json.load(f)
    
    # Iterate through the predictions
    for item in predictions:
        index = item["index"]
        prediction = item["prediction"]

        # Process only "Fake" predictions
        if prediction == "Fake":
            source_image = os.path.join(source_folder, f"{index}.png")
            target_image = os.path.join(output_folder, f"{index}.png")

            # Check if the source image exists
            if os.path.exists(source_image):
                shutil.copy(source_image, target_image)
                print(f"Copied: {source_image} -> {target_image}")
            else:
                print(f"Image not found: {source_image}")

if __name__ == "__main__":
    # Define file paths
    json_file_path = "../../results/73_task1.json"  # Replace with the path to your JSON file
    source_images_folder = "../../test/images"   # Replace with the folder containing your images
    fake_images_folder = "test_adobe/fake"     

    # Run the function
    organize_fake_images(json_file_path, source_images_folder, fake_images_folder)