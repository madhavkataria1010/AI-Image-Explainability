# # import json

# # def convert_image_extensions(json_file_path, output_file_path):
# #     """
# #     Convert the extensions of image file names in a JSON file to `.png`.

# #     Args:
# #         json_file_path (str): Path to the input JSON file.
# #         output_file_path (str): Path to save the updated JSON file.
# #     """
# #     try:
# #         # Load the JSON file
# #         with open(json_file_path, 'r') as file:
# #             data = json.load(file)

# #         # Update the image extensions to .png
# #         for item in data:
# #             if "image" in item and isinstance(item["image"], str):
# #                 item["image"] = item["image"].rsplit('.', 1)[0] + ".png"

# #         # Save the updated JSON file
# #         with open(output_file_path, 'w') as file:
# #             json.dump(data, file, indent=4)

# #         print(f"Updated JSON file saved to: {output_file_path}")
# #     except Exception as e:
# #         print(f"An error occurred: {e}")

# # # Specify the paths
# # input_json_path = "final.json"  # Replace with the path to your input JSON file
# # output_json_path = "final_n.json"  # Replace with the path to save the output JSON file

# # # Run the function
# # convert_image_extensions(input_json_path, output_json_path)


# import json

# def update_image_paths(json_file_path, output_file_path):
#     """
#     Update the `image` paths in a JSON file to remove intermediate folders,
#     resulting in `Dice/image_name` format.

#     Args:
#         json_file_path (str): Path to the input JSON file.
#         output_file_path (str): Path to save the updated JSON file.
#     """
#     try:
#         # Load the JSON file
#         with open(json_file_path, 'r') as file:
#             data = json.load(file)

#         # Update the image paths
#         for item in data:
#             if "image" in item and isinstance(item["image"], str):
#                 parts = item["image"].split('/')
#                 if len(parts) >= 2:  # Ensure there are intermediate folders
#                     item["image"] = f"Dice/{parts[-1]}"

#         # Save the updated JSON file
#         with open(output_file_path, 'w') as file:
#             json.dump(data, file, indent=4)

#         print(f"Updated JSON file saved to: {output_file_path}")
#     except Exception as e:
#         print(f"An error occurred: {e}")

# # Specify the paths
# input_json_path = "final.json"  # Replace with the path to your input JSON file
# output_json_path = "final_n.json"  # Replace with the path to save the output JSON file

# # Run the function
# update_image_paths(input_json_path, output_json_path)

import os
import shutil

def copy_images_from_subfolders(source_folder, destination_folder):
    """
    Copy all image files from subfolders of the source folder to the destination folder.

    Args:
        source_folder (str): Path to the main folder containing subfolders with images.
        destination_folder (str): Path to the folder where all images will be copied.
    """
    try:
        # Create the destination folder if it does not exist
        if not os.path.exists(destination_folder):
            os.makedirs(destination_folder)

        # Traverse all subfolders and files within the source folder
        for root, _, files in os.walk(source_folder):
            for file in files:
                # Check if the file is an image
                if file.lower().endswith(('.png', '.jpeg', '.jpg', '.bmp', '.gif', '.tiff', '.webp')):
                    # Build full file paths
                    source_path = os.path.join(root, file)
                    destination_path = os.path.join(destination_folder, file)

                    # Copy the file to the destination folder
                    shutil.copy2(source_path, destination_path)

        print(f"All images have been successfully copied to: {destination_folder}")
    except Exception as e:
        print(f"An error occurred: {e}")

# Define the source and destination directories
source_folder = "Dice"  # Path to the Dice folder (replace with the actual path)
destination_folder = "X"  # Path to the destination folder

# Run the function
copy_images_from_subfolders(source_folder, destination_folder)