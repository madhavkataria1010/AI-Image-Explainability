#!/bin/bash
nvidia-smi
cd ../LLaVA_Finetune
pwd
python3 fix.py
# Define file paths
pwd                                                     
JSON_FILE="../../results/73_task1.json"                    ## FLAG ##
pwd # Path to your JSON file
SOURCE_IMAGES_FOLDER="../../test/images"                          ## FLAG ##
pwd
FAKE_IMAGES_FOLDER="test_adobe/fake"
RESULT_FOLDER="test_adobe/results_task2"                # Folder to store VLM outputs for each fake image

mkdir "$RESULT_FOLDER"
# Directory paths
input_dir="$FAKE_IMAGES_FOLDER"
output_dir="$RESULT_FOLDER"

# Count total number of files
total_files=$(ls "$input_dir"/*.png 2>/dev/null | wc -l)
counter=0

# Process each image
for image in "$input_dir"/*.png; do 
    # Increment counter
    counter=$((counter + 1))

    # Generate output file path
    output_file="$output_dir/$(basename "${image%.*}").txt"

    # Display progress
    echo "Processing image $counter/$total_files: $(basename "$image")"

    ## MIGHT NEED TO cd LLaVA/ ##
    ## cd LLaVA/

    # Run the Python command
    python3 -m llava.serve.cli \
        --model-path llava-ftmodel \
        --image-file "$image" \
        --prompt_question "The given image is AI generated. Give me reasons why it is AI generated. You can select from the following list of artifacts:
        - Inconsistent object boundaries
        - Discontinuous surfaces
        - Non-manifold geometries in rigid structures
        - Floating or disconnected components
        - Asymmetric features in naturally symmetric objects
        - Misaligned bilateral elements in animal faces
        - Irregular proportions in mechanical components
        - Texture bleeding between adjacent regions
        - Texture repetition patterns
        - Over-smoothing of natural textures
        - Artificial noise patterns in uniform surfaces
        - Unrealistic specular highlights
        - Inconsistent material properties
        - Metallic surface artifacts
        - Dental anomalies in mammals
        - Anatomically incorrect paw structures
        - Improper fur direction flows
        - Unrealistic eye reflections
        - Misshapen ears or appendages
        - Impossible mechanical connections
        - Inconsistent scale of mechanical parts
        - Physically impossible structural elements
        - Inconsistent shadow directions
        - Multiple light source conflicts
        - Missing ambient occlusion
        - Incorrect reflection mapping
        - Incorrect perspective rendering
        - Scale inconsistencies within single objects
        - Spatial relationship errors
        - Depth perception anomalies
        - Over-sharpening artifacts
        - Aliasing along high-contrast edges
        - Blurred boundaries in fine details
        - Jagged edges in curved structures
        - Random noise patterns in detailed areas
        - Loss of fine detail in complex structures
        - Artificial enhancement artifacts
        - Incorrect wheel geometry
        - Implausible aerodynamic structures
        - Misaligned body panels
        - Impossible mechanical joints
        - Distorted window reflections
        - Anatomically impossible joint configurations
        - Unnatural pose artifacts
        - Biological asymmetry errors
        - Regular grid-like artifacts in textures
        - Repeated element patterns
        - Systematic color distribution anomalies
        - Frequency domain signatures
        - Color coherence breaks
        - Unnatural color transitions
        - Resolution inconsistencies within regions
        - Unnatural Lighting Gradients
        - Incorrect Skin Tones
        - Fake depth of field
        - Abruptly cut off objects
        - Glow or light bleed around object boundaries
        - Ghosting effects: Semi-transparent duplicates of elements
        - Cinematization Effects
        - Excessive sharpness in certain image regions
        - Artificial smoothness
        - Movie-poster like composition of ordinary scenes
        - Dramatic lighting that defies natural physics
        - Artificial depth of field in object presentation
        - Unnaturally glossy surfaces
        - Synthetic material appearance
        - Multiple inconsistent shadow sources
        - Exaggerated characteristic features
        - Impossible foreshortening in animal bodies
        - Scale inconsistencies within the same object class" > "$output_file" 2>&1

    ## MIGHT NEED TO cd ../ ##
    
    ## cd ../

done

# Completion message
echo "Processing complete. Results saved in $output_dir."

python3 inference.py

