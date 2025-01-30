import argparse
import base64
import json
import os
import random
import re
from pathlib import Path
from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

key = os.getenv("OPENAI_API_KEY")
if not key:
    raise ValueError("OpenAI API key is not set. Please add it to the .env file.")

client = OpenAI(api_key=key)

ARTIFACTS_LIST = [
    "Inconsistent object boundaries",
    "Discontinuous surfaces",
    "Non-manifold geometries in rigid structures",
    "Floating or disconnected components",
    "Asymmetric features in naturally symmetric objects",
    "Misaligned bilateral elements in animal faces",
    "Irregular proportions in mechanical components",
    "Texture bleeding between adjacent regions",
    "Texture repetition patterns",
    "Over-smoothing of natural textures",
    "Artificial noise patterns in uniform surfaces",
    "Unrealistic specular highlights",
    "Inconsistent material properties",
    "Metallic surface artifacts",
    "Dental anomalies in mammals",
    "Anatomically incorrect paw structures",
    "Improper fur direction flows",
    "Unrealistic eye reflections",
    "Misshapen ears or appendages",
    "Impossible mechanical connections",
    "Inconsistent scale of mechanical parts",
    "Physically impossible structural elements",
    "Inconsistent shadow directions",
    "Multiple light source conflicts",
    "Missing ambient occlusion",
    "Incorrect reflection mapping",
    "Incorrect perspective rendering",
    "Scale inconsistencies within single objects",
    "Spatial relationship errors",
    "Depth perception anomalies",
    "Over-sharpening artifacts",
    "Aliasing along high-contrast edges",
    "Blurred boundaries in fine details",
    "Jagged edges in curved structures",
    "Random noise patterns in detailed areas",
    "Loss of fine detail in complex structures",
    "Artificial enhancement artifacts",
    "Incorrect wheel geometry",
    "Implausible aerodynamic structures",
    "Misaligned body panels",
    "Impossible mechanical joints",
    "Distorted window reflections",
    "Anatomically impossible joint configurations",
    "Unnatural pose artifacts",
    "Biological asymmetry errors",
    "Regular grid-like artifacts in textures",
    "Repeated element patterns",
    "Systematic color distribution anomalies",
    "Frequency domain signatures",
    "Color coherence breaks",
    "Unnatural color transitions",
    "Resolution inconsistencies within regions",
    "Unnatural Lighting Gradients",
    "Incorrect Skin Tones",
    "Fake depth of field",
    "Abruptly cut off objects",
    "Glow or light bleed around object boundaries",
    "Ghosting effects: Semi-transparent duplicates of elements",
    "Cinematization Effects",
    "Excessive sharpness in certain image regions",
    "Artificial smoothness",
    "Movie-poster like composition of ordinary scenes",
    "Dramatic lighting that defies natural physics",
    "Artificial depth of field in object presentation",
    "Unnaturally glossy surfaces",
    "Synthetic material appearance",
    "Multiple inconsistent shadow sources",
    "Exaggerated characteristic features",
    "Impossible foreshortening in animal bodies",
    "Scale inconsistencies within the same object class"
]

def generate_prompt():
    shuffled_artifacts = random.sample(ARTIFACTS_LIST, len(ARTIFACTS_LIST))
    artifacts_text = "\n- " + "\n- ".join(shuffled_artifacts)
    return f"""
You are an expert image analyzer excelling in analyzing fake images based on some parameters. This is an AI generated image. Please analyze it and return a JSON containing the artifact names and the reasons on the basis of this you would mark it fake. Return a JSON with keys as relevant artifact names. Do not make up any new artifact by yourself and be clear and confident with the reasons you have given. The artifacts to look for are as follows:{artifacts_text}
"""

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def analyze_image(image_path):
    try:
        base64_image = encode_image(image_path)
        dynamic_prompt = generate_prompt()
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": dynamic_prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ]
        )
        return {
            "image_path": str(image_path),
            "response": response.choices[0].message.content,
            "status": "success"
        }
    except Exception as e:
        return {
            "image_path": str(image_path),
            "error": str(e),
            "status": "failed"
        }

def process_images_and_format(input_dir, output_file):
    Path(input_dir).mkdir(parents=True, exist_ok=True)
    image_extensions = ('.png', '.jpg', '.jpeg', '.webp')
    image_files = [f for f in Path(input_dir).glob('*') if f.suffix.lower() in image_extensions]
    formatted_data = []
    image_id = 0  

    for image_file in tqdm(image_files, desc="Processing images", unit="image"):
        analysis_result = analyze_image(image_file)
        if analysis_result['status'] == 'success':
            try:
                match = re.search(r"```json\n(.*?)\n```", analysis_result['response'], re.DOTALL)
                if match:
                    response_dict = json.loads(match.group(1))
                    human_prompt = f"<image>\nThe given image is AI generated. Give me reasons why it is AI generated. You can select from the following list of artifacts:\n" + "\n".join(f"- {artifact}" for artifact in ARTIFACTS_LIST)
                    gpt_response = "\n".join(f"{key}: {reason}" for key, reason in response_dict.items())

                    formatted_data.append({
                        "id": image_id,
                        "image": str(image_file),
                        "conversations": [
                            {"from": "human", "value": human_prompt},
                            {"from": "gpt", "value": gpt_response}
                        ]
                    })
                    image_id += 1
                else:
                    raise ValueError("Invalid JSON format")
            except (json.JSONDecodeError, ValueError):
                print(f"Error parsing response for {analysis_result['image_path']}")
                if os.path.exists(analysis_result['image_path']):
                    os.remove(analysis_result['image_path'])
                    print(f"Deleted image: {analysis_result['image_path']}")
        else:
            print(f"Failed to analyze image: {analysis_result['image_path']}")

    with open(output_file, "w") as f:
        json.dump(formatted_data, f, indent=4)
    
    print(f"Formatted data saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Analyze AI-generated images for artifacts and format the results.")
    parser.add_argument("input_dir", type=str, help="Directory containing the images to analyze.")
    parser.add_argument("output_file", type=str, help="Path to save the output JSON file.")
    args = parser.parse_args()

    process_images_and_format(args.input_dir, f"{args.output_file}.json")

if __name__ == "__main__":
    main()