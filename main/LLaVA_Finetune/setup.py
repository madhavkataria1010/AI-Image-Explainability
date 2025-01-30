import os

# Install necessary packages
os.system("pip install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html")
os.system("pip install transformers")
os.system("pip install deepspeed")
os.system("pip install wandb")

# Clone the LLaVA repository
os.system("git clone https://github.com/haotian-liu/LLaVA.git")

# Change directory to LLaVA
os.chdir("LLaVA")

print("Setup complete!")
