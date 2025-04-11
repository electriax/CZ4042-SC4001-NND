# CZ4042-SC4001-NND

# Instructions
  1. Ensure that system has python and CUDA enabled
`python --version`
`nvidia-smi`
  
  2.  Create virtual environment and download relevant packages

  `conda create -n nnd python=3.10 -y`
    `pip install -r requirements.txt`

  3. All relevant scripts are inside the architecture directory:

    1. resnet_online_aug_baseline.ipynb
    2. resnet_online_aug_34.ipynb
    3. resnet_online_aug_50.ipynb
    4. resnet_online_aug_crisscross_attention.py
    5. resnet_online_deformed_aug_conv.py 

  5. To run the scripts, edit the filepath to either CURRENT_DIR or MAIN_FOLDER according to current directory.
    `OUTPUT_FOLDER = os.path.join(MAIN_FOLDER, 'aligned') ` 
    `FOLD_DATA = os.path.join(MAIN_FOLDER, 'fold_data') `



