# Create directories
New-Item -Path "datasets\tum" -ItemType Directory -Force

# Move to the target directory
Set-Location -Path "datasets\tum"

# Download and extract datasets
# TUM fr1/desk
Invoke-WebRequest -Uri "https://vision.in.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_desk.tgz" -OutFile "rgbd_dataset_freiburg1_desk.tgz"
tar -xvzf rgbd_dataset_freiburg1_desk.tgz

# TUM fr2/xyz
Invoke-WebRequest -Uri "https://vision.in.tum.de/rgbd/dataset/freiburg2/rgbd_dataset_freiburg2_xyz.tgz" -OutFile "rgbd_dataset_freiburg2_xyz.tgz"
tar -xvzf rgbd_dataset_freiburg2_xyz.tgz

# TUM fr3/office
Invoke-WebRequest -Uri "https://vision.in.tum.de/rgbd/dataset/freiburg3/rgbd_dataset_freiburg3_long_office_household.tgz" -OutFile "rgbd_dataset_freiburg3_long_office_household.tgz"
tar -xvzf rgbd_dataset_freiburg3_long_office_household.tgz

# Return to original directory
Set-Location -Path "..\..\"