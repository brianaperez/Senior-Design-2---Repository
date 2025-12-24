import torch
import os
import cv2
import numpy as np
from torchvision import transforms
from .model.unet import UNet
from .utils.dataloader import LaneDataset
from PIL import Image
'''
# Establish paths
test_folder = "data/custom/inputs" # Folder containing test images of road in Sonoma Simulation
output_folder = "data/custom/outputs" # Folder to save output masks
os.makedirs(output_folder, exist_ok=True) 
'''
# Establish device (similar to train.py line)
device = "cuda" if torch.cuda.is_available() else "cpu"
weight_path = "/home/ubuntu/workspace/ros2_ws/src/perception_ros/Perception/epoch_21.pt"

# Loading model using the best epoch weights from training (epoch_26.pt)
model = UNet()
checkpoint = torch.load(weight_path, map_location=device) #Load best epoch model from training
model.load_state_dict(checkpoint["model"])
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.ToPILImage(), #Converts Numpy array to PIL Image; Resize needs PIL Image input
    transforms.Resize((180, 330)), #Resizing image to training resolutions given in train.py
    transforms.ToTensor() # Converts image to a PyTorch tensor
])

dummy_dataset = LaneDataset([], [])

# Evaluation function definition
def evaluate(model, image_bgr: np.ndarray, thresh: float = 0.4) -> np.ndarray: #expects a given model and image path as parameters

    # Load image from path
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY) # Converts 3-Channel BGR image to a single-channel grayscale image

    # Edge channels (same as LaneDataset from dataloader.py)
    edges, edges_inv = dummy_dataset.find_edge_channel(image_bgr) # Using find edge channel method from dataloader with dummy dataset access to dataloder.py methods

    # Construct 3-channel input (matching from LaneDataset in dataloader.py)
    output_image = np.zeros((gray.shape[0], gray.shape[1], 3), dtype=np.uint8) # Creates empty image with same height/width as gray image but with 3 channels
    output_image[:, :, 0] = gray # First channel is grayscale or grayscale intensity
    output_image[:, :, 1] = edges # Second channel is edge-detected image (lane boundaries and road markings)
    output_image[:, :, 2] = edges_inv # Third channel is inverse edge-detected image (helps with differentiating lanes from road)

    # Apply same transform
    img_tensor = transform(output_image).unsqueeze(0).to(device) # Unsqueeze adds a batch dimension at index 0 moving other indexes up by 1

    with torch.no_grad(): # No gradient calculation needed for evaluation
        output = model(img_tensor) # Feeding preprocessed image through the model; Results in logit output for each pixel
        output = torch.sigmoid(output) # Converts logits to probability in range of [0,1]
        mask = (output > thresh).float()   # thresholding at 0.5 to get binary mask

    # Convert to PIL for saving
    #mask_img = transforms.ToPILImage()(mask.squeeze(0).cpu()) #Removes batch dimension, moves tensor to CPU from GPU, converts tensor to PIL Image
    pred = mask.squeeze().cpu().numpy().astype(np.uint8) #(180x330) 0/1 return

    return pred # Return the masked image

'''
# ---------------- Run on test images ----------------
for fileName in os.listdir(test_folder): #for each fileName within input test folder
    if fileName.endswith(".png") or fileName.endswith(".jpg"): # Checks if file name is either png or jpg

        input_path = os.path.join(test_folder, fileName)    #declare input path by joining test folder path and file name
        output_path = os.path.join(output_folder, fileName) #declare output path by joining output folder path and file name

        mask = evaluate(model, input_path) #set output or mask image output of the evaluate function passing through the model and input image path containg all test images
        mask.save(output_path) #save output mask to output path

        print(f"Saved mask for {fileName} → {output_path}") #Logging saving images to output folder
'''