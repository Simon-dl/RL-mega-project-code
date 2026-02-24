import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from model import UNet
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from matplotlib import pyplot as plt
import pickle




def get_dataset():

    def unpickle(file):
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict

    all_images = []
    first = True
    for i in range(1, 6):
        train_data = unpickle(f'DDPM/cifar10/data_batch_{i}')
        print(train_data.keys())
        images = train_data[b'data']
        # CIFAR-10 data is stored as (N, 3072) where each 3072 is organized as:
        # first 1024 = red channel, next 1024 = green channel, last 1024 = blue channel
        # So we need to reshape to (N, 3, 32, 32) first, then transpose to (N, 32, 32, 3)
        images = images.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        if first:
            all_images = images
            first = False
        else:
            all_images = np.concatenate([all_images, images], axis=0)
        # Ensure data type is uint8 for PIL

    print("dataset size: ",all_images.shape)

    return all_images


def normalize_images(dataset):
    """
    
    For batch processing of multiple images (e.g., shape (N, H, W, 3)), 
    apply normalization along axes (1, 2) to compute per-channel statistics across spatial dimensions.
    """
    # Convert to numpy if it's a tensor
    if isinstance(dataset, torch.Tensor):
        dataset = dataset.numpy()
    
    # Handle different input shapes: (N, C, H, W) or (N, H, W, C)
    if dataset.ndim == 4:
        # If shape is (N, C, H, W), normalize each image independently
        # Reshape to (N, H, W, C) for easier per-image processing, or keep as is
        normalized = np.zeros_like(dataset, dtype=np.float32)
        
        for i in range(dataset.shape[0]):
            img = dataset[i]
            img_min = img.min()
            img_max = img.max()
            # Avoid division by zero
            if img_max - img_min > 0:
                normalized[i] = 2 * ((img - img_min) / (img_max - img_min)) - 1
            else:
                normalized[i] = img  # If all values are the same, return as is
    else:
        # Single image case
        img_min = dataset.min()
        img_max = dataset.max()
        if img_max - img_min > 0:
            normalized = 2 * ((dataset - img_min) / (img_max - img_min)) - 1
        else:
            normalized = dataset
    
    return normalized

def denormalize_images(dataset, original_min, original_max):
    """
    Inverse of normalize_images
    
    For batch processing of multiple images.
    If original_min and original_max are scalars, applies the same range to all images.
    If they are arrays of shape (N,), applies per-image ranges.
    """
    # Convert to numpy if it's a tensor
    if isinstance(dataset, torch.Tensor):
        dataset = dataset.numpy()
    
    # Convert min/max to numpy arrays if they're tensors
    if isinstance(original_min, torch.Tensor):
        original_min = original_min.numpy()
    if isinstance(original_max, torch.Tensor):
        original_max = original_max.numpy()
    
    # Handle batch processing
    if dataset.ndim == 4:
        denormalized = np.zeros_like(dataset, dtype=np.float32)
        
        # Check if min/max are per-image (arrays) or global (scalars)
        if np.ndim(original_min) == 0 and np.ndim(original_max) == 0:
            # Global min/max - apply same range to all images
            for i in range(dataset.shape[0]):
                denormalized[i] = (dataset[i] + 1) / 2 * (original_max - original_min) + original_min
        elif np.ndim(original_min) == 1 and np.ndim(original_max) == 1:
            # Per-image min/max - shape should be (N,)
            assert len(original_min) == dataset.shape[0], "original_min length must match batch size"
            assert len(original_max) == dataset.shape[0], "original_max length must match batch size"
            for i in range(dataset.shape[0]):
                denormalized[i] = (dataset[i] + 1) / 2 * (original_max[i] - original_min[i]) + original_min[i]
        else:
            raise ValueError("original_min and original_max must be scalars or 1D arrays")
    else:
        # Single image case
        if np.ndim(original_min) == 0 and np.ndim(original_max) == 0:
            denormalized = (dataset + 1) / 2 * (original_max - original_min) + original_min
        else:
            raise ValueError("For single image, original_min and original_max must be scalars")
    
    return denormalized


def ddpm_update(x,t,tm1,eps_hat):
    """
    Just following the equation from here

    """
    alpha_t = torch.cos(t * (torch.pi / 2))
    sigma_t = torch.sin(t * (torch.pi / 2))

    alpha_tm1 = torch.cos(tm1 * (torch.pi / 2))
    sigma_tm1 = torch.sin(tm1 * (torch.pi / 2))


    eta = (sigma_tm1 / sigma_t) * torch.sqrt(1 - (alpha_t**2) / (alpha_tm1**2))

    x0_hat = torch.clip((x - sigma_t * eps_hat) / alpha_t, min=-1, max=1)
    term1 = alpha_tm1 * x0_hat
    term2 = torch.sqrt(torch.clamp(sigma_tm1**2 - eta**2, min=0)) * eps_hat  #clip this to 0 later
    term3 = eta * torch.randn_like(x)

    x_tm1 = term1 + term2 + term3
    return x_tm1

def ddpm_sampler(model,image_shape,num_steps=100,images_to_sample=10,grid = False):

    if grid:
        sample_list = np.linspace(0, num_steps - 1, 10).astype(int)
        sample_list = sample_list[1:]  # Remove first element if you don't want step 0
        print("sample list: ", sample_list)
        sample_grid = []
        first_sample = True

    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x = torch.randn((images_to_sample,) + image_shape).to(device)
    ts = torch.linspace(1 - 1e-4, 1e-4, num_steps + 1).to(x.device)

    with torch.no_grad():
        for i in tqdm(range(num_steps),desc="Sampling DDPM"):

            t = ts[i]
            t = t.unsqueeze(0)
        
            tm1 = ts[i+1]
            tm1 = tm1.unsqueeze(0)

            eps_hat = model(x,t)
            x = ddpm_update(x,t,tm1,eps_hat)


            if grid and i in sample_list:
                if first_sample:
                    sample_grid = x.unsqueeze(0)
                    first_sample = False
                else:
                    sample_grid = torch.cat([sample_grid, x.unsqueeze(0)], dim=0)
        
    if grid:
        sample_grid = torch.cat([sample_grid, x.unsqueeze(0)], dim=0)
        print("total sample grid shape: ", sample_grid.shape)
        return sample_grid.detach().cpu().numpy()
    else:
        return x.detach().cpu().numpy()



# def show_grid(sampled_grid, original_min, original_max):
#     """
#     Show a grid of sampled images, usually input size is (10,10,3,32,32)
    
#     Creates a 10x10 grid where each row represents a timestep and each column represents one of the 10 images.
#     """
#     fig, axes = plt.subplots(10, 10, figsize=(15, 15))
    
#     for i in range(sampled_grid.shape[0]):  # 10 timesteps
#         for j in range(sampled_grid.shape[1]):  # 10 images per timestep
#             sampled_image = sampled_grid[i, j]  # Get image at timestep i, image j
#             sampled_image = denormalize_images(sampled_image, original_min, original_max)
#             sampled_image = sampled_image.clip(0, 255).astype(np.uint8)
            
#             # Display the image in the appropriate subplot
#             axes[i, j].imshow(sampled_image.transpose(1, 2, 0))
#             axes[i, j].axis('off')  # Remove axes for cleaner display
    
#     plt.tight_layout()
#     plt.show()
    
def show_grid(sampled_grid, original_min, original_max):
    """
    Show a grid of sampled images, usually input size is (10,10,3,32,32)
    
    Creates a 10x10 grid where each row represents one image and each column represents a timestep.
    This shows each image denoising horizontally from left to right.
    """
    fig, axes = plt.subplots(10, 10, figsize=(15, 15))
    
    for i in range(sampled_grid.shape[1]):  # 10 images (rows)
        for j in range(sampled_grid.shape[0]):  # 10 timesteps (columns)
            sampled_image = sampled_grid[j, i]  # Get image i at timestep j
            sampled_image = denormalize_images(sampled_image, original_min, original_max)
            sampled_image = sampled_image.clip(0, 255).astype(np.uint8)
            
            # Display the image in the appropriate subplot
            # Row i shows image i, column j shows timestep j
            axes[i, j].imshow(sampled_image.transpose(1, 2, 0))
            axes[i, j].axis('off')  # Remove axes for cleaner display
    
    plt.tight_layout()
    plt.show()




def train_ddpm(dataset):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    original_min = 0
    original_max = 255

    epochs = 120
    batch_size = 128

    img_shape = dataset[0].shape

    loss_fn = nn.MSELoss(reduction='mean') #HW says its fine to take mean
    model = UNet(in_channels=3, hidden_dims=[64, 128, 256, 512], blocks_per_dim=2).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4) 
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    dataset = torch.tensor(dataset)
    dataset = TensorDataset(dataset)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


    
    mean_losses = []
    

    for epoch in tqdm(range(epochs),desc="Training DDPM"):
        loss_list = []

        pbar = tqdm(total=len(loader))
        count = 0
        for batch in loader: #put a progress bar here
            optimizer.zero_grad()
            img_batch = batch[0].to(device)

            #get a batch of t values
            t = torch.rand(img_batch.shape[0]).to(device)

            alpha_t = torch.cos(t * (torch.pi / 2)).view(-1, 1, 1, 1).to(device) #to make sure its broadcasted correctly
            sigma_t = torch.sin(t * (torch.pi / 2)).view(-1, 1, 1, 1).to(device)

            sample_noise = torch.randn_like(img_batch).to(device)
            noisy_image = alpha_t * img_batch + sigma_t * sample_noise   

            eps = model(noisy_image,t).to(device)

            loss = loss_fn(eps, sample_noise).to(device)
            loss.backward()
            optimizer.step()

            loss_list.append(loss.item())
            count += 1
            pbar.update(1)

        pbar.close()
        mean_loss = np.mean(loss_list)
        print("mean loss: ", mean_loss)
        mean_losses.append(mean_loss)
        scheduler.step()
        
    print("final loss: ", mean_losses[-1])

    torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'epoch': epoch,
    'scheduler_state_dict': scheduler.state_dict(),
    }, 'DDPM/model.pth')

    sampled_images = ddpm_sampler(model, img_shape, 1000, 10, grid=True)
    show_grid(sampled_images, original_min, original_max)

    return mean_losses


# car_image = Image.open('DDPM/car.jpg')
# car_image = np.array(car_image, dtype=np.float32)
# dataset_max = np.max(car_image)
# dataset_min = np.min(car_image)

# dataset = [car_image] * 100
# dataset = np.array(dataset).transpose(0, 3, 1, 2)
# print("dataset shape: ", dataset.shape)

dataset = get_dataset()
dataset = dataset.transpose(0, 3, 1, 2)
normalized_images = normalize_images(dataset)


mean_losses = train_ddpm(normalized_images)
x_axis = np.arange(1, len(mean_losses) + 1)
plt.plot(x_axis, mean_losses)
plt.xlabel('Epoch')
plt.ylabel('Mean Loss')
plt.title('Training Loss vs Epochs')
plt.grid(True)
plt.show()