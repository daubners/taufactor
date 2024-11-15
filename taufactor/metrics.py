import numpy as np
import torch
import torch.nn.functional as F
import warnings

from scipy.ndimage import convolve
from skimage import measure


def volume_fraction(img, phases={}, device=torch.device('cuda')):
    """
    Calculates volume fractions of phases in an image
    :param img: segmented input image with n phases
    :param phases: a dictionary of phases to be calculated with keys as labels and phase values as values, default empty
    :return: list of volume fractions if no labels, dictionary of label: volume fraction pairs if labelled
    """

    if type(img) is not type(torch.tensor(1)):
        img = torch.tensor(img)

    if phases=={}:
        volume = torch.numel(img)
        labels, counts = torch.unique(img, return_counts=True)
        labels = labels.int()
        counts = counts.float()
        counts /= volume
        vf_out = {}
        for i, label in enumerate(labels):
            vf_out[str(label.item())] = counts[i].item()
    else:
        vf_out={}
        for p in phases:
            vf_out[p]=(img==phases[p]).to(torch.float).mean().item()

    return vf_out

def crop_area_of_interest_torch(tensor, labels):
    indices = torch.nonzero(torch.isin(tensor, labels), as_tuple=True)
    min_idx = [torch.min(idx).item() for idx in indices]
    max_idx = [torch.max(idx).item() for idx in indices]

    # Slice the tensor to the bounding box
    # Make sure to stay inside the bounds of total array
    sub_tensor = tensor[max(min_idx[0]-3,0):min(max_idx[0]+4,tensor.shape[0]),
                        max(min_idx[1]-3,0):min(max_idx[1]+4,tensor.shape[1]),
                        max(min_idx[2]-3,0):min(max_idx[2]+4,tensor.shape[2])]
    return sub_tensor

def crop_area_of_interest_numpy(array, labels):
    indices = np.nonzero(np.isin(array, labels))
    min_idx = [np.min(idx) for idx in indices]
    max_idx = [np.max(idx) for idx in indices]

    # Slice the array to the bounding box
    # Make sure to stay inside the bounds of the total array
    sub_array = array[max(min_idx[0]-3, 0):min(max_idx[0]+4, array.shape[0]),
                      max(min_idx[1]-3, 0):min(max_idx[1]+4, array.shape[1]),
                      max(min_idx[2]-3, 0):min(max_idx[2]+4, array.shape[2])]
    return sub_array

def gaussian_kernel_3d_torch(size=3, sigma=1.0, device=torch.device('cuda')):
    """Creates a 3D Gaussian kernel using PyTorch"""
    ax = torch.linspace(-(size // 2), size // 2, size)
    xx, yy, zz = torch.meshgrid(ax, ax, ax, indexing="ij")

    # Calculate Gaussian function for each point in the grid
    kernel = torch.exp(-(xx**2 + yy**2 + zz**2) / (2 * sigma**2))
    kernel /= kernel.sum()
    kernel = kernel.to(device)
    return kernel.unsqueeze(0).unsqueeze(0)

def gaussian_kernel_3d_numpy(size=3, sigma=1.0):
    """Creates a 3D Gaussian kernel using NumPy"""
    ax = np.linspace(-(size // 2), size // 2, size)
    xx, yy, zz = np.meshgrid(ax, ax, ax)

    # Calculate Gaussian function for each point in the grid
    kernel = np.exp(-(xx**2 + yy**2 + zz**2) / (2 * sigma**2))
    kernel /= np.sum(kernel)
    return kernel

def specific_surface_area(img, spacing, phases={}, method='gradient', device=torch.device('cuda'), smoothing=True):
    """
    Calculate the specific surface area of all (specified) phases
    :param img: labelled microstructure where each integer value represents a phase
    :param spacing: voxel size in each dimension [dx,dy,dz]
    :param phases: dictionary of phases {'name': label, ...}. If empty do all by default.
    :param periodic: list of bools indicating if the image is periodic in each dimension
    :return: the surface area in faces per unit volume
    """
    [dx,dy,dz] = spacing
    surface_areas = {}

    if (method == 'gradient') | (method == 'faces'):
        if type(img) is not type(torch.tensor(1)):
            tensor = torch.tensor(img)
        else:
            tensor = img
        tensor = tensor.to(device)

    if method == 'gradient':
        if phases=={}:
            labels = torch.unique(tensor)
            labels = labels.int()
            phases = {str(label.item()): label.item() for label in labels}
        gaussian = gaussian_kernel_3d_torch(device=device)

        volume = torch.numel(tensor)
        for name, label in phases.items():
            sub_tensor = crop_area_of_interest_torch(tensor, label)
            # Create binary mask for the label within the slice
            mask = (sub_tensor == label).float()
            if smoothing:
                mask = mask.unsqueeze(0).unsqueeze(0)
                mask = F.pad(mask, (1,1,1,1,1,1), mode='reflect')
                mask = F.conv3d(mask, gaussian, padding='valid')
                mask = mask.squeeze()

            grad = torch.gradient(mask, spacing=(dx,dy,dz))
            norm2 = grad[0].pow(2) + grad[1].pow(2) + grad[2].pow(2)
            surface_area = torch.sum(torch.sqrt(norm2)).item()

            surface_areas[name] = surface_area / volume

    elif method == 'faces':
        tensor = tensor.to(torch.int32)

        # TODO: treat dimensions such that dx!=dz is accounted for
        volume = torch.numel(tensor)*dx
        phasepairs = torch.tensor([[0,0]], device=device)
        # for i in range(3):
        #     shifted = torch.roll(tensor, 1, i)
        #     neighbour_idx = torch.nonzero(tensor != torch.roll(tensor, j, i), as_tuple=True)
        #     print(neighbour_idx)
        #     neighbour_list = torch.stack([tensor[neighbour_idx], shifted[neighbour_idx]])
        #     phasepairs = torch.cat((phasepairs,torch.transpose(neighbour_list,0,1)), 0)
        neighbour_idx = torch.nonzero(tensor[:-1,:,:] != tensor[1:,:,:], as_tuple=True)
        neighbour_list = torch.stack([tensor[:-1,:,:][neighbour_idx], tensor[1:,:,:][neighbour_idx]])
        phasepairs = torch.cat((phasepairs,torch.transpose(neighbour_list,0,1)), 0)
        neighbour_idx = torch.nonzero(tensor[:,:-1,:] != tensor[:,1:,:], as_tuple=True)
        neighbour_list = torch.stack([tensor[:,:-1,:][neighbour_idx], tensor[:,1:,:][neighbour_idx]])
        phasepairs = torch.cat((phasepairs,torch.transpose(neighbour_list,0,1)), 0)
        neighbour_idx = torch.nonzero(tensor[:,:,:-1] != tensor[:,:,1:], as_tuple=True)
        neighbour_list = torch.stack([tensor[:,:,:-1][neighbour_idx], tensor[:,:,1:][neighbour_idx]])
        phasepairs = torch.cat((phasepairs,torch.transpose(neighbour_list,0,1)), 0)

        # Crop initial dummy values
        phasepairs = phasepairs[1:]

        if phases=={}:
            labels, counts = torch.unique(phasepairs, return_counts=True)
            labels = labels.int()
            counts = counts.float()
            counts /= volume
            surface_areas = {str(label.item()): counts[i].item() for i, label in enumerate(labels)}
        else:
            for name, label in phases.items():
                count = torch.sum((phasepairs == label).int()).item()
                surface_areas[name] = count / volume

    elif method == 'marching':
        if device != 'cpu':
            warnings.warn("The marching cubes algorithm is performed on the CPU based on scikit-image package.")
        if dx != dy | dx!= dz | dy!=dz:
            raise ValueError("Surface area computation based on marching cubes assumes dx=dy=dz.")

        if type(img) is type(torch.tensor(1)):
            array = np.array(img.cpu())
        else:
            array = img

        if phases=={}:
            labels = np.unique(array).astype(int)
            phases = {str(label): label for label in labels}

        volume = array.size*dx
        gaussian = gaussian_kernel_3d_numpy(size=3, sigma=1.0)
        for name, label in phases.items():
            sub_array = crop_area_of_interest_numpy(array, label)
            sub_array = (sub_array == label).astype(float)
            if smoothing:
                sub_array = convolve(sub_array, gaussian, mode='nearest')
            vertices, faces, _, _ = measure.marching_cubes(sub_array, 0.5, method='lewiner')
            surface_area = measure.mesh_surface_area(vertices, faces)
            surface_areas[name] = surface_area/volume

    else:
        raise ValueError("Choose method\n 'gradient' for fast phase-field approach\n 'faces' for face counting or\n 'marching' for marching cubes method.")

    return surface_areas

def triple_phase_boundary(img):
    """Calculate triple phase boundary density i.e. fraction of voxel verticies that touch at least 3 phases

    Args:
        img (numpy array): image to calculate metric on
    Returns:
        float: triple phase boundary density 
    """
    phases = torch.unique(torch.tensor(img))
    if len(phases)!=3:
        raise ValueError('Image must have exactly 3 phases')
    shape = img.shape
    dim = len(shape)
    ph_maps = []
    img = F.pad(torch.tensor(img), (1,)*dim*2, 'constant', value=-1)
    if dim==2:
        x, y = shape
        total_edges = (x-1)*(y-1)
        for ph in phases:
            ph_map = torch.zeros_like(img)
            ph_map_temp = torch.zeros_like(img)
            ph_map_temp[img==ph] = 1
            for i in [0, 1]:
                for j in [0, 1]:
                    ph_map += torch.roll(torch.roll(ph_map_temp, i, 0), j, 1)
            ph_maps.append(ph_map)
        tpb_map = torch.ones_like(img)
        for ph_map in ph_maps:
            tpb_map *= ph_map
        tpb_map[tpb_map>1] = 1
        tpb_map = tpb_map[1:-1, 1:-1]
        tpb = torch.sum(tpb_map)
    else:
        tpb = 0
        x, y, z = shape
        total_edges = z*(x-1)*(y-1) + x*(y-1)*(z-1) + y*(x-1)*(z-1)
        print(total_edges)
        for d in range(dim):
            ph_maps = []
            for ph in phases:
                ph_map = torch.zeros_like(img)
                ph_map_temp = torch.zeros_like(img)
                ph_map_temp[img==ph] = 1
                for i in [0, 1]:
                    for j in [0, 1]:
                        d1 =( d + 1) % 3
                        d2 = (d + 2) % 3
                        ph_map += torch.roll(torch.roll(ph_map_temp, i, d1), j, d2)
                ph_maps.append(ph_map)
            tpb_map = torch.ones_like(img)
            for ph_map in ph_maps:
                tpb_map *= ph_map
            tpb_map[tpb_map>1] = 1
            tpb_map = tpb_map[1:-1, 1:-1, 1:-1]
            tpb += torch.sum(tpb_map)

    return tpb/total_edges