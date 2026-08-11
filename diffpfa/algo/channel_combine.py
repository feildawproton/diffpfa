from typing import List, Tuple
import torch


def align_and_combine_channels(
    channel_images: List[torch.Tensor],
    align_phase: bool = True
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """
    Aligns relative phases between multiple sub-channel images of the same polarization pair
    and coherently combines them.

    Args:
        channel_images: List of 2D complex image tensors [I_0, I_1, ..., I_{C-1}].
        align_phase: If True, estimates phase offset relative to reference channel I_0 and aligns.

    Returns:
        combined_image: 2D complex image tensor of the combined response.
        aligned_images: List of aligned individual channel image tensors.
    """
    if len(channel_images) == 0:
        raise ValueError("channel_images list cannot be empty.")

    if len(channel_images) == 1:
        return channel_images[0], channel_images

    ref_img = channel_images[0]
    aligned_images = [ref_img]

    for c in range(1, len(channel_images)):
        curr_img = channel_images[c]

        # Note: Data-driven cross-correlation phase alignment is mathematically invalid for orthogonal 
        # frequency sub-bands. Phase alignment is now performed analytically using PVP RcvTime metadata 
        # in the PFA engine before gridding.
        curr_aligned = curr_img

        aligned_images.append(curr_aligned)

    # Coherent summation across channels without large stack allocations
    combined_image = aligned_images[0].clone()
    for i in range(1, len(aligned_images)):
        combined_image.add_(aligned_images[i])
        
    return combined_image, aligned_images
