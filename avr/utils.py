import torch
from torchvision import transforms
from panel_transforms import HorizontalFlip, VerticalFlip, RotateByAngle


grayscale_to_rgb_batch = transforms.Lambda(
    lambda x: torch.stack([a.reshape((1, a.shape[0], a.shape[1])).repeat(3, 1, 1) for a in x])
)

rpm_transforms = transforms.Compose([transforms.Resize((224, 224))])

def construct_rpm_from_patches(context_panels, choice_panel, masked_patch=False, transforms=None, augmentation=None):
    """ Constructs a whole RPM from the 8 context panels """

    if augmentation:
        context_panels = [augmentation(x) for x in context_panels]
        choice_panel = augmentation(choice_panel)

    row1 = [context_panels[0], context_panels[1], context_panels[2]]
    row2 = [context_panels[3], context_panels[4], context_panels[5]]

    if masked_patch:
        mask = torch.zeros((context_panels.shape[-3], context_panels.shape[-2], context_panels.shape[-1]))
        row3 = [context_panels[6], context_panels[7], mask]
    else:
        row3 = [context_panels[6], context_panels[7], choice_panel]

    row1 = torch.cat(row1, dim=1)
    row2 = torch.cat(row2, dim=1)
    row3 = torch.cat(row3, dim=1)

    rpm = torch.cat([row1, row2, row3], dim=2)
    if transforms:
        rpm = transforms(rpm)

    return rpm


def rpm_panels_to_img_samples(rpm_sample, target, augmentations=None):
    """ For one RPM problem, generates binary image samples. """

    q_sample = rpm_sample[:8, :, :]
    a_sample = rpm_sample[8:, :, :]

    q_sample = grayscale_to_rgb_batch(q_sample)
    a_sample = grayscale_to_rgb_batch(a_sample)

    binary_samples, binary_targets = [], []

    for i in range(8):
        if i == target:
            # Un-augmented sample
            binary_samples.append(construct_rpm_from_patches(q_sample, a_sample[i], transforms=rpm_transforms))
            binary_targets.append(1)

            if augmentations:
                for augmentation in augmentations:
                    binary_samples.append(construct_rpm_from_patches(q_sample, a_sample[i], transforms=rpm_transforms, augmentation=augmentation))
                    binary_targets.append(1)

        else:
            # Un-augmented sample
            binary_samples.append(construct_rpm_from_patches(q_sample, a_sample[i], transforms=rpm_transforms))
            binary_targets.append(0)

    binary_samples = torch.stack(binary_samples)
    binary_targets = torch.tensor(binary_targets)

    return binary_samples, binary_targets


def batch_to_bin_images(batch, target, augmentations=None, test_mode=False):
    """ Converts a batch to complete RPMs vs targets for binary classification.
    
    Parameters
    ----------
    batch: torch.Tensor
        A 4D batch tensor. Shape: (batch_size, 16, height, width)
        Eg. shape: (10, 16, 80, 80)
    target: torch.Tensor
        A 1D tensor of targets of the batch. Shape: (batch_size)
        Eg. shape: (10)
    augmentations: List[transforms]
        List of either torchvision.transforms or custom transforms.
        Functional transforms are not allowed; need to be called to be executed.
        Only executed if `test_mode = False`
        Eg. [RotateByAngle(90), RotateByAngle(-90)]
    test_mode: bool
        True if being called during validation or test phase. Different
        output format accordingly.

    Returns
    -------
    X_bin: torch.Tensor or List[torch.Tensor]
        A 4D tensor if not test mode else list of 3D torch tensors
    y_bin: torch.Tensor or List[int]
        A 1D tensor if not test mode else list of ints
    """

    X_bin, y_bin = [], []

    for x, y in zip(batch, target):
        if test_mode:
            binary_samples, _ = rpm_panels_to_img_samples(x, y, augmentations=None)

            X_bin.append(binary_samples)
            y_bin.append(y)
        else:
            binary_samples, binary_targets = rpm_panels_to_img_samples(x, y, augmentations=augmentations)

            X_bin += binary_samples
            y_bin += binary_targets

    if not test_mode:
        X_bin = torch.stack(X_bin)
        y_bin = torch.stack(y_bin)

        # Randomly shuffle training data
        rand_perm = torch.randperm(X_bin.size()[0])
        X_bin, y_bin = X_bin[rand_perm], y_bin[rand_perm]

    return X_bin, y_bin