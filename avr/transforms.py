from torchvision import transforms

class HorizontalFlip:

    def __init__(self, reshape=False):
        self.reshape = reshape

    def __call__(self, x):
        if self.reshape:
            x = x.reshape((1, x.shape[0], x.shape[1]))

        return transforms.functional.hflip(x)

class VerticalFlip:

    def __init__(self, reshape=False):
        self.reshape = reshape

    def __call__(self, x):
        if self.reshape:
            x = x.reshape((1, x.shape[0], x.shape[1]))

        return transforms.functional.vflip(x)

class RotateByAngle:

    def __init__(self, angle, reshape=False):
        self.angle = angle
        self.reshape = reshape

    def __call__(self, x):
        if self.reshape:
            x = x.reshape((1, x.shape[0], x.shape[1]))

        return transforms.functional.rotate(x, self.angle)