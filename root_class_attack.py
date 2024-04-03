# Root class attack

class Attack(object):
    '''
    Root class for all adversarial attack classes.
    '''

    def __init__(self, model, targeted=False, img_range=(0,1)):

        if img_range[0] >= img_range[1]:
            raise ValueError(
                ":img_range: Upper bound for pixel values must be greater than"
                " lower bound.")

        self.model = model
        self.device = next(model.parameters()).device
        self.targeted = targeted
        self.img_range = img_range

    def __repr__(self):
        return str(self.__dict__)

    def to(self, device):
        self.model.to(device)
        self.device = device

