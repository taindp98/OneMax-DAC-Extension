from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import os


class Logger:
    def __init__(self, out_dir: str):
        log_dir = out_dir
        self.writer = SummaryWriter(log_dir)

    def log_scalar(self, tag, value, step):
        self.writer.add_scalar(tag, value, step)

    def log_dict(self, dictionary, step):
        for tag, value in dictionary.items():
            self.log_scalar(tag, value, step)

    def close(self):
        self.writer.close()

    def log_figure(self, tag, figure, step, out_dir):
        self.writer.add_image(tag, figure, step, dataformats="NCHW")
        figure = figure.squeeze(0)
        # Convert the tensor to a NumPy array
        figure_np = figure.permute(1, 2, 0).cpu().numpy()
        # Convert the NumPy array to a PIL image
        pil_image = Image.fromarray(figure_np)
        pil_image.save(os.path.join(out_dir, f"{tag}.png"))

    def log_dict(self, dictionary, step):
        for tag, value in dictionary.items():
            self.log_scalar(tag, value, step)
