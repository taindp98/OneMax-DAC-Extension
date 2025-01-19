from torch.utils.tensorboard import SummaryWriter
import wandb
from PIL import Image
import os
import json
import numpy as np

class Logger:
    def __init__(
        self,
        config,
        use_wandb=False,
        project: str = "OneMaxDAC",
        save_dir: str = "outputs/logs",
    ):
        """
        Tools: Tensorboard, Wandb, JSON
        Args:
            config (dict): Configuration dictionary.
            use_wandb (bool): Whether to use Weights & Biases.
            project (str): The name of the project in Weights & Biases.
            save_dir (str): The directory to save logs
        """
        self.save_dir = save_dir
        self.use_wandb = use_wandb
        run_id = save_dir.split("/")[-1]
        if use_wandb:
            self.run = wandb.init(
                project=project,
                name=run_id,
                config=config,
            )
            # # define our custom x axis metric
            wandb.define_metric("episode")
            # # set all other train/ metrics to use this step
            wandb.define_metric("Loss/episode", step_metric="episode")
            wandb.define_metric("Reward/episode", step_metric="episode")
            self.rw_cnt = 0

        self.writer = SummaryWriter(self.save_dir)
        self.metrics = {}

        ## log JSON
        self.log_json_fpath = os.path.join(self.save_dir, "evaluations.json")

    def log_scalar(self, tag, value, step):
        """
        Log a scalar value.
        Args:
            tag (str): The name of the scalar.
            value (float): The value of the scalar.
            step (int): The step at which the scalar was recorded.
        """
        self.writer.add_scalar(tag, value, step)
        if self.use_wandb:
            if "episode" in tag:
                self.rw_cnt += 1
                wandb.log(
                    {
                        "episode": self.rw_cnt,
                        tag: value,
                    }
                )
            else:
                wandb.log({tag: value}, step=step)

    def log_figure(self, tag, figure, step, out_dir=None):
        """
        Log a figure.
        Args:
            tag (str): The name of the figure.
            figure (matplotlib.figure.Figure): The figure to log.
            step (int): The step at which the figure was recorded.
            out_dir (str): The directory to save the figure as a PNG file.
        """
        self.writer.add_image(tag, figure, step, dataformats="NCHW")
        figure = figure.squeeze(0)
        # Convert the tensor to a NumPy array
        figure_np = figure.permute(1, 2, 0).cpu().numpy()
        # Convert the NumPy array to a PIL image
        pil_image = Image.fromarray(figure_np)
        pil_image.save(os.path.join(out_dir, f"{tag}.png"))
        if self.use_wandb:
            wandb.log({tag: wandb.Image(pil_image)}, step=step)

    def log_dict(self, dictionary, step):
        for tag, value in dictionary.items():
            self.log_scalar(tag, value, step)

    def close(self):
        self.writer.close()
        if self.use_wandb:
            self.run.finish()

    def update(self, **kwargs):
        """
        Update metrics with new values.
        Args:
            kwargs: Metric names and their corresponding values.
        """
        for key, value in kwargs.items():
            if key not in self.metrics:
                self.metrics[key] = {"total": 0.0, "count": 0, "last": 0.0}
            self.metrics[key]["total"] += value
            self.metrics[key]["count"] += 1
            self.metrics[key]["last"] = value  # Store the most recent value

    def average(self, metric_name):
        """
        Compute the average for a specific metric.
        Args:
            metric_name (str): The name of the metric to compute the average for.
        Returns:
            float: The average value, or None if the metric does not exist.
        """
        if metric_name in self.metrics:
            total = self.metrics[metric_name]["total"]
            count = self.metrics[metric_name]["count"]
            return total / count
        return None

    def get_all_last_values(self):
        """
        Get the last reported values for all metrics.
        Returns:
            dict: A dictionary of the most recent values for each metric.
        """
        return {key: self.metrics[key]["last"] for key in self.metrics}

    def reset(self):
        """Reset all tracked metrics."""
        self.metrics = {}

    def __repr__(self):
        """String representation of the most recent metric values."""
        last_values = self.get_all_last_values()
        return " | ".join(f"{key}: {value:.2f}" for key, value in last_values.items())
    
    def log_json(self, phase: str = "trainval", **kwargs):
        """
        Append a new entry of metrics to the JSON file.
        Args:
            kwargs: Metric names and their corresponding values.
        """
        # Function to convert any numpy types to native Python types
        def convert_numpy_types(obj):
            if isinstance(obj, np.int64):  # Check for numpy int64
                return int(obj)
            elif isinstance(obj, np.float64):  # Check for numpy float64
                return float(obj)
            elif isinstance(obj, np.ndarray):  # If it's a NumPy array, recursively apply conversion
                return obj.tolist()  # Convert NumPy array to list
            elif isinstance(obj, list):  # If it's a list, apply conversion to each item
                return [convert_numpy_types(item) for item in obj]
            return obj  # If it's a regular Python type, return it as is

        # Convert all the kwargs values to native Python types
        kwargs = {k: convert_numpy_types(v) for k, v in kwargs.items()}

        # Ensure the file exists, if not create an empty list
        if not os.path.exists(self.log_json_fpath):
            init_data = {
                phase: []
            }
            with open(self.log_json_fpath, mode="w") as file:
                json.dump(init_data, file)  # Start with an empty list

        # Load existing data from the JSON file
        with open(self.log_json_fpath, mode="r") as file:
            data = json.load(file)
            if phase not in data:
                data[phase] = []
            data_phase = data[phase]

        # Append the new metrics to the list
        data_phase.append(kwargs)
        data[phase] = data_phase

        # Write the updated list back to the file
        with open(self.log_json_fpath, mode="w") as file:
            json.dump(data, file, indent=4)  # Pretty print with indentation