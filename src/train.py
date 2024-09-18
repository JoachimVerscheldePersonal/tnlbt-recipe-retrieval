import torch
from torch.nn.parallel import DistributedDataParallel
from models import VisualTransformerEncoder, HierarchicalTransformerEncoder
from pytorch_metric_learning.losses import TripletMarginLoss
from pytorch_metric_learning.miners import TripletMarginMiner
from config import parse_config

config = parse_config("config.yaml")

image_model = DistributedDataParallel(VisualTransformerEncoder())
recipe_model = DistributedDataParallel(HierarchicalTransformerEncoder())

triplet_loss = TripletMarginLoss(margin=config.training.triplet_loss_margin)
hard_sample_miner = TripletMarginMiner(margin=config.training.triplet_loss_margin, type_of_triplets="hard")