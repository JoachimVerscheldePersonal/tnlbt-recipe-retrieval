import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
from torch.autograd import Variable
from models import VisualTransformerEncoder, HierarchicalTransformerEncoder
from pytorch_metric_learning.losses import TripletMarginLoss
from pytorch_metric_learning.miners import TripletMarginMiner
from config import parse_config

config = parse_config("config.yaml")

image_model = DistributedDataParallel(VisualTransformerEncoder())
recipe_model = DistributedDataParallel(HierarchicalTransformerEncoder())

triplet_loss = TripletMarginLoss(margin=config.training.triplet_loss_margin)
hard_sample_miner = TripletMarginMiner(margin=config.training.triplet_loss_margin, type_of_triplets="hard")
image2recipe_criterion = nn.MultiLabelMarginLoss().cuda()

weights_class = torch.Tensor(config.im2recipe_model.numClasses).fill_(1)
weights_class[0] = 0
class_criterion = nn.CrossEntropyLoss(weight=weights_class).cuda()

recipe2image_cross_entropy = nn.BCELoss().cuda()

fc_layer = nn.Sequential(
    nn.Linear()
)

noise_dim = config.text_img.noise_dim
fixed_noise = Variable(torch.FloatTensor(config.model.batch_size, noise_dim).normal_(0, 1)).cuda()
real_labels = Variable(torch.FloatTensor(config.model.batch_size).fill_(1)).cuda()
fake_labels = Variable(torch.FloatTensor(config.model.batch_size).fill_(0)).cuda()
