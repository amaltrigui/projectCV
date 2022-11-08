# this class is a pl.LightningModule

from utils import DataTransform, sliceData
from torch.utils.data import DistributedSampler, DataLoader

class fastMRIDataModel(pl.LightningModule):
  """
  This is a subclass of the LightningModule class from pytorch_lightning, with
    some additional functionality specific to fastMRI:
        - fastMRI data loaders
        - Evaluating reconstructions
        - Visualization
        - Saving test reconstructions

    To implement a new reconstruction model, inherit from this class and implement the
    following methods:
        - train_data_transform, val_data_transform, test_data_transform:
            Create and return data transformer objects for each data split
        - training_step, validation_step, test_step:
            Define what happens in one step of training, validation and testing respectively
        - configure_optimizers:
            Create and return the optimizers
    Other methods from LightningModule can be overridden as needed.
  """
  def __init__(self, hparams):
    super().__init__()
    self.hparams=hparams
    
  def create_data_loader (self, data_transform, data_partition, sample_rate=None):
    """
    """
    sample_rate = sample_rate or self.hparams.sample_rate
    if data_partition = 'train':
      datapath = self.hparams.traindata_path
    elif data_partition = 'val' :
      datapath = self.hparams.valdata_path
    else:
      datapath = self.hparams.testdata_path
    
    dataset = sliceData(
      root = datapath,
      transform = data_transform,
      challenge = self.hparams.challenge,
      sample_rate = sample_rate
    )
    
    sampler = DistributedSampler(dataset)
    return Dataloader (
      dataset = dataset,
      batch_size = self.hparams.batch.size,
      num_workers = 0,
      pin_memory = True,
      sampler = sampler
    )
  
