# this code will be executed in cmd prompt
# specify all parameters in cmd

def create_trainer(args, logger):
    '''
    simply call Trainer (from pytorchlightning)
    see: https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html 
    - we can add : 
         * run learning rate finder, results override hparams.learning_rate
           trainer = Trainer(auto_lr_find=True)
         * gpus=x has been deprecated in v1.7 and will be removed in v2.0. Please use accelerator='gpu' and devices=x instead.
         * trainer = Trainer(max_epochs=1000)
         * trainer = Trainer(max_time={"days": 1, "hours": 5})
         * trainer = Trainer(enable_progress_bar=True)
         
     - val_check_interval : How often within one training epoch to check the validation set. Can specify as float or int.
    '''
    return Trainer(
        logger=logger,
        default_save_path=args.exp_dir,
        checkpoint_callback=True,
        max_nb_epochs=args.num_epochs,
        gpus=args.gpus,
        distributed_backend='ddp',
        check_val_every_n_epoch=1,
        val_check_interval=1.,
        early_stop_callback=False
    )

def main(args):
    """
    - TestTubeLogger is implemented class in pytorchLightening -> returns Logger
    - create_trainer: returns a Trainer (which is a class of pytorchlightning) 
      Question: why not simply say: trainer = Trainer (..)?
       
    """
    if args.mode == 'train':
        load_version = 0 if args.resume else None
        logger = TestTubeLogger(save_dir=args.exp_dir, name=args.exp, version=load_version)
        trainer = create_trainer(args, logger)
        model = UnetMRIModel(args)
        trainer.fit(model)
    else:  # args.mode == 'test'
        assert args.checkpoint is not None
        model = UnetMRIModel.load_from_checkpoint(str(args.checkpoint))
        model.hparams.sample_rate = 1.
        trainer = create_trainer(args, logger=False)
        trainer.test(model)

if __name__ == '__main__':
    parser = Args()
    # first initialize parameters , if not given manually then default values will be taken into consideration
    parser.add_argument('--mode', choices=['train', 'test'], default='train')
    
    # add the predifined hyperparamaetrs of UnetMRIModel to the parser
    # note: if you want to change them as well, we may wanna write them here in parser.add_argument()
    parser = UnetMRIModel.add_model_specific_args(parser)
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # execute the function main + take all args with her
    main(args)
