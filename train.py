# this code will be executed in cmd prompt
# specify all parameters in cmd

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
