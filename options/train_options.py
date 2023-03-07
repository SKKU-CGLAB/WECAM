from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    def initialize(self, parser):
        BaseOptions.initialize(self, parser)
        parser.add_argument('--save_latest_freq', type=int, default=50, help='frequency of saving the latest results')
        parser.add_argument('--save_epoch_freq', type=int, default=1, help='frequency of saving checkpoints at the end of epochs')
        parser.add_argument('--print_freq', type=int, default=10, help='frequency of showing training results on console')
        parser.add_argument('--display_freq', type=int, default=50, help='frequency of showing training results on screen')

        # for training
        parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        parser.add_argument('--n_epochs', type=int, default=50, help='# of epoch at starting learning rate. This is NOT the total #epochs. Total #epochs is n_epochs + n_epochs_decay')
        parser.add_argument('--n_epochs_decay', type=int, default=0, help='# of epochs to linearly decay learning rate to zero')
        parser.add_argument('--first_epoch', type=int, default=1, help='the starting epoch count, we save the model by <first_epoch>, <first_epoch>+<save_latest_freq>, ...')
        parser.add_argument('--optimizer', type=str, default='adam')
        parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        parser.add_argument('--beta2', type=float, default=0.999, help='momentum term of adam')
        parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
        parser.add_argument('--lr_policy', type=str, default='linear', help='learning rate policy. [linear | step | plateau | cosine]')
        parser.add_argument('--weight_decay', type=float, default=0, help='weight decay rate')
        # for feature extractor
        parser.add_argument('--transfer_learning', action='store_true', help='if specified, train parameters of feature extractor')

        self.isTrain = True
        return parser
