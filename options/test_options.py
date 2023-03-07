from .base_options import BaseOptions


class TestOptions(BaseOptions):
    def initialize(self, parser):
        BaseOptions.initialize(self, parser)
        parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
        parser.add_argument('--how_many', type=int, default=float("inf"), help='how many test images to run')
        parser.add_argument('--show_input', action='store_true', help='show input images with the synthesized image')
        parser.add_argument('--transfer_learning', action='store_true',
                            help='if specified, train parameters of feature extractor')

        parser.set_defaults(serial_batches=True)
        parser.set_defaults(no_flip=True)
        parser.set_defaults(phase='test')
        parser.add_argument('--print_freq', type=int, default=1, help='frequency of showing training results on console')
        parser.add_argument('--display_freq', type=int, default=1, help='frequency of showing training results on screen')

        self.isTrain = False
        return parser
