import os
import torch
from models import Autoformer, Transformer, TimesNet, Nonstationary_Transformer, DLinear, FEDformer, \
    Informer, LightTS, Reformer, ETSformer, Pyraformer, PatchTST, PatchTST2, PatchTST_sep, PatchTST_sep_pretrain,PatchTST_sepcmix_simple_flatten_pretrain, PatchTST_sep_cmix, PatchTST_sepcmix, PatchTST_sepcmix_simple, PatchTST_sepcmix_simple_flatten,PatchTST_sepcmix_simple_flatten_covid, PatchTST_sepcmix_simple_flatten_wo_cmix_covid, PatchTST_sepcmix_simple_flatten_change, PatchTST_cmix_simple_flatten_change_wo_sub, PatchTST_sep_rag, PatchTST_rag, MICN, Crossformer, FiLM, iTransformer, \
    Koopa, TiDE, FreTS, TimeMixer, TSMixer, SegRNN, MambaSimple, TemporalFusionTransformer, SCINet, PAttn, TimeXer


class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            'TimesNet': TimesNet,
            'Autoformer': Autoformer,
            'Transformer': Transformer,
            'Nonstationary_Transformer': Nonstationary_Transformer,
            'DLinear': DLinear,
            'FEDformer': FEDformer,
            'Informer': Informer,
            'LightTS': LightTS,
            'Reformer': Reformer,
            'ETSformer': ETSformer,
            'PatchTST': PatchTST,
            'PatchTST2': PatchTST2,
            'PatchTST_sep': PatchTST_sep,
            'PatchTST_sep_pretrain': PatchTST_sep_pretrain,
            'PatchTST_cmix_simple_flatten_change_wo_sub':PatchTST_cmix_simple_flatten_change_wo_sub,
            'PatchTST_sep_cmix': PatchTST_sep_cmix,
            'PatchTST_sepcmix':PatchTST_sepcmix,
            'PatchTST_sepcmix_simple':PatchTST_sepcmix_simple,
            'PatchTST_sepcmix_simple_flatten':PatchTST_sepcmix_simple_flatten,
            'PatchTST_sepcmix_simple_flatten_pretrain':PatchTST_sepcmix_simple_flatten_pretrain,
            'PatchTST_sepcmix_simple_flatten_covid':PatchTST_sepcmix_simple_flatten_covid,
            'PatchTST_sepcmix_simple_flatten_wo_cmix_covid':PatchTST_sepcmix_simple_flatten_wo_cmix_covid,
            'PatchTST_sepcmix_simple_flatten_change':PatchTST_sepcmix_simple_flatten_change,
            'PatchTST_sep_rag': PatchTST_sep_rag,
            'PatchTST_rag': PatchTST_rag,
            'Pyraformer': Pyraformer,
            'MICN': MICN,
            'Crossformer': Crossformer,
            'FiLM': FiLM,
            'iTransformer': iTransformer,
            'Koopa': Koopa,
            'TiDE': TiDE,
            'FreTS': FreTS,
            'MambaSimple': MambaSimple,
            'TimeMixer': TimeMixer,
            'TSMixer': TSMixer,
            'SegRNN': SegRNN,
            'TemporalFusionTransformer': TemporalFusionTransformer,
            "SCINet": SCINet,
            'PAttn': PAttn,
            'TimeXer': TimeXer
        }
        if args.model == 'Mamba':
            print('Please make sure you have successfully installed mamba_ssm')
            from models import Mamba
            self.model_dict['Mamba'] = Mamba

        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
