import os
import torch
from models.Bptt import BpttLSTM, BpttGRU, BpttMinLSTM, BpttMinGRU, BpttSRU, BpttLSTM_Delay, TBptt_Accum, TBptt_LastOnly, FullBpttLSTM_Delay
from models.Sptt import (SpttGRU_Window, SpttLSTM_Window, 
                         SpttMinGRU_Window, SpttMinLSTM_Window, 
                         SpttSRU_Window, SpttLSTM_Delay, SpttLSTM_Delay_Stablity,
                         Sptt_Window_Accum, Sptt_Window_LastOnly)

class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            'BpttLSTM': BpttLSTM,
            'BpttGRU': BpttGRU,
            'SpttLSTM': SpttLSTM_Window,
            'SpttGRU': SpttGRU_Window,
            'SpttMinLSTM': SpttMinLSTM_Window,
            'SpttMinGRU': SpttMinGRU_Window,
            'SpttSRU': SpttSRU_Window,
            'BpttMinLSTM': BpttMinLSTM,
            'BpttMinGRU': BpttMinGRU,
            'BpttSRU': BpttSRU,
            'SpttLSTM_Delay': SpttLSTM_Delay,
            'TBpttLSTM': BpttLSTM_Delay,
            'SpttStability': SpttLSTM_Delay_Stablity,
            'TBptt_Accum': TBptt_Accum,
            'TBptt_LastOnly': TBptt_LastOnly,
            'Sptt_Window_Accum': Sptt_Window_Accum,
            'Sptt_Window_LastOnly': Sptt_Window_LastOnly,
            'FullBPTT': FullBpttLSTM_Delay,
        }

        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        raise NotImplementedError
    def _acquire_device(self):
        if self.args.use_gpu and self.args.gpu_type == 'cuda':
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        elif self.args.use_gpu and self.args.gpu_type == 'mps':
            device = torch.device('mps')
            print('Use GPU: mps')
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
