from glow.freq_domain_c import Fw_AnalyticPointLens_C
from glow import lenses
import astropy.constants as c
import numpy as np
from dingo.gw.transforms.utils import get_batch_size_of_input_sample
import lal

""" Lensed waveforms using GLOW """

def compute_Ff(MLz, y, domain, GMsun8pi):
    """" Returns Point lens amplification factor """
    w = GMsun8pi * MLz * domain
    Ff = Fw_AnalyticPointLens_C(y, {"parallel": False}).eval_Fw(w)
    Ff[0] = 1.0
        
    return np.conjugate(Ff) #updated convention
    
    
class LensingTransformPL(object):
    """
    Return lensed waveform
    """
    
    def __init__(self, domain):      
        self.domain = domain
        self.GMsun8pi = (8 * c.G * c.M_sun * np.pi / c.c**3).decompose().value
        
        
    def __call__(self, input_sample):
        sample = input_sample.copy()
        
        parameters = sample["parameters"].copy()
        extrinsic_parameters = sample["extrinsic_parameters"].copy()
        
        LogMLz_arr=extrinsic_parameters.pop("LogMLz")
        y_arr=extrinsic_parameters.pop("y")

        MLz_arr = 10**LogMLz_arr
        
        batched, batch_size = get_batch_size_of_input_sample(sample)
        
        if not batched:
            Ffs = compute_Ff(MLz_arr,y_arr,self.domain,self.GMsun8pi)
        else:
            Ffs = np.array([compute_Ff(MLz,y,self.domain,self.GMsun8pi) for (MLz,y) in zip(MLz_arr,y_arr)])

                
        h_plus = sample["waveform"]["h_plus"].copy()
        h_cross = sample["waveform"]["h_cross"].copy()
        sample["waveform"]["h_plus"] = Ffs * h_plus   
        sample["waveform"]["h_cross"] = Ffs * h_cross

        parameters["LogMLz"] = LogMLz_arr
        parameters["y"] = y_arr
        sample["parameters"] = parameters
        sample["extrinsic_parameters"] = extrinsic_parameters
        
        return sample