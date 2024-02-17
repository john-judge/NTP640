import ipywidgets as widgets
import numpy as np
import math


class ReceptorLevelComputations:

    def __init__(self) -> None:
        self.k_on_slider = None
        self.k_off_slider = None
        self.beta_slider = None
        self.alpha_slider = None
        self.d_slider = None
        self.r_slider = None

        self.dt_slider = None
        self.tMax_slider = None
        self.AgConc_slider = None
        self.StimType_slider = None
        self.PulseDur_slider = None
        self.PulseFreq_slider = None
        self.TrainDur_slider = None

    def validate_int_log10(self, value, default_value):
        if value < 0:
            return default_value
        return int(math.log10(value))
    
    def create_simulation_sliders(self, default_dt=0.01, default_tMax=100, default_AgConc=10**-6, default_StimType=0, default_PulseDur=1, default_PulseFreq=1, default_TrainDur=1):
        layout = widgets.Layout(width='50%')
        style = {'description_width': 'initial'}

        default_dt = self.validate_int_log10(default_dt, 2)
        default_tMax = self.validate_int_log10(default_tMax, 2)
        default_AgConc = self.validate_int_log10(default_AgConc, 6)
        default_PulseDur = self.validate_int_log10(default_PulseDur, 1)
        default_PulseFreq = self.validate_int_log10(default_PulseFreq, 1)
        default_TrainDur = self.validate_int_log10(default_TrainDur, 1)

        self.dt_slider = widgets.IntSlider(value=default_dt, min=-4, max=0, step=1, description='dt = 10^', layout=layout, style=style)
        self.tMax_slider = widgets.IntSlider(value=default_tMax, min=-1, max=5, step=1, description='tMax = 10^', layout=layout, style=style)
        self.AgConc_slider = widgets.IntSlider(value=default_AgConc, min=-7, max=-1, step=1, description='AgConc = 10^', layout=layout, style=style)
        self.StimType_slider = widgets.Dropdown(value=default_StimType, options=['PulseTrain', 'Noise'], description='StimType')
        self.PulseDur_slider = widgets.IntSlider(value=default_PulseDur, min=-2, max=5, step=1, description='PulseDur = 10^', layout=layout, style=style)
        self.PulseFreq_slider = widgets.IntSlider(value=default_PulseFreq, min=-2, max=5, step=1, description='PulseFreq = 10^', layout=layout, style=style)
        self.TrainDur_slider = widgets.IntSlider(value=default_TrainDur, min=-2, max=5, step=1, description='TrainDur = 10^', layout=layout, style=style)

        return [self.dt_slider, self.tMax_slider, self.AgConc_slider, self.StimType_slider, self.PulseDur_slider, self.PulseFreq_slider, self.TrainDur_slider]
        
    def create_rc_sliders(self, default_k_on=10**6, default_k_off=10**3, default_beta=10**3, default_alpha=10**3, default_d=10**3, default_r=10**3):
        
        default_k_on = self.validate_int_log10(default_k_on, 6)
        default_k_off = self.validate_int_log10(default_k_off, 3)
        default_beta = self.validate_int_log10(default_beta, 3)
        default_alpha = self.validate_int_log10(default_alpha, 3)
        default_d = self.validate_int_log10(default_d, 3)
        default_r = self.validate_int_log10(default_r, 3)
        
        self.k_on_slider = widgets.IntSlider(value=default_k_on, min=0, max=7, step=1, description='k_on = 10^')
        self.k_off_slider = widgets.IntSlider(value=default_k_off, min=0, max=4, step=1, description='k_off = 10^')
        self.beta_slider = widgets.IntSlider(value=default_beta, min=0, max=4, step=1, description='beta = 10^')
        self.alpha_slider = widgets.IntSlider(value=default_alpha, min=0, max=4, step=1, description='alpha = 10^')
        self.d_slider = widgets.IntSlider(value=default_d, min=0, max=4, step=1, description='d = 10^')
        self.r_slider = widgets.IntSlider(value=default_r, min=0, max=4, step=1, description='r = 10^')
        return [self.k_on_slider, self.k_off_slider, self.beta_slider, self.alpha_slider, self.d_slider, self.r_slider]


    def get_rate_constants(self):
        return 10**self.k_on_slider.value, 10**self.k_off_slider.value, 10**self.beta_slider.value, 10**self.alpha_slider.value, 10**self.d_slider.value, 10**self.r_slider.value
    
    def get_simulation_params(self):
        return 10**self.dt_slider.value, 10**self.tMax_slider.value, 10**self.AgConc_slider.value, self.StimType_slider.value, 10**self.PulseDur_slider.value, 10**self.PulseFreq_slider.value, 10**self.TrainDur_slider.value

    def print_simulation_params(self):
        print(f'dt: {self.dt_slider.value}')
        print(f'tMax: {self.tMax_slider.value}')
        print(f'AgConc: {self.AgConc_slider.value}')
        print(f'StimType: {self.StimType_slider.value}')
        print(f'PulseDur: {self.PulseDur_slider.value}')
        print(f'PulseFreq: {self.PulseFreq_slider.value}')
        print(f'TrainDur: {self.TrainDur_slider.value}')

    def print_rate_constants(self):
        print(f'k_on: {self.k_on_slider.value}')
        print(f'k_off: {self.k_off_slider.value}')
        print(f'beta: {self.beta_slider.value}')
        print(f'alpha: {self.alpha_slider.value}')
        print(f'd: {self.d_slider.value}')
        print(f'r: {self.r_slider.value}')

    def update_plot(self, line, Ag, fig):
        line.set_ydata(Ag)
        fig.canvas.draw()

    def set_default_position(self, controls, **kwargs):
        """
        Set initial values of sliders assuming that you
        are creating ipywidgets sliders, not Matpotlib sliders.

        Parameters
        ----------
        controls : controls object
        setters : dict
        ** kwargs:
            pairs of param names and a tuple of (x, y, z) where
               x: the lower bound of slider
               y: the upper bound of slider
               z: value to set the slider to.
            Interpolation to an index in [0,49] is needed 
            since the values in the IntSlider object must be 
            integer indices, not necessarily the
            value that you expect the slider to display. 
        """
        for name, val in kwargs.items():
            start, end, default = val
            
            # there are 50 index values for each slider. Interpolate the value of default to the index
            # that is closest to the default value
            index = int((default - start) / (end - start) * 50)

            controls.controls[name].children[0].value = index

    def convert_frac_to_idx(self, time, StimStartFrac, StimEndFrac):
        # Find the indices of stimulus start and end    
        StimStartIndx = int(np.ceil(StimStartFrac * len(time)))
        StimEndIndx = int(np.ceil(StimEndFrac * len(time)))
        return StimStartIndx, StimEndIndx
    
    def create_stim_mask(self, time, StimStartFrac, StimEndFrac, TrainDur, dt):
        StimStartIndx, StimEndIndx = self.convert_frac_to_idx(time, StimStartFrac, StimEndFrac)
        Mask = np.zeros_like(time)
        Mask[StimStartIndx:StimStartIndx + int(np.ceil(TrainDur / dt))] = 1
        return Mask

