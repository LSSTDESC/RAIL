from numbers import number

zinfo = {'zmin': 0.,
         'zmax': 2.,
         'dz': 0.02}

class LSSTErrorModel():
    
    def __init__(self, limiting_mags=None, err_params=None, undetected_flag=99):
        
        if limiting_mags is not None:
            self.limiting_mags = limiting_mags
        else:
            # defaults are 10 year 5-sigma point source depth
            # from https://www.lsst.org/scientists/keynumbers
            self.limiting_mags = {'u': 26.1,
                                  'g': 27.4,
                                  'r': 27.5,
                                  'i': 26.8,
                                  'z': 26.1,
                                  'y': 27.9}
            
        if err_params is not None:
            self.err_params = err_params
        else:
            # defaults are gamma values in Table 2
            # from https://arxiv.org/pdf/0805.2366.pdf
            self.err_params = {'u': 0.038,
                               'g': 0.039,
                               'r': 0.039,
                               'i': 0.039,
                               'z': 0.039,
                               'y': 0.039}
            
        # check that the keys match
        err_str = 'limiting_mags and err_params have different keys'
        assert self.limiting_mags.keys() == self.err_params.keys(), err_str
        
        # check that all the values are numbers
        all_numbers = all(isinstance(val, Number) for val in self.limiting_mags.values())
        err_str = 'All limiting magnitudes must be numbers'
        assert all_numbers, err_str
        all_numbers = all(isinstance(val, Number) for val in self.err_params.values())
        err_str = 'All error parameters must be numbers'
        assert all_numbers, err_str
        
    def __call__(self, data, seed=None):
        # Gaussian errors using Equation 5
        # from https://arxiv.org/pdf/0805.2366.pdf
        # then flag all magnitudes beyond 5-sig limit
        
        rng = np.random.default_rng(err_seed)

        for band self.limiting_mags.keys():
            
            # calculate err with Eq 5
            m5 = self.limiting_mags[band]
            gamma = self.err_params[band]
            x = 10**(0.4*(data[band] - m5))
            err = np.sqrt((0.04 - gamma)*x + gamma*x**2)
            
            # Add errs to data frame
            data[f'{band}_err'] = err
            data[band] = rng.normal(data[band], data[f'{band}_err'])
            
            # flag mags beyond limiting mag
            data.loc[data.eval(f'{band} > {m5}'), band] = 99
                
