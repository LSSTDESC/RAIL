import numpy as np
import pandas as pd

import rail.creation.utils as rcu

class Creator():
    """
    An object that supplies mock data for redshift estimation experiments.
    The mock data is drawn from a probability distribution defined by the generator, with an optional selection function applied.
    """

    def __init__(self, generator, err_fn=rcu.LSSTErrorModel(), selection_fn=None, params=None):
        """
        Parameters
        ----------
        generator: rail.Generator object
            object defining a redshift probability distribution.
            Must have sample, log_prob and pz_estimate methods (see generator.py)
        err_fn: function, optional
            a function that applies an error model to the sample
            Must take a pandas dataframe and a seed number, and return
            a pandas dataframe with the error model applied.
            By default, uses the LSST Error Model of Eq 5 
            from https://arxiv.org/pdf/0805.2366.pdf
        selection_fn: function, optional
            a selection function to apply to the generated sample
            Must take a pandas dataframe and a seed number, and return
            a pandas dataframe with the selection function applied.
        params: dictionary, optional
            additional information desired to be stored with the instance as a dictionary.
        """
        self.generator = generator
        self.err_fn = err_fn
        self.selection_fn = selection_fn
        self.params = params

    def sample(self, n_samples, seed=None, include_err=False, err_seed=None, 
               include_pdf=False, zinfo=rcu.zinfo):
        """
        Draws n_samples from the generator

        Parameters
        ----------
        n_samples: int
            number of samples to draw
        include_err: boolean, optional
            if True, the error function is applied and errors are convolved in the posterior pdfs
        include_pdf: boolean, optional
            if True, then posteriors are returned for each galaxy.
            The posteriors are saved in the column pz_pdf, and the redshift grid saved as df.attrs['pz_grid'].
            The grid is defined by np.arange(zmin, zmax+dz, dz)
        seed: int, optional
            sets the random seed for drawing samples
        err_seed: int, optional
            sets the seed for the Gaussian errors
            If err_seed is not set, but seed is set, then err_seed = seed
        zinfo: dictionary, optional
            must contain `zmin`, `zmax`, `dz` floats

        Returns
        -------
        sample: pandas DataFrame
            samples from model, containing photometry, true redshift, and posterior PDF if requested

        Notes
        -----
        We should rename `sample` output to not be the same as the name of the method `sample()`.
        Output posterior format is currently hardcoded to grid evaluations but could be integrated with qp.
        We will probably change the output format to dovetail with the evaluation module when ready.
        """

        rng = np.random.default_rng(seed)
        if err_seed is None and seed is not None:
            err_seed = seed

        # get samples
        samples = self.generator.sample(n_samples, seed=seed)
        
        if include_err:
            # apply the error model
            samples = self.err_fn(samples, err_seed)

        if self.selection_fn is not None:
            # apply selection function
            samples = self.selection_fn(samples, seed=seed)
            # calculate fraction that survives the cut
            selected_frac = len(samples)/n_samples
            # draw more samples and cut until we have enough samples
            while len(samples) < n_samples:
                # estimate how many extras to draw
                n_supplement = int(1.1/selected_frac*(n_samples - len(samples)))
                # draw new samples
                new_samples = self.generator.sample(n_supplement,
                                                    seed=rng.integers(1e18))
                if include_err:
                    # apply error model to the new samples
                    new_samples = self.err_fn(new_samples, err_seed)
                # apply selection function to the new samples
                new_samples = self.selection_fn(new_samples,
                                                seed=rng.integers(1e18))
                # add these to the larger set
                samples = pd.concat((samples, new_samples), ignore_index=True)
            # cut out the extras
            samples = samples.iloc[:n_samples]

        # calculate posteriors
        if include_pdf:
            posteriors = self.generator.pz_estimate(
                samples, zmin=zinfo['zmin'], zmax=zinfo['zmax'], dz=zinfo['dz']
            )
            samples.attrs['pz_grid'] = np.arange(
                zinfo['zmin'], zinfo['zmax'] + zinfo['dz'], zinfo['dz']
            )
            samples['pz_pdf'] = list(posteriors)

        return samples
