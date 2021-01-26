import numpy as np
from scipy import stats
import plots


class Metrics:
    """
       ***   Metrics class   ***
    Receives a Sample object as input.
    Computes PIT and QQ vectors on the initialization.
    It's the basis for the other metrics, such as KS, AD, and CvM.

    Parameters
    ----------
    sample: `Sample`
        sample object defined in ./sample.py
    n_quant: `int`, (optional)
        number of quantiles for the QQ plot
    pit_min: `float`
        lower limit to define PIT outliers
        default is 0.0001
    pit_max:
        upper limit to define PIT outliers
        default is 0.9999
    """

    def __init__(self, sample, n_quant=100, pit_min=0.0001, pit_max=0.9999):
        self._sample = sample
        self._n_quant = n_quant
        self._pit = np.array([self._sample._pdfs[i].cdf(self._sample._ztrue[i])[0][0]
                              for i in range(len(self._sample))])
        Qtheory = np.linspace(0., 1., self.n_quant)
        Qdata = np.quantile(self._pit, Qtheory)
        self._qq_vectors = (Qtheory, Qdata)
        pit_n_outliers = len(self._pit[(self._pit < pit_min) | (self._pit > pit_max)])
        self._pit_out_rate = float(pit_n_outliers) / float(len(self._pit))

        # placeholders for metrics to be calculated
        self._ks_stat = None
        self._ks_pvalue = None
        self._cvm_stat = None
        self._cvm_pvalue = None
        self._ad_stat = None
        self._ad_critical_values = None
        self._ad_significance_levels = None






    @property
    def sample(self):
        return self._sample

    @property
    def n_quant(self):
        return self._n_quant

    @property
    def pit(self):
        return self._pit

    @property
    def qq_vectors(self):
        return self._qq_vectors

    @property
    def pit_out_rate(self):
        return self._pit_out_rate

    def plot_pit_qq(self, bins=None, label=None, title=None, show_pit=True,
                    show_qq=True, show_pit_out_rate=True, savefig=False):
        return plots.plot_pit_qq(self, bins=bins, label=label, title=title,
                                 show_pit=show_pit, show_qq=show_qq,
                                 show_pit_out_rate=show_pit_out_rate,
                                 savefig=savefig)

    @property
    def ks_stat(self):
        return self._ks_stat

    @ks_stat.setter
    def ks_stat(self, value):
        self._ks_stat = value

    @property
    def ks_pvalue(self):
        return self._ks_pvalue

    @ks_pvalue.setter
    def ks_pvalue(self, value):
        self._ks_pvalue = value


    @property
    def cvm_stat(self):
        return self._cvm_stat

    @cvm_stat.setter
    def cvm_stat(self, value):
        self._cvm_stat = value

    @property
    def cvm_pvalue(self):
        return self._cvm_pvalue

    @ks_pvalue.setter
    def cvm_pvalue(self, value):
        self._cvm_pvalue = value



    @property
    def ad_stat(self):
        return self._ad_stat

    @ad_stat.setter
    def ad_stat(self, value):
        self._ad_stat = value

    @property
    def ad_critical_values(self):
        return self._ad_critical_values

    @ad_critical_values.setter
    def ad_critical_values(self, value):
        self._ad_critical_values = value

    @property
    def ad_significance_levels(self):
        return self._ad_significance_levels

    @ad_significance_levels.setter
    def ad_significance_levels(self, value):
        self._ad_significance_levels = value







    @property
    def cde_loss(self, zgrid=None):
        """Computes the estimated conditional density loss described in
        Izbicki & Lee 2017 (arXiv:1704.08095).

        Parameters:
        grid: np array of values at which to evaluate the pdf.
        Returns:
        an estimate of the cde loss.
        """
        if zgrid is None:
            zgrid = self._sample._zgrid

        # grid, pdfs = self.ensemble_obj.evaluate(zgrid, norm=True)
        pdfs = self._sample._pdfs.pdf([zgrid])  # , norm=True)

        n_obs, n_grid = pdfs.shape

        # Calculate first term E[\int f*(z | X)^2 dz]
        term1 = np.mean(np.trapz(pdfs ** 2, zgrid))

        # Calculate second term E[f*(Z | X)]
        nns = [np.argmin(np.abs(zgrid - true_z)) for true_z in self._sample._ztrue]
        term2 = np.mean(pdfs[range(n_obs), nns])

        self._cde_loss = term1 - 2 * term2
        return self._cde_loss






    def all(self):
        metrics_table = str( #f"### {self._sample._name}\n" +
                            "|Metric|Value|\n" +
                            "|---|---|\n" +
                            f"PIT out rate | {self._pit_out_rate:8.4f}\n" +
                            f"CDE loss     | {self._cde_loss:8.4f}\n" +
                            f"KS           | {self.KS()[0]:8.4f}\n" +
                            f"CvM          | {self.CvM()[0]:8.4f}\n" +
                            f"AD           | {self.AD()[0]:8.4f}")

        return metrics_table



class KS:
    """
    Compute the Kolmogorov-Smirnov statistic and p-value for the PIT
    values by comparing with a uniform distribution between 0 and 1.
    Parameters
    ----------
    pit: `numpy.ndarray`
    array with PIT values for all galaxies in the sample
    """
    def __init__(self, pit):
        self._pit = pit
        self._stat, self._pvalue = stats.kstest(self._pit, "uniform")

    @property
    def stat(self):
        return  self._stat
    @property
    def pvalue(self):
        return self._pvalue



class CvM:
    """
    Compute the Cramer-von Mises statistic and p-value for the PIT values
    by comparing with a uniform distribution between 0 and 1.
    Parameters
    ----------
    pit: `numpy.ndarray`
        array with PIT values for all galaxies in the sample
    """
    def __init__(self, pit):
        self._pit = pit
        cvm_result = stats.cramervonmises(self._pit, "uniform")
        self._stat, self._pvalue = cvm_result.statistic, cvm_result.pvalue

    @property
    def stat(self):
        return self._stat
    @property
    def pvalue(self):
        return self._pvalue


class AD:
    """
    Compute the Anderson-Darling statistic and p-value for the PIT
    values by comparing with a uniform distribution between 0 and 1.
    Since the statistic diverges at 0 and 1, PIT values too close to
    0 or 1 are discarded.
    Parameters
    ----------
    pit: `numpy.ndarray`
        array with PIT values for all galaxies in the sample
    ad_pit_min, ad_pit_max: floats
        PIT values outside this range are discarded
    """
    def __init__(self, pit, ad_pit_min=0.001, ad_pit_max=0.999):
        mask = (pit > ad_pit_min) & (pit < ad_pit_max)
        self._stat, self._critical_values, self._significance_levels = stats.anderson(pit[mask])

    @property
    def stat(self):
        return self._stat
    @property
    def critical_values(self):
        return self._critical_values
    @property
    def significance_levels(self):
        return self._significance_levels








class CRPS():
    ''' = continuous rank probability score (Gneiting et al., 2006)'''
    def __init__(self):
        raise NotImplementedError

