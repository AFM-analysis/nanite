import inspect

import numpy as np
import scipy.ndimage.filters as spfilt

#: Valid keyword arguments for feature types
VALID_FEATURE_TYPES = ["all", "binary", "continuous"]


class IndentationFeatures(object):
    def __init__(self, dataset=None):
        #: current dataset from which features are computed
        self.dataset = dataset

    @property
    def is_fitted(self):
        if self.is_valid:
            return self.dataset.fit_properties["success"]
        else:
            return False

    @property
    def is_valid(self):
        return bool(self.dataset.fit_properties)

    @property
    def has_contact_point(self):
        if self.is_fitted:
            pint = self.dataset.fit_properties["params_fitted"]
            return "contact_point" in pint
        else:
            return False

    @property
    def contact_point(self):
        if self.has_contact_point:
            pint = self.dataset.fit_properties["params_fitted"]
            return pint["contact_point"].value
        else:
            raise ValueError("No contact point in data!")

    @property
    def datafit_apr(self):
        seg = ~self.dataset["segment"].values
        f = self.dataset["fit"][seg].values.copy()
        return f

    @property
    def datares_apr(self):
        return self.datafit_apr - self.datay_apr

    @property
    def datax_apr(self):
        xaxis = self.dataset.fit_properties["x_axis"]
        seg = ~self.dataset["segment"].values
        x = self.dataset[xaxis][seg].values.copy()
        # Make sure everything is ok
        assert x[0] > x[-1], "Approach from large distances towards lower"
        return x

    @property
    def datay_apr(self):
        yaxis = self.dataset.fit_properties["y_axis"]
        seg = ~self.dataset["segment"].values
        y = self.dataset[yaxis][seg].values.copy()
        return y

    @property
    def meta(self):
        return self.dataset.metadata

    @staticmethod
    def compute_features(idnt, which_type="all", names=None,
                         ret_names=False):
        """Compute the features for a data set

        Parameters
        ----------
        idnt: nanite.Indentation
            A dataset to rate
        names: list of str
            The names of the rating methods to use,
            e.g. ["rate_apr_bumps", "rate_apr_mon_incr"].
            If None (default), all available rating methods are
            used.

        Notes
        -----
        `names` may include features that are excluded by `which_type`.
        E.g. if a "bool" feature is in `names` but `which_type` is
        "float", then the "bool" feature will be silently ignored.
        """
        inst = IndentationFeatures(idnt)
        # make sure to keep the order of `names`, i.e. only compute
        # names if they are not given.
        if names is None or which_type != "all":
            names = IndentationFeatures.get_feature_names(
                which_type=which_type, names=names)
        samples = []
        for nameii in names:
            sii = float(getattr(inst, nameii)())
            samples.append(sii)
        samples = np.array(samples, dtype=float).flatten()
        if ret_names:
            return samples, names
        else:
            return samples

    @classmethod
    def get_feature_funcs(cls, which_type="all", names=None):
        """Return functions that compute features from a dataset

        Parameters
        ----------
        names: list of str
            The names of the rating methods to use,
            e.g. ["rate_apr_bumps", "rate_apr_mon_incr"].
            If None (default), all available rating methods are
            returned.
        which_type: str
            Which features to return: ["all", "bool", "float"].

        Returns
        -------
        raters: list of tuples (name, callable)
            Each item in the list consists contains the name
            of the rating method and the corresponding rating
            method.
        """
        fnames = cls.get_feature_names(which_type=which_type, names=names)
        ffuncs = [(ff, getattr(cls, ff)) for ff in fnames]
        return ffuncs

    @classmethod
    def get_feature_names(cls, which_type="all", names=None,
                          ret_indices=False):
        """Get features names

        Parameters
        ----------
        which_type: str or list of str
            Return only features that are of a certain type.
            See `VALID_FEATURE_TYPES` for valid strings.
        names: list of str
            Return only features that are in this list.
        ret_indices: bool
            If True, also return the internal feature indices.

        Returns
        -------
        name_list: list of str
            List of feature names (callables of this class)
        """
        if isinstance(which_type, (list, tuple)):
            # recurse
            fnames = []
            for wt in which_type:
                fnames += cls.get_feature_names(which_type=wt)
        else:
            if which_type not in VALID_FEATURE_TYPES:
                msg = "`which_type` must be one of {}, got '{}'".format(
                    VALID_FEATURE_TYPES, which_type)
                raise ValueError(msg)
            if which_type == "all":
                fstart = "feat_"
            elif which_type == "binary":
                fstart = "feat_bin_"
            elif which_type == "continuous":
                fstart = "feat_con_"
            ffuncs = inspect.getmembers(cls, lambda a: (inspect.isroutine(a)))
            fnames = [ff[0] for ff in ffuncs if ff[0].startswith(fstart)]
        # keep only names requested by the user
        if names:
            # convenience: make sure the requested feature names all exists
            refnames = cls.get_feature_names()
            unknown = [nn for nn in names if nn not in refnames]
            if unknown:
                msg = "Unknown feature names: '{}'".format(",".join(unknown))
                raise ValueError(msg)
            # only use selected features
            fnames = [ff for ff in fnames if ff in names]
        fnames = sorted(fnames)
        if ret_indices:
            # return the internal index
            allnames = cls.get_feature_names()
            indices = []
            for ii, aff in enumerate(allnames):
                if aff in fnames:
                    indices.append(ii)
            return fnames, np.array(indices, dtype=int)
        else:
            return fnames

    # for bool features, 1 means good and 0 means bad
    def feat_bin_apr_spikes_count(self):
        """spikes during IDT

        Sudden spikes in indentation curve
        """
        if self.has_contact_point:
            cp = self.contact_point
            # indentation part
            indidx = self.datax_apr < cp
            diff = self.datares_apr[indidx]
            diff = diff[~np.isnan(diff)]
            if len(diff) > 50:
                # find regions with peaks
                diff_smooth = spfilt.gaussian_filter1d(diff, sigma=11)
                delta1 = diff - diff_smooth
                diff_smooth2 = spfilt.gaussian_filter1d(diff, sigma=1)
                delta2 = diff_smooth2 - diff_smooth
                std = np.std(delta1)
                peakarray = np.diff(np.abs(delta2) > 3*std)
                npeaks = np.sum(peakarray)
                value = npeaks <= 5
            else:
                value = np.nan
        else:
            value = np.nan
        return value

    def feat_bin_cp_position(self):
        """CP outside of data range

        Contact point position outside of range
        """
        if self.has_contact_point:
            cp = self.contact_point
            x = self.datax_apr
            if cp < np.min(x) or cp > np.max(x):
                value = False
            else:
                value = True
            return value
        else:
            return np.nan

    def feat_bin_size(self):
        """dataset too small

        Number of points in indentation curve
        """
        if self.is_valid:
            a_ind = self.datay_apr
            num = a_ind.shape[0]
            if num < 600:
                value = False
            else:
                value = True
        else:
            value = np.nan
        return value

    def feat_con_apr_flatness(self):
        """flatness of APR residuals

        fraction of the positive-gradient residuals in the approach part
        """
        if self.has_contact_point:
            cp = self.contact_point
            # baseline indices of approach curve
            # (approaches from pos values)
            blidx = self.datax_apr > cp
            # get residuals of approach curve
            r_bl = self.datares_apr[blidx]
            r_bl = r_bl[~np.isnan(r_bl)]
            # perform gaussian blur
            sigma = max(5, (np.int(r_bl.shape[0]/120)//2)*2+1)
            y = spfilt.gaussian_filter1d(r_bl, sigma=sigma)
            if len(y) > 2:
                grad = np.gradient(y)
                pos = np.sum(grad > 0)
                neg = np.sum(grad < 0)
                value = pos/(pos + neg)
            else:
                value = np.nan
        else:
            value = np.nan
        return value

    def feat_con_apr_sum(self):
        """residuals of APR

        absolute sum of the residuals in the approach part
        """
        if self.has_contact_point:
            cp = self.contact_point
            # indentation part
            indidx = self.datax_apr > cp
            diff = self.datares_apr[indidx]
            diff = diff[~np.isnan(diff)]
            xin = self.datax_apr
            yin = self.datay_apr

            norm = xin.size * np.max(yin)
            value = np.sum(np.abs(diff)) / norm * 100
            value = np.log(1+value)
        else:
            value = np.nan
        return value

    def feat_con_apr_size(self):
        """relative APR size

        length of the approach part relative to the indentation part
        """
        if self.has_contact_point:
            cp = self.contact_point
            # baseline indices of approach curve
            # (approaches from pos values)
            x = self.datax_apr
            aprsize = np.sum(x > cp)
            tolsize = x.shape[0]
            value = 1 - aprsize/tolsize
            return value
        else:
            return np.nan

    def feat_con_bln_slope(self):
        """slope of BLN

        slope obtained from a linear least-squares fit to the baseline
        """
        if self.has_contact_point:
            cp = self.contact_point
            x = self.datax_apr
            y = self.datares_apr
            # break point is between xmax and cp
            breakp = (np.max(x) + cp) / 2
            # left of break point and no nans
            valid = (x > breakp) * (~np.isnan(x))

            # slope range
            x_sl = x[valid]
            y_sl = y[valid]

            if x_sl.size > 20:
                # fit a line to the data
                A = np.vstack([x_sl, np.ones(len(x_sl))]).T
                m, _c = np.linalg.lstsq(A, y_sl, rcond=None)[0]
                # normalize with maximum force
                value = m / np.max(self.datay_apr)
                value = np.log(1 + np.abs(value)) / 10
            else:
                value = np.nan
        else:
            value = np.nan
        return value

    def feat_con_bln_variation(self):
        """variation in BLN

        comparison of the forces at the beginning and at the end
        of the baseline
        """
        if self.has_contact_point:
            cp = self.contact_point
            r_bl = self.datares_apr[self.datax_apr > cp]
            r_bl = r_bl[~np.isnan(r_bl)]
            # skip 10%
            offset = int(r_bl.shape[0]*.1)
            if offset:
                r_bl = r_bl[:-offset]
            if r_bl.shape[0] > 20:
                avg1 = np.average(r_bl[:10])
                avg2 = np.average(r_bl[-10:])
                # normalize with maximum force and multiply by 1000
                maxforce = np.max(self.datay_apr)
                value = np.abs(avg1-avg2)/maxforce * 1e3
                value = np.log(1 + value) / 5
            else:
                value = np.nan
            return value
        else:
            return np.nan

    def feat_con_cp_curvature(self):
        """curvature at CP

        curvature of the force-distance data at the contact point
        """
        if self.has_contact_point:
            cp = self.contact_point
            x = self.datax_apr
            y = self.datay_apr
            # 10 % of indentation is range
            cpid = np.argmin(np.abs(x - cp))
            maxid = np.argmax(y)
            incl = np.abs(maxid - cpid) // 10
            if incl > 5:
                reg = y[cpid - incl:cpid + incl]
                if len(reg):
                    lin = np.linspace(reg.min(), reg.max(), reg.size)
                    diff = np.sum(reg-lin)
                    value = diff / y.max() * 10
                    value = np.log(1 + np.abs(value)) * np.sign(value) / 4
                else:
                    value = np.nan
            else:
                value = np.nan
        else:
            value = np.nan
        return value

    def feat_con_cp_magnitude(self):
        """residuals at CP

        mean value of the residuals around the contact point
        """
        if self.has_contact_point:
            cp = self.contact_point
            # use 10% of the indentation part as size
            r_range = int(np.sum(self.datax_apr < cp) * .1)
            cpidx = np.nanargmin(np.abs(self.datax_apr-cp))
            rat_range = np.abs(
                self.datares_apr[cpidx - r_range:cpidx + r_range])
            # normalization with setpoint of curve gives us a number
            if len(rat_range):
                value = np.nansum(rat_range) / np.max(self.datay_apr) / 100
            else:
                value = np.nan
            return value
        else:
            return np.nan

    def feat_con_idt_maxima_75perc(self):
        """maxima in IDT residuals

        sum of the indentation residuals' maxima in three intervals
        in-between 25% and 100% relative to the maximum indentation
        """
        if self.has_contact_point:
            yin = self.datay_apr
            fit = self.datafit_apr
            xin = self.datax_apr
            cp = self.contact_point
            id100 = np.argmin(xin)
            id000 = np.argmin(np.abs(xin - cp))
            id025 = int(id000 + .25 * (id100 - id000))
            idmin, idmax = min(id025, id100), max(id025, id100)
            # find zeros
            idcen = idmin + (idmax - idmin) // 2
            smooth = spfilt.gaussian_filter1d(yin-fit, sigma=11)
            idzero1 = idmin + np.argmin(np.abs(smooth[idmin:idcen]))
            idzero2 = idcen + np.argmin(np.abs(smooth[idcen:idmax]))
            # change of sign in 1st, 2nd, and 3rd part of indentation
            ydiffs = []
            if idmin != idzero1:
                y1 = np.max(np.abs((yin - fit)[idmin:idzero1]))
                ydiffs.append(y1)
            if idzero1 != idzero2:
                y2 = np.max(np.abs((yin - fit)[idzero1:idzero2]))
                ydiffs.append(y2)
            if idzero2 != idmax:
                y3 = np.max(np.abs((yin - fit)[idzero2:idmax]))
                ydiffs.append(y3)
            if ydiffs:
                # combine the changes (ydiff2 has different sign)
                ydiff = np.sum(ydiffs)
                ydiff /= yin.max()
                value = np.log(1 + ydiff) * 2
            else:
                value = np.nan
        else:
            value = np.nan
        return value

    def feat_con_idt_monotony(self):
        """monotony of IDT

        change of the gradient in the indentation part
        """
        if self.has_contact_point:
            cp = self.contact_point
            # indentation part
            indidx = self.datax_apr < cp
            # get approach curve
            a_ind = self.datay_apr[indidx]
            # perform gaussian blur
            y = spfilt.gaussian_filter1d(a_ind, sigma=2)
            if len(y) > 2:
                grad = np.gradient(y)
                gz = np.abs(np.sum(grad[grad > 0]))
                lz = np.abs(np.sum(grad[grad < 0]))
                value = np.sum(indidx) * lz / gz
                value = np.log(1 + value) / 10
            else:
                value = np.nan
        else:
            value = np.nan
        return value

    def feat_con_idt_sum(self):
        """overall IDT residuals

        sum of the residuals in the indentation part
        """
        if self.has_contact_point:
            cp = self.contact_point
            x = self.datax_apr
            y = self.datay_apr
            f = self.datafit_apr
            idind = x < cp
            diff = (y - f)[idind]
            area = np.nansum(np.abs(diff))/diff.shape[0]
            norm = np.abs(y.max() - y.min())/2
            value = area/norm
            value = np.log(1 + value) * 5
            return value
        else:
            return np.nan

    def feat_con_idt_sum_75perc(self):
        """residuals in 75% IDT

        sum of the residuals in the indentation part in-between
        25% and 100% relative to the maximum indentation
        """
        if self.has_contact_point:
            yin = self.datay_apr
            fit = self.datafit_apr
            xin = self.datax_apr
            cp = self.contact_point
            id100 = np.argmin(xin)
            id000 = np.argmin(np.abs(xin - cp))
            id025 = int(id000 + .25 * (id100 - id000))
            idmin, idmax = min(id025, id100), max(id025, id100)
            ydiff = np.sum(np.abs(yin - fit)[idmin:idmax])
            xnorm = np.abs(xin[idmin] - xin[idmax])
            # area under the curve (normalization with absolute x interval)
            area = ydiff * xnorm
            # normalize by max force and multiply by 1e6 to get values around 1
            value = area / np.max(yin) * 1e6
            value = np.log(1+value) / 8
        else:
            value = np.nan
        return value

    def feat_con_idt_spike_area(self):
        """area of IDT spikes

        area of spikes appearing in the indentation part
        """
        if self.has_contact_point:
            cp = self.contact_point
            # indentation part
            indidx = self.datax_apr < cp
            diff = self.datares_apr[indidx]
            diff = diff[~np.isnan(diff)]
            if len(diff) > 20:
                # find regions with peaks
                diff_smooth = spfilt.gaussian_filter1d(diff, sigma=11)
                delta1 = diff - diff_smooth
                diff_smooth2 = spfilt.gaussian_filter1d(diff, sigma=1)
                delta2 = np.abs(diff_smooth2 - diff_smooth)
                std = np.std(delta1)
                peaks = np.sum(delta2[delta1 > 3*std])
                # add SD to avoid too many zero-valued samples
                value = (std + peaks) / np.max(self.datay_apr)
                value = np.log(1 + value) * 20
            else:
                value = np.nan
        else:
            value = np.nan
        return value
