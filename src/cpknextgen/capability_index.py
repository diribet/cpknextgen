import numpy as np


class ResultTypes:
    """
    Human-readable string with short comment about cp/cpk evaluation procedure.
    """

    class Cp:
        STANDARD = "Standard evaluation"
        LOWER_MISSING = "Lower specification missing"
        LOWER_EXTREME = "Non-Standard Evaluation: Lower extreme as specification limit"
        UPPER_MISSING = "Upper specification missing"
        UPPER_EXTREME = "Non-Standard Evaluation: Upper extreme as specification limit"
        SPECIFICATIONS_MISSING = "Upper and Lower specifications missing"
        EXTREMES = "Non-Standard Evaluation: Extremes as specification limits"

    class Cpk:
        STANDARD = "Standard evaluation"
        LOWER_MISSING = "Evaluation to Upper Specification only, c_pk = c_pk_upper"
        UPPER_MISSING = "Evaluation to Lower Specification only, c_pk = c_pk_lower"
        SPECIFICATIONS_MISSING = "Upper and Lower specifications missing"
        EXTREMES = "Non-Standard Evaluation: Extremes as specification limits"


class ProcessStats:
    """
    Container for results of capability index calculations

    ## Instance attributes:

    `sample` (list) - samples from the distribution of the statistic

    `point` (float) - point estimate of the statistic

    `interval` (tuple) - interval estimate of the statistic
     """
    confidence = 0.95

    def __init__(self, sample=None, point=None):
        self.sample = sample if sample is not None else []
        if point is None or np.isnan(point):
            self.point = None
        else:
            self.point = float(point)
        self.interval = (None, None) if sample is None else self._get_confidence_interval()

    def _get_confidence_interval(self):
        quantiles = ((1 - ProcessStats.confidence) / 2, ProcessStats.confidence + (1 - ProcessStats.confidence) / 2)
        arr = np.nanquantile(self.sample, q=quantiles)
        return arr[0], arr[1]


class CapabilityIndex:
    """
    Class for results of capability indexes and process characteristics.

    ## Instance attributes:

    `cp` (ProcessStats)

    `cp_type` (str) - information about the type of cp evaluation

    `cpk` (ProcessStats)

    `cpk_type` (str) - information about the type of cpk evaluation

    `process_center` (ProcessStats)

    `process_high` (ProcessStats)

    `process_low` (ProcessStats)

    `process_width` (ProcessStats)
    """

    def __init__(self, process_low: ProcessStats = None,
                 process_high: ProcessStats = None,
                 process_center: ProcessStats = None,
                 specification_lower=None, specification_upper=None,
                 lower_boundary=None, upper_boundary=None):

        self.process_low = process_low
        self.process_high = process_high
        self.process_center = process_center
        self.specification_lower = specification_lower
        self.specification_upper = specification_upper
        self.upper_boundary = upper_boundary
        self.lower_boundary = lower_boundary

        self.standard_evaluation_cp = None
        self.standard_evaluation_cpk = None

        self.cp = ProcessStats()
        self.cp_type = ""
        self.cpk = ProcessStats()
        self.cpk_type = ""

        if process_high is not None and process_high.point is not None and \
                process_low is not None and process_low.point is not None:
            self.process_width = ProcessStats(sample=process_high.sample - process_low.sample,
                                              point=process_high.point - process_low.point)
        else:
            return

        self._cpk_upper = None
        self._cpk_lower = None

        if None not in (specification_lower, specification_upper):
            self.cp, self.cp_type, self.cpk, self.cpk_type = self._standard()
            self.standard_evaluation_cp = True
            self.standard_evaluation_cpk = True

        elif specification_lower is None and specification_upper is not None:
            self.cp, self.cp_type, self.cpk, self.cpk_type = self._lower_specification_missing()
            self.standard_evaluation_cp = False
            self.standard_evaluation_cpk = True

        elif specification_lower is not None and specification_upper is None:
            self.cp, self.cp_type, self.cpk, self.cpk_type = self._upper_specification_missing()
            self.standard_evaluation_cp = False
            self.standard_evaluation_cpk = True

        else:
            self.cp, self.cp_type, self.cpk, self.cpk_type = self._specifications_missing()
            self.standard_evaluation_cp = False
            self.standard_evaluation_cpk = False

    @property
    def cpk_upper(self):
        if self._cpk_upper is None:
            if None not in [self.specification_upper, self.process_center, self.process_high]:
                self._cpk_upper = ProcessStats(
                    sample=(self.specification_upper - self.process_center.sample) / (
                            self.process_high.sample - self.process_center.sample),
                    point=(self.specification_upper - self.process_center.point) / (
                            self.process_high.point - self.process_center.point))
            elif None not in [self.upper_boundary, self.process_center, self.process_high]:
                self._cpk_upper = ProcessStats(
                    sample=(self.upper_boundary - self.process_center.sample) / (
                            self.process_high.sample - self.process_center.sample),
                    point=(self.upper_boundary - self.process_center.point) / (
                            self.process_high.point - self.process_center.point))
        return self._cpk_upper

    @property
    def cpk_lower(self):
        if self._cpk_lower is None:
            if None not in [self.specification_lower, self.process_center, self.process_low]:
                self._cpk_lower = ProcessStats(
                    sample=(self.process_center.sample - self.specification_lower) /
                           (self.process_center.sample - self.process_low.sample),
                    point=(self.process_center.point - self.specification_lower) /
                    (self.process_center.point - self.process_low.point)
                )
            elif None not in [self.lower_boundary, self.process_center, self.process_low]:
                self._cpk_lower = ProcessStats(
                    sample=(self.process_center.sample - self.lower_boundary) /
                           (self.process_center.sample - self.process_low.sample),
                    point=(self.process_center.point - self.lower_boundary) /
                    (self.process_center.point - self.process_low.point)
                )
        return self._cpk_lower

    def _get_cp(self, upper_limit=None, lower_limit=None) -> ProcessStats:
        result = ProcessStats(sample=(upper_limit - lower_limit) / self.process_width.sample,
                              point=(upper_limit - lower_limit) / self.process_width.point)
        return result

    def _standard(self):
        cp = self._get_cp(upper_limit=self.specification_upper, lower_limit=self.specification_lower)
        cpk = ProcessStats(
            sample=np.minimum(self.cpk_lower.sample, self.cpk_upper.sample),
            point=np.minimum(self.cpk_lower.point, self.cpk_upper.point)
        )
        return cp, ResultTypes.Cp.STANDARD, cpk, ResultTypes.Cpk.STANDARD

    def _lower_specification_missing(self):
        cpk = self.cpk_upper
        if self.lower_boundary is not None:
            cp = self._get_cp(upper_limit=self.specification_upper, lower_limit=self.lower_boundary)
            return cp, ResultTypes.Cp.LOWER_EXTREME, cpk, ResultTypes.Cpk.LOWER_MISSING
        else:
            cp = ProcessStats()
            return cp, ResultTypes.Cp.LOWER_MISSING, cpk, ResultTypes.Cpk.LOWER_MISSING

    def _upper_specification_missing(self):
        cpk = self.cpk_lower
        cpk_type = ResultTypes.Cpk.UPPER_MISSING
        if self.upper_boundary is not None:
            cp = self._get_cp(upper_limit=self.upper_boundary, lower_limit=self.specification_lower)
            return cp, ResultTypes.Cp.UPPER_EXTREME, cpk, cpk_type
        else:
            cp = ProcessStats()
            return cp, ResultTypes.Cp.UPPER_MISSING, cpk, cpk_type

    def _specifications_missing(self):
        if None not in (self.lower_boundary, self.upper_boundary):
            cp = self._get_cp(lower_limit=self.lower_boundary, upper_limit=self.upper_boundary)
            cpk = ProcessStats(sample=np.minimum(self.cpk_lower.sample, self.cpk_upper.sample),
                               point=np.minimum(self.cpk_lower.point, self.cpk_upper.point))
            cpk_type = ResultTypes.Cpk.EXTREMES
            return cp, ResultTypes.Cp.EXTREMES, cpk, cpk_type
        else:
            cp = ProcessStats()
            cpk = ProcessStats()
            cpk_type = ResultTypes.Cpk.SPECIFICATIONS_MISSING

            return cp, ResultTypes.Cp.SPECIFICATIONS_MISSING, cpk, cpk_type
