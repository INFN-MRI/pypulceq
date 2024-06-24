"""Porting of TOPPE systemspec.m"""

__all__ = ["SystemSpecs"]

from dataclasses import dataclass, field


@dataclass
class SystemSpecs:
    """
    Scanner hardware limits.

    Different gradient models has the following:

    Scanner  Gradient coil   chronaxie rheobase alpha  gmax  smax
    -------  -------------   --------- -------- -----  ----  ----
    MR750w   XRMW            360d-6    20.0     0.324  33    120
    MR750    XRM             334d-6    23.4     0.333  50    200
    HDx      TRM WHOLE       370d-6    23.7     0.344  23    77
    HDx      TRM ZOOM        354d-6    29.1     0.309  40    150
    UHP      HRMB            359d-6    26.5     0.370  100   200
    Premier  HRMW            642.4d-6  17.9     0.310  70    200
    Magnus   MAGNUS          611d-6    52.2     0.324  300   750

    Values on scanner from /w/config/Scandbdt.cfg or GRSubsystemHWO.xml
    (e.g., /export/home/mx/host/config/current/GRSubsystemHWO.xml)
    (alpha = EffectivedBdTlength<X,Y,Z>/100)

    Attributes
    ----------
    raster : float, optional
        Raster time for gradient and RF waveforms, defaults to 4 microseconds.
    gamma : float, optional
        Gyromagnetic ratio in Hz/T, defaults to 42.576e6 Hz/T.
    B0 : float, optional
        Magnetic field strength in Tesla, defaults to 3.0 Tesla.
    gradient : str, optional
        Gradient coil type, defaults to 'xrm'.
    psd_rf_wait : int, optional
        RF/gradient delay in microseconds, defaults to 148 microseconds.
    psd_grd_wait : int, optional
        ADC/gradient delay in microseconds, defaults to 156 microseconds.
    segmentRingdownTime : int, optional
        Delay at the end of the block group, equals 4us + timssi in microseconds, defaults to 116 microseconds.
    forbiddenEspRange : list, optional
        Forbidden echo spacings (mechanical resonance) in microseconds, defaults to [410, 510] microseconds.
    tminwait : int, optional
        Minimum duration of wait pulse in EPIC code in microseconds, defaults to 12 microseconds.
    maxGrad : float, optional
        Maximum gradient strength in mT/m, defaults to 40 mT/m.
    maxSlew : float, optional
        Maximum slew rate in T/m/s, defaults to 150 T/m/s.
    maxRF : float, optional
        Maximum RF strength in uT, defaults to 15 uT.
    rfDeadTime : int, optional
        RF dead time in microseconds, must be >= 72 microseconds, defaults to 100 microseconds.
    rfRingdownTime : int, optional
        RF ringdown time in microseconds, must be >= 54 microseconds, defaults to 210 microseconds.
    adcDeadTime : int, optional
        ADC dead time in microseconds, must be >= 40 microseconds, defaults to 40 microseconds.
    maxSlice : int, optional
        Maximum dabslice, UI won't allow more than this to be entered, defaults to 2048.
    maxView : int, optional
        Maximum view limit, defaults to 600.
    maxEcho : int, optional
        Maximum echo limit, determined empirically, defaults to 16.
    rfUnit : str, optional
        Unit for RF strength, defaults to "uT".
    gradUnit : str, optional
        Unit for gradient strength, defaults to "mT/m".
    slewUnit : str, optional
        Unit for slew rate, defaults to "T/m/s".

    """

    raster: float = 4  # us. Raster time for gradient and RF waveforms.
    gamma: float = 42.576e6  # Hz / T

    # Scanner-specific settings
    B0: float = 3.0  # field strength (T)
    gradient: str = None  # gradient coil
    psd_rf_wait: int = 200  # rf/gradient delay (us)
    psd_grd_wait: int = 200  # ADC/gradient delay (us).
    segmentRingdownTime: int = (
        116  # Delay at the end of the block group, equals 4us + timssi.
    )
    forbiddenEspRange: list = field(
        default_factory=lambda: [410, 510]
    )  # (us) Forbidden echo spacings (mechanical resonance).
    tminwait: int = 12  # minimum duration of wait pulse in EPIC code (us)

    # Design choices (need not equal scanner limits)
    maxGrad: float = None  # mT/m
    maxSlew: float = None  # T/m/s
    maxRF: float = 15  # uT - Not sure what the hardware limit is here
    rfDeadTime: int = 72  # us. Must be >= 72us
    rfRingdownTime: int = 54  # us. Must be >= 54us
    adcDeadTime: int = 40  # us. Must be >= 40us

    # The following determine the slice/echo/view indexing in the data file
    maxSlice: int = 2048  # max dabslice. UI won't allow more than this to be entered
    maxView: int = 600  # not sure what limit is here
    maxEcho: int = 1  # determined empirically

    # units
    rfUnit: str = "uT"
    gradUnit: str = "mT/m"
    slewUnit: str = "T/m/s"

    def __post_init__(self):
        if self.gradient is not None:
            self.gradient = self.gradient.lower()
            self.validate_gradient()
        if self.gradient is not None and self.maxGrad is None:
            self.maxGrad = _gradspecs(self.gradient)["maxGrad"]
        if self.gradient is not None and self.maxSlew is None:
            self.maxSlew = _gradspecs(self.gradient)["maxSlew"]
        assert (
            self.maxGrad is not None
        ), "Please either specify maxGrad or gradient model"
        assert (
            self.maxSlew is not None
        ), "Please either specify maxSlew or gradient model"

    def validate_gradient(self):
        valid_gradient_coils = [
            "xrmw",
            "xrm",
            "whole",
            "zoom",
            "hrmb",
            "hrmw",
            "magnus",
        ]
        if self.gradient is not None and self.gradient not in valid_gradient_coils:
            raise ValueError(f"Gradient coil ({self.gradient}) unknown")


# %% local subroitines
def _gradspecs(gradient):
    if gradient == "xrmw":
        return {"maxGrad": 33, "maxSlew": 120}
    if gradient == "xrm":
        return {"maxGrad": 50, "maxSlew": 120}
    if gradient == "whole":
        return {"maxGrad": 23, "maxSlew": 77}
    if gradient == "zoom":
        return {"maxGrad": 40, "maxSlew": 150}
    if gradient == "hrmb":
        return {"maxGrad": 100, "maxSlew": 200}
    if gradient == "hrmw":
        return {"maxGrad": 70, "maxSlew": 200}
    if gradient == "hrmw":
        return {"maxGrad": 300, "maxSlew": 750}

    # Scanner  Gradient coil   chronaxie rheobase alpha  gmax  smax


# -------  -------------   --------- -------- -----  ----  ----
# MR750w   XRMW            360d-6    20.0     0.324  33    120
# MR750    XRM             334d-6    23.4     0.333  50    200
# HDx      TRM WHOLE       370d-6    23.7     0.344  23    77
# HDx      TRM ZOOM        354d-6    29.1     0.309  40    150
# UHP      HRMB            359d-6    26.5     0.370  100   200
# Premier  HRMW            642.4d-6  17.9     0.310  70    200
# Magnus   MAGNUS          611d-6    52.2     0.324  300   750
