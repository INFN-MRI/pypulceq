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
    
    """
    raster: float = 4  # us. Raster time for gradient and RF waveforms.
    gamma: float = 42.576e6  # Hz / T

    # Scanner-specific settings
    B0: float = 3.0  # field strength (T)
    gradient: str = 'xrm'  # gradient coil
    psd_rf_wait: int = 148  # rf/gradient delay (us)
    psd_grd_wait: int = 156  # ADC/gradient delay (us).
    segmentRingdownTime: int = 116  # Delay at the end of the block group, equals 4us + timssi.
    forbiddenEspRange: list = field(default_factory=lambda: [410, 510])  # (us) Forbidden echo spacings (mechanical resonance).
    tminwait: int = 12  # minimum duration of wait pulse in EPIC code (us)

    # Design choices (need not equal scanner limits)
    maxGrad: float = 40  # mT/m
    maxSlew: float = 150  # T/m/s
    maxRF: float = 15  # uT - Not sure what the hardware limit is here
    rfDeadTime: int = 100  # us. Must be >= 72us
    rfRingdownTime: int = 210  # us. Must be >= 54us
    adcDeadTime: int = 40  # us. Must be >= 40us

    # The following determine the slice/echo/view indexing in the data file
    maxSlice: int = 2048  # max dabslice. UI won't allow more than this to be entered
    maxView: int = 600  # not sure what limit is here
    maxEcho: int = 16  # determined empirically
    
    # units
    rfUnit: str = "uT"
    gradUnit: str = "mT/m"
    slewUnit: str = "T/m/s"
    
    def __post_init__(self):
        self.gradient = self.gradient.lower()

    def validate_gradient(self):
        valid_gradient_coils = ['xrmw', 'xrm', 'whole', 'zoom', 'hrmb', 'hrmw', 'magnus']
        if self.gradient not in valid_gradient_coils:
            raise ValueError(f'Gradient coil ({self.gradient}) unknown')