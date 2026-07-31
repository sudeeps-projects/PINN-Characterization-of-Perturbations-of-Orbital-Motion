"""Reconstruct the exact ISS satellite record from the OMM elements captured in
Notebook 2's output (epoch 2026-06-18 19:14:58 UTC) and validate against the
known epoch state vector NB2 printed, so we reproduce the paper's ISS orbit
rather than a freshly-downloaded (different) TLE."""
import numpy as np
from sgp4.api import Satrec, WGS72, jday

# OMM elements from Notebook 2 executed output
INC=51.6331; RAAN=292.1003; ECC=0.00046669; ARGP=202.292; MA=157.7867
MM=15.49302471          # rev/day
BSTAR=0.00015438664
NDOT=0.000082           # rev/day^2 (not used by SGP4 propagation)
# epoch 2026-06-18 19:14:58.749504 UTC
jd, fr = jday(2026,6,18,19,14,58.749504)
jdsatepoch = jd+fr
epoch_days_1949 = jdsatepoch - 2433281.5   # days since 1949-12-31 00:00 UT

sat = Satrec()
sat.sgp4init(
    WGS72, 'i', 25544, epoch_days_1949,
    BSTAR,
    NDOT*(2*np.pi)/(1440.0*1440.0),      # rad/min^2 (unused, for completeness)
    0.0,
    ECC,
    np.deg2rad(ARGP),
    np.deg2rad(INC),
    np.deg2rad(MA),
    MM*(2*np.pi)/1440.0,                  # no_kozai rad/min
    np.deg2rad(RAAN),
)
e,r,v = sat.sgp4(jd,fr)
print("epoch state r=",np.round(r,3)," v=",np.round(v,4))
print("expected NB2 r= [2558.58 -6300.92 0.006] v= [4.3998 1.7923 6.0055]")
# export a canonical TLE for reuse
from sgp4 import exporter
l1,l2 = exporter.export_tle(sat)
print(l1); print(l2)
