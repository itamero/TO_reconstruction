import numpy as np
import logging
import mrcfile

from utils import show
from aspire.downloader.data_fetcher import fetch_data
from aspire.source import Simulation
from aspire.volume import Volume

from ab_initio_TO import cryo_abinitio_TO
from aspire.operators import FunctionFilter, RadialCTFFilter
from aspire.noise import AnisotropicNoiseEstimator, CustomNoiseAdder

logger = logging.getLogger(__name__)

show_projections = True

# %%
# Parameters
# ---------------
img_size = 89   # Downsample the volume to a desired resolution
num_imgs = 50    # Number of images in our source.
noise_variance = 5e-7   # Control noise added to simulated projections

path = fetch_data("emdb_4905.map")

with mrcfile.open(path) as mrc:
    data = mrc.data

og_vol = Volume.load(path, symmetry_group="O")
logger.info("Original volume map data" f" shape: {og_vol.shape} dtype:{og_vol.dtype}")


logger.info("Initialize CTF filters.")
# Create some CTF effects
pixel_size = og_vol.pixel_size  # Pixel size (in angstroms)
voltage = 200  # Voltage (in KV)
defocus_min = 1.5e4  # Minimum defocus value (in angstroms)
defocus_max = 2.5e4  # Maximum defocus value (in angstroms)
defocus_ct = 7  # Number of defocus groups.
Cs = 2.0  # Spherical aberration
alpha = 0.1  # Amplitude contrast

# Create filters
ctf_filters = [
    RadialCTFFilter(pixel_size, voltage, defocus=d, Cs=2.0, alpha=0.1)
    for d in np.linspace(defocus_min, defocus_max, defocus_ct)
]

def noise_function(x, y):
    alpha = 1
    beta = 1
    # White
    f1 = noise_variance
    # Violet-ish
    f2 = noise_variance * (x * x + y * y) / (img_size * img_size)
    return (alpha * f1 + beta * f2) / 2.0


custom_noise = CustomNoiseAdder(noise_filter=FunctionFilter(noise_function))

src = Simulation(
    n=num_imgs,
    vols=og_vol,
    offsets=0,
    amplitudes=1,
    noise_adder=custom_noise,
    unique_filters=ctf_filters,
)
if show_projections:
    show(src.images[:10], Title = f'Simulated projection images (noise variance = {noise_variance})') # Before downsample
true_rotations = src.rotations


# Use phase_flip to attempt correcting for CTF.
logger.info("Perform phase flip to input images.")
src = src.phase_flip()

# Estimate the noise and `Whiten` based on the estimated noise
aiso_noise_estimator = AnisotropicNoiseEstimator(src)
src = src.whiten(aiso_noise_estimator)

if show_projections:
    show(src.images[:10], Title = 'Whitened denoised images')

src = src.downsample(img_size)

if show_projections:
    show(src.images[:10], Title = 'After down-sampling')

numpy_projections = src.images[:].asnumpy()

file_name = f'O_projections_{num_imgs}_size_{img_size}_noise_{noise_variance}.mrc'
with mrcfile.new(file_name, overwrite=True) as mrc:
    mrc.set_data(numpy_projections.astype(np.float32))

estimated_volume, est_rots = cryo_abinitio_TO('O', file_name, 'reconstructed_' + file_name,
                     rotation_resolution=150,
                     n_theta = 360,
                     n_r_perc=50,
                     cg_max_iterations=50,
                     max_shift_perc=0,
                     true_rotations=true_rotations)

if show_projections:
    est_rots = np.float32(est_rots)
    projections_est = estimated_volume.project(est_rots[0:10])
    show(projections_est, Title='Projections from reconstructed volume and estimated rotations')
                                # can be compared to those shown previously

norms = np.linalg.norm(est_rots[:10], ord=2, axis=(1, 2))
# Print the norms
print(norms)