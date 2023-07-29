<p align="center">
  <img src="crip.png" />
</p>
An all-in-one tool for Cone-Beam CT Data IO, Pre/Post-process, Physics, Dual Energy, Low Dose and everything only except Reconstruction.

## Install

- Via [PyPI](https://pypi.org/project/crip/)

  ```sh
  pip install crip
  ```

- Via Wheel

  Download the latest release from [Releases](https://github.com/z0gSh1u/crip/releases). Install the wheel by

  ```sh
  pip install /path/to/latest/release.whl
  ```

## Modules

All functions are (or will be) with clear documentation. Just consider the function name and explore the source code, or refer to examples to start.

- `crip.de` for Dual-Energy CT.
- `crip.io` for file read/write (RAW, DICOM and TIFF supported).
- `crip.physics` for attenuation, spectrum, coefficient calculation, etc.
- `crip.postprocess` for data post-processing.
- `crip.preprocess` for data pre-processing.
- `crip.shared` for other common operations.
- `crip.lowdose` for Low-Dose researches.
- `crip.mangoct` for [mangoct](https://github.com/CandleHouse/mandoct) reconstruction tool package integration.
- `crip.metric` for metrics computation.
- `crip.paplot` for figure drawing.

crip is still under development. More features will be added in the future. And contributions are strongly welcomed.

## Documentation and Example

The [Official documentation](https://z0gsh1u.github.io/crip/), and [Official example](./example). Ask everything in [issue](https://github.com/z0gSh1u/crip/issues).

### Flat Field Correction and Prepare Sinogram, then Recon

```python
from crip.io import imreadRaw, imwriteRaw, listDirectory
from crip.preprocess import averageProjections, flatDarkFieldCorrection, projectionsToSinograms
# Average to get flat field
flats = [imreadRaw(path, H, W) for path in listDirectory('...', style='fullpath')]
flat = averageProjections(flats)
# Flat field correction to get post-log
projs = [imreadRaw(path, H, W) for path in listDirectory('...', style='fullpath')]
projs = flatDarkFieldCorrection(projs, flat)
sinos = projectionsToSinograms(projs)
imwriteRaw(sinos, '...')
# Recon using mangoct
config = MgfbpConfig()
config.setIO('./sgm/', './rec/', 'sgm.*', OutputFileReplace=['sgm', 'rec'])
config.setGeometry(SID, SDD, -360)
config.setSgmConeBeam(W, NViews, NViews, ElementSize, H, ElementSize)
config.setRecConeBeam(512, PixelSize, 128, SliceThickness, 'HammingFilter', 1, 90)
fbp = Mgfbp(tempDir='./Temp')
fbp.exec(config)
```

### Dual-Energy Material Decomposition

```python
from crip.io import listDirectory, imreadDicom
from crip.physics import Atten, Spectrum
from crip.de import deDecompGetCoeff, deDecompProj, deDecompRecon
# Decompose in the projection domain.
LowSpec = Spectrum.fromFile(LowSpecPath, 'eV')
HighSpec = Spectrum.fromFile(HighSpecPath, 'eV')
LowEProj = imreadRaw(f'sgm_{PhantomName}_{LowE}.raw', NView, W, nSlice=H)
HighEProj = imreadRaw(f'sgm_{PhantomName}_{HighE}.raw', NView, W, nSlice=H)
Base1 = Atten.fromBuiltIn('Al')
Base1Range = range(10, 50 + 10, 10) # The length range [mm] to fit \mu.
Base2 = Atten.fromBuiltIn('Water')
Base2Range = range(10, 200 + 10, 10)
Alpha, Beta = deDecompGetCoeff(LowSpec, HighSpec, Base1, Base1Range, Base2, Base2Range)
Decomp1, Decomp2 = deDecompProj(LowEProj, HighEProj, Alpha, Beta)
# Decompose in the image domain.
LowSlices = huNoRescale([imreadDicom(x) for x in listDirectory(LowDir, style='fullpath')]) # linearize HU and \mu
HighSlices = huNoRescale([imreadDicom(x) for x in listDirectory(HighDir, style='fullpath')])
Decomp1, Decomp2 = deDecompRecon(LowSlices, HighSlices, mu1Low, mu1High, mu2Low, mu2High) # \mu_i for each base
```

### Correct Truncation Artifact

```python
from crip.io import imreadTiff, imwriteTiff, listDirectory
from crip.preprocess import padSinogram
from crip.postprocess import fovCrop, fovCropRadius
# Pad the sinogram.
for file in tqdm(listDirectory(SrcFolder)):
  proj = imreadTiff(os.path.join(SrcFolder, file))
  projPad = padSinogram(proj, padding)
  imwriteTiff(projPad, os.path.join(DstFolder, file), np.float32)
# Recon...
# Crop the FOV
rec = imreadTiff(reconVolumePath)
r = fovCropRadius(SOD, SDD, detWidth, reconPixSize)
crop = fovCrop(rec, r, 0)
```

### Calculate Linear Attenuation Coefficient of Water

```python
from crip.physics import calcMu, Atten, Spectrum, getClassicDensity
# A sample spectrum file.
SpectrumFile = '''
48000  100
49000  50
50000  -1
'''
WaterDensity = getClassicDensity('Water')
spec = Spectrum.fromText(SpectrumFile, 'eV')
atten = Atten.fromBuiltIn('Water', WaterDensity)
mu = calcMu(atten, spec, 'PCD')
print(f'mu = {mu} mm-1.')
```

### Radon Transform with Spectrum

```python
from crip.io import imreadRaw
from crip.physics import Atten, Spectrum, forwardProjectWithSpectrum
spec = Spectrum.fromFile(item['Path'], 'eV')
boneMap = imreadRaw(os.path.join(dirname__, 'materialLength/sgm_BoneMap.raw'), nView, W, nSlice=H)
waterMap = imreadRaw(os.path.join(dirname__, 'materialLength/sgm_WaterMap.raw'), nView, W, nSlice=H)
BoneAtten = Atten.fromBuiltIn('Bone')
WaterAtten = Atten.fromBuiltIn('Water')
flat = forwardProjectWithSpectrum([], [], spec, 'EID')
sgmNew = []
for i in trange(H):
  bm = boneMap[i]; wm = waterMap[i]
  sgmNew.append(
    forwardProjectWithSpectrum([bm, wm], [BoneAtten, WaterAtten], spec, 'EID', fastSkip=True, flat=flat))
```

### And more...

## License

MIT

## Cite This

```
@software{ZHUO_crip,
  author = {ZHUO, Xu and LU, Yuchen},
  license = {MIT},
  title = {{crip}},
  url = {https://github.com/z0gSh1u/crip}
}
```

We are considering publishing crip to [Journal of Open Source Software](https://joss.theoj.org/) and/or apply for a Software Copyright in China in the future.
