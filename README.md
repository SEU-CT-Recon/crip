<p align="center">
  <img src="crip.png" />
</p>
An all-in-one tool for Cone-Beam CT Data IO, Pre/Post-process, Physics, Dual Energy, Low Dose and everything only except Reconstruction.

:warning: crip is still under very "agile" development and **no** API consistency between versions are guaranteed. Use at your own risk! :construction:

## Install

- Via [PyPI](https://pypi.org/project/crip/)

  [![Upload Python Package](https://github.com/SEU-CT-Recon/crip/actions/workflows/python-publish.yml/badge.svg)](https://github.com/SEU-CT-Recon/crip/actions/workflows/python-publish.yml)

  ```sh
  pip install crip
  ```

- Via Wheel

  Download the latest release from [Releases](https://github.com/SEU-CT-Recon/crip/releases). Install the wheel by

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
- `crip.mangoct` for [mangoct](https://github.com/SEU-CT-Recon/mandoct) reconstruction tool package integration.
- `crip.metric` for metrics computation.
- `crip.plot` for figure drawing.

crip is still under development. More features will be added in the future. And contributions are strongly welcomed.

## Documentation and Example

The [Official documentation](seu-ct-recon.github.io/crip), and [Official example](./example). Ask everything in [issue](https://github.com/SEU-CT-Recon/crip/issues).

## License

MIT

## Cite This

```
@software{ZHUO_crip,
  author = {ZHUO, Xu and LU, Yuchen and SEU-CT-Recon},
  license = {MIT},
  title = {{crip}},
  url = {https://github.com/SEU-CT-Recon/crip}
}
```

We appreciate if you [let us know](https://github.com/SEU-CT-Recon/crip/issues) you are using crip in your workflow or publications.

We are considering publishing crip to [Journal of Open Source Software](https://joss.theoj.org/) and/or apply for a Software Copyright in China in the future.
