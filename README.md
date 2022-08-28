<p align="center">
  <img src="crip.png" />
</p>
An all-in-one tool for Cone-Beam CT Data IO, Pre/Post-process, and Physics, Dual Energy, Low Dose, Deep Learning researches and everything only except Reconstruction.

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

All functions are with clear documentation by comments. Just consider the function name and explore the source code, or refer to examples to start.

- `crip.de` for Dual-Energy CT.
- `crip.io` for file read/write (RAW, DICOM and TIFF supported), and enhanced `listdir`.
- `crip.physics` for attenuation, spectrum, \mu calculation, etc.
- `crip.postprocess` for data post-processing.
- `crip.preprocess` for data pre-processing, like flat field correction.
- `crip.shared` for other common operations.

crip is still under development. More features will be added in the future. And contributions are strongly welcomed.

## Documentation and Example

- [Official documentation](https://z0gsh1u.github.io/crip/)
- [Official example](./example)
- [ArtifactReduction](https://github.com/CandleHouse/ArtifactReduction) using crip
- Ask everything in [issue](https://github.com/z0gSh1u/crip/issues)

## License

MIT
