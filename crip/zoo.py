def simuBeerLambert(saveDir: str, filename: str, phantom: NDArray, mapping: Dict[Or[int, str], Atten], spec: Spectrum,
                    photon: int, fpjCfg: MgfpjConfig, fbpCfg: MgfbpConfig, clean: bool):
    # prepare working directories
    list(map(lambda x: os.makedirs(join(saveDir, x), exist_ok=True), ['len_img', 'len_sgm', 'sgm', 'rec', 'rec_tif']))

    # separate materials in the image domain
    for key in mapping:
        sep = np.float32(np.bool_(phantom == int(key)))
        imwriteRaw(sep, join(saveDir, f'len_img/img_{key}.raw'))

    # forward project materials to get lengths
    cfg = fpjCfg
    cfg.setIO(abspath(join(saveDir, 'len_img')),
              abspath(join(saveDir, 'len_sgm')),
              '.*',
              OutputFileReplace=['img_', 'sgm_'])
    Mgfpj().exec(cfg)
    lenSgms = []
    attens = []
    for key in mapping:
        lenSgms.append(imreadRaw(join(saveDir, f'len_sgm/sgm_{key}.raw'), cfg.Views, cfg.DetectorElementCount))
        attens.append(mapping[key])

    # Beer-Lambert forward projection
    flat = forwardProjectWithSpectrum([], [], spec, 'EID')
    sgm = forwardProjectWithSpectrum(lenSgms, attens, spec, 'EID')
    postlog = flatDarkFieldCorrection(sgm, flat)
    if photon > 0:
        postlog = injectPoissonNoise(postlog, 'postlog', photon)
    imwriteRaw(postlog, join(saveDir, f'sgm/sgm_{filename}.raw'), np.float32)

    # recon
    cfg = fbpCfg
    cfg.setIO(abspath(join(saveDir, f'sgm')), abspath(join(saveDir, 'rec')), '.*', OutputFileReplace=['sgm_', 'rec_'])
    Mgfbp().exec(cfg)
    img = imreadRaw(join(saveDir, f'rec/rec_{filename}.raw'), cfg.ImageDimension, cfg.ImageDimension)
    imwriteTiff(img, join(saveDir, f'./rec_tif/rec_{filename}.tif'))

    if clean:
        list(map(lambda x: shutil.rmtree(join(saveDir, x)), ['len_img', 'len_sgm', 'sgm', 'rec']))

    return img
