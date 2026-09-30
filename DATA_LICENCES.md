# Data and pretrained weights

This repository redistributes no images, annotations, or pretrained weights.
`warpaudit prepare-data` downloads each archive from its official source and
refuses to extract it unless the SHA-256 hash matches. `manifests/` records
relative paths, image sizes, and hashes only.

| Dataset / artifact | Official source | Verified archive | Terms |
|---|---|---|---|
| FIRE | https://projects.ics.forth.gr/cvrl/fire/ | `FIRE.7z`, 276,589,879 bytes, SHA-256 `4f164d89a572d88a5f46bf7fe433813a21561163c83481d6e101573621f79700` | Public download; no explicit licence found on the official page, so FIRE files are used locally only. Cite Hernandez-Matas et al., J. Model. Ophthalmol. 1(4), 2017. |
| COph100 annotations | https://doi.org/10.6084/m9.figshare.27061084.v1 | 128,066,919 bytes, SHA-256 `407ca917280e4f5395b236ffd57096b7ddb2ff3ec362fe150c8ff47c6468e639` | CC BY 4.0. Cite Hu et al., Sci. Data 12, 99 (2025). |
| RIDIRP source images (for COph100) | https://doi.org/10.6084/m9.figshare.24565681.v1 | 2,692,475,985 bytes, SHA-256 `c07f82340210e8a99bffca99a270a5e6d2c00bb5f1b75e3bc7dac98ad4696163` | CC0. Cite Timkovič et al., Sci. Data 11, 814 (2024). |
| XFeat weights | https://github.com/verlab/accelerated_features (commit `e92685f`) | `xfeat.pt` | Upstream terms |
| SuperPoint + LightGlue weights | https://github.com/cvg/LightGlue/releases/tag/v0.1_arxiv (commit `eb42fee`) | `superpoint_v1.pth`, `superpoint_lightglue_v0-1_arxiv.pth` | Upstream terms |
| SuperRetina weights | https://github.com/ruc-aimc-lab/SuperRetina (released `SuperRetina.pth`) | see `scripts/run_isbi_submission.ps1` | Upstream terms |

COph100 pairs use the exact RIDIRP examinations named by the annotations. The
image payloads embedded in the LabelMe files differ from those examinations and
are never used.

The only source pixels in this repository are the two CC0 RIDIRP images in the
paper's Fig. 1 (`paper/isbi_2027/assets/qualitative_pole.*`).

The committed results in `reports/` hold derived per-case values (landmark
errors, homography entries, scores, predictions), not images.
