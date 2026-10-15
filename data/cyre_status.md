# CYRE dataset availability audit

**Audit date:** 2026-10-01  
**Conclusion:** **Not publicly available as a dataset in the sources checked.** The papers are publicly accessible; no public raw CYRE scene/video, participant-rating, or physiology package was located. This is a finding about the audited access points, not a claim that the authors do not retain the data.

## Sources checked

1. **The requested paper (2020):** [Predicting the Level of Cybersickness via Spatiotemporal Perceptual Features Extracted from Virtual Reality Content](https://doi.org/10.5573/ieie.2020.57.11.45). The [Hansung repository record](https://dspace.hansung.ac.kr/handle/2024.oak/1068) lists one downloadable item: the article PDF (1.18 MB). It does not list a dataset archive. The paper describes 52 synthetic scenes and 150 volunteers, but does not provide a data download URL in the repository record. This 2020 study should not be assumed to be identical to the later CYRE dataset release.
2. **The later CYRE article (2022):** [Cybersickness and Its Severity Arising from Virtual Reality Content: A Comprehensive Study](https://doi.org/10.3390/s22041314), also available on [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC8963115/). It explicitly names CYRE and reports 52 scenes and 154 participants. Its Data Availability Statement says “Not applicable”; it does not link a data package.
3. **Hansung repository record for the CYRE article:** [repository item 2024.oak/1798](https://dspace.hansung.ac.kr/handle/2024.oak/1798) lists the article PDF (5.24 MB) as its only file and states repository items are all rights reserved unless otherwise indicated. That PDF is an article, not the dataset.
4. **Author institutional profile and linked homepage:** [Heeseok Oh's Hansung profile](https://hansung.ac.kr/CreCon/2778/subview.do) links to the [Visual Intelligence Lab homepage](https://sites.google.com/view/hs-vilab). The current lab page was checked for CYRE/data links; it describes the lab but exposes no CYRE download or dataset license.
5. **Repository/search checks:** exact-title and CYRE searches, including GitHub-targeted searches, did not surface an author-maintained CYRE data repository or a verified raw-data URL. Search failure alone is not proof of nonexistence; the conclusion above is based on the paper's data statement and the inspected repository/homepage contents.

## Access, license, and size

- **Dataset download URL:** none verified. Do not use the article PDF URLs as data URLs.
- **Dataset license:** unknown/not stated in the inspected dataset sources. The 2022 article is CC BY 4.0 as an article; that does not establish a license for underlying data. The 2020 paper is also available as an article, not a data release.
- **Dataset size:** unknown; no data package was exposed. For distinction only, the repository lists the 2020 and 2022 article PDFs as 1.18 MB and 5.24 MB respectively.
- **Access date:** 2026-10-01. No dataset was downloaded.

## Selected downgrade path

**Path A — Mazed and Confused / VRWalking**, using its VR POV imagery with FMS/SSQ labels and explicitly marking extracted visual features as **path-dependent features**. The [public project repository](https://github.com/Jyotinag/VRWalking_Dataset/) describes an open dataset with VR images and subjective cybersickness measures; it includes a [`VideoLink.txt`](https://github.com/Jyotinag/VRWalking_Dataset/blob/main/VideoLink.txt) entry linking to a Google Drive video. The repository reports FMS ratings on a 1–10 scale once per minute and SSQ before/after exposure, and says 36 of 39 participants were included in the analysis.

The external Drive item's access permission and dataset-specific license were not independently verified during this audit. Before obtaining or processing video files, confirm access and applicable terms. The selected path is not a scene-independent/pre-exposure evaluation: POV imagery reflects the participant's route, head direction, movement, and exposure history. Claims must be limited accordingly. If the linked image/video files or terms cannot be confirmed, pause dataset use and record that limitation rather than silently replacing FMS/SSQ with a proxy or rule-generated label.

## Interpretation of the 150 vs 154 counts

The 2020 spatiotemporal-features paper reports 150 volunteers. The 2022 article that explicitly names the dataset CYRE reports 154 valid participants and 8,008 participant-scene labels. Keep these as separate source statements; do not merge counts or label the 2020 dataset as CYRE without author confirmation.
