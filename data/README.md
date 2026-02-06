# Celtic Coins - merged dataset - Die Muenze 44

This folder contains scripts for building a merged metadata table for Celtic coin images (obverse/reverse) from the *occ* and *coinarchives* dataset plus a **manual 3-level condition rating per side**.

This documentation describes the process of merging the datasets in full detail.

## Files

- **`merged_dataset_rated.csv`**: main dataset table (one row per coin record).
- **`obv/`**: obverse images and their corresponding mask named `<id>.jpg` and `<id>_mask.jpg` respectively.
- **`rev/`**: reverse images and their corresponding mask named `<id>.jpg` and `<id>_mask.jpg` respectively.

## Row semantics

Each row represents one coin record.

- `id`: record identifier inside the merged dataset. If you need a stable key across sources, prefer `(dataset, id)`.
- `dataset`: source tag (observed values include `occ` and `coinarchives`).
- `label`: coin type label (e.g. `Manching 2`, `Stachelhaar a`).

## Condition ratings

- `obv_condition`, `rev_condition`: per-side condition rating in `{0, 1, 2}`.
  - These values were **assigned manually**.
  - Treat the rating as a **coarse image/coin-side quality score** (higher generally means “better / more usable”).

## Merging process (how the dataset was created)

The dataset was created by cleaning the two source datasets (OCC and Coinarchives), standardizing their metadata schema, generating per-image coin masks, and finally merging everything into a single table.

### 1) Cleaning up the OCC dataset

Preprocessing (images):

- Crop images where a scale is visible in the background (IDs `6541`–`6552`).

Raw input format notes:

- OCC metadata is stored as CSV with **semicolon** (`;`) as delimiter.
- Relevant original columns: `Id`, `Type | Code | Code`, `Type | Obverse design | Obverse design | Description`, `Type | Reverse design | Reverse design | Description`, `Collection | Name`, `Collection | Surname`, `Collection | Name 2`, `Inventory number`, `Weight`, `Diameter max.`, `Findspot | Name`.

Cleanup steps (metadata):

- Remove rows with blank type.
- Normalize/align labels in `Type | Code | Code` to match Coinarchives labels where needed, e.g.
  - `KS 1/8/1 Horse with backward facing head` &rarr; `Horse with backward facing head`
- Drop columns `Collection | Name` and `Collection | Surname` (empty).
- Rename columns to the merged schema:
  - `Id` &rarr; `id`
  - `Type | Code | Code` &rarr; `label`
  - `Type | Obverse design | Obverse design | Description` &rarr; `obverse_description`
  - `Type | Reverse design | Reverse design | Description` &rarr; `reverse_description`
  - `Collection | Name 2` &rarr; `collection`
  - `Inventory number` &rarr; `inv_num`
  - `Weight` &rarr; `weight`
  - `Diameter max.` &rarr; `max_diameter`
  - `Findspot | Name` &rarr; `findspot`

Handling duplicate OCC IDs:

- If two rows share the same `id`, keep the first row and store the second row’s `label` in `label_alt`.
- If there is only one row for an `id`, `label_alt` stays empty.
- Exception: if the first duplicate row has label `Manching 2`, keep the **second** row and set its `label_alt` to `Manching 2`.

De-duplication and segmentation:

- Detect duplicate images by generating an image fingerprint per side (after applying EXIF rotation if present).
- Create coin masks for all images using SAM3; store masks as `<id>_mask.jpg` next to `<id>.jpg`.

Manual modifications after running the OCC cleanup:

- Remove duplicates flagged by the script (from `occ_image_duplicates.csv`), manually resolving cases where only one side is duplicated or a pair is mismatched:
  - `2250` &lrarr; `5058` (remove `2250`)
  - `2251` &lrarr; `5059` (remove `2251`)
  - `2252` &lrarr; `5060` (remove `2252`)
  - `2254` &lrarr; `5061` (remove `2254`)
  - `2255` &lrarr; `5062` (remove `2255`)
  - `4777` &lrarr; `4784` (remove `4784`)
  - `5919` (obv and rev identical; removed)
  - `6012` &lrarr; `6014` (remove `6014`)
  - `4826` &lrarr; `4825` (remove `4825`)
  - `5651` &lrarr; `5663` (remove `5663`)
  - `5784` &lrarr; `5790` (remove `5784`)
  - `5804` &lrarr; `5817` (remove `5804`)
- Manually rate the condition for each side using `{0,1,2}` (GUI tool).
- Remove items that could not be segmented (from `occ_coin_mask_quality.csv` where `mask_quality = -1`):
  - `4833`, `5119`, `5120`, `5320`, `5423`, `5459`, `5570`, `5622`, `5801`, `5847`, `5916`, `5924`, `5925`, `5928`, `6595`

Implementation reference: the OCC cleanup logic lives in [data/cleanup/occ/clean_occ_csv.py](cleanup/occ/clean_occ_csv.py).

### 2) Cleaning up the Coinarchives dataset

Manual changes to input images before running cleanup:

- `1239539_image00039.jpg` (remove duplicate pair)
- `1361019_image01946.jpg` (remove drawing and duplicate face)
- `2277041_image01036.jpg` (remove coin pair with label on top)

Raw input format notes:

- Coinarchives metadata columns: `LotID`, `Titel`, `Beschreibung`, `Auktionsdatum`, `Detail-Link`, `Bild-Link`, `Typ nach OCC 2025-10-27`.

Cleanup steps (metadata):

- Rename/reorder columns to the merged schema:
  - `LotID` &rarr; `id`
  - `Typ nach OCC 2025-10-27` &rarr; `label`
  - `Titel` &rarr; `title`
  - `Beschreibung` &rarr; `description`
  - `Auktionsdatum` &rarr; `auction_date`
  - `Detail-Link` &rarr; `url`
  - `Bild-Link` &rarr; `img_url`
- Extract `weight` and `max_diameter` from natural-language text (keep only the numeric part):
  - Weight regex: `(-|\s|\()+[0-9]+(,|\.)[0-9]+\s*g(\s|\.|\)|,)`
  - Diameter regex: `(=|-|\s|\()+[0-9]+(,|\.)?[0-9]?\s*mm(\s|\.|\)|,|;|m)`
- Remove entries with invalid pairs:
  - `2192365`, `2192366`, `2444062`, `2444063`, `2543763`, `2543764`, `2280339`

Mask/segmentation steps (images):

- Detect a coin mask and check it forms one coherent mass (otherwise flag artifacts).
- Detect label region and check overlap with the coin mask (if overlap, stop and output a message).
- Segment the coin and store the mask separately; add a border of `max(coin_width, coin_height) * 0.015` pixels around the segmented coin and fill the remaining background with white.

Notes from running `clean_coinarchives_csv.py` (script outputs to review):

- Hole in masks: `34140`, `59089`, `615182`, `1463622`, `1626094`
- Multiple coins: `338050`, `338062`, `709172`, `1124217`, `1220767`, `1463623`, `1638576`, `1848409`, `2091429`, `2192345`, `2192346`, `2277040`, `2277049`, `2277050`, `2444066`, `2444083`
- Not enough masks/boxes (threshold 0.25): `2277047`, `2363028`, `2363030`
- Logo overlaps with mask and was cut out: `1848409`, `1848411`, `2091429`, `2192345`, `2192346`, `2277040`, `2277049`, `2444065`, `2444066`, `2444083`, `2543751`

Manual modifications after running the Coinarchives cleanup:

1. Fix holes/artifacts in masks for: `34140`, `59089`, `615182`, `1463622`, `1626094`
2. Split multi-coin images and adapt weights/diameters for: `338050`, `338062`, `709172`, `1124217`, `1220767`, `1463623`, `1638576`, `1848409`, `2091429`, `2192345`, `2192346`, `2277040`, `2277049`, `2277050`, `2444066`, `2444083`
3. Remove `2444065` (contains 4 individual coins, not pairs)
4. Manually split and generate masks for: `2277047`, `2363028`, `2363030`
5. Run the side-classification network and correct misclassifications after a brief manual check:
   - `1994868`, `2444068`, `2543766`, `2545993`

Implementation reference: the Coinarchives cleanup logic lives in [data/cleanup/coinarchives/clean_coinarchives_csv.py](cleanup/coinarchives/clean_coinarchives_csv.py).

### 3) Merging both cleaned datasets

After both datasets are cleaned and standardized to the same schema (including image files and masks), they are combined into a single merged table.

Implementation reference: [data/cleanup/merge_datasets.py](cleanup/merge_datasets.py).

## Columns (schema)

The CSV has the following columns (some may be empty depending on the source):

- `id`: record ID
- `dataset`: source dataset name
- `label`: coin type label
- `obverse_description`: text description of the obverse motif
- `reverse_description`: text description of the reverse motif
- `collection`: holding institution / collection name
- `inv_num`: inventory number
- `weight`: weight (as provided by source; typically grams)
- `max_diameter`: maximum diameter (as provided by source; units may vary)
- `findspot`: provenance / findspot (if available)
- `label_alt`: alternative label (assigned if two entries with the same id existed in the original occ dataset)
- `title`: listing title (primarily for auction/online sources)
- `description`: long-form listing/lot description
- `auction_date`: auction date (if applicable)
- `url`: web page URL for the record/lot
- `img_url`: source image URL (if applicable)
- `obv_condition`: manual condition rating for the obverse (0/1/2)
- `rev_condition`: manual condition rating for the reverse (0/1/2)
