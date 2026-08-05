Here's the integration — I've kept your existing structure exactly intact and just inserted the label-matching logic at the two points where it needs to plug in.

**1. Add these two helper functions once, near your other extraction helpers** (`extract_slide_text`, `extract_table_text`, `get_speaker_notes`, `clean_text`) — likely in the same cell block earlier in the notebook, above your `for shape_index, shape in enumerate(...)` loop:

```python
def get_shape_bounds(shape):
    return {
        "left": shape.left,
        "top": shape.top,
        "right": shape.left + shape.width,
        "bottom": shape.top + shape.height,
        "center_x": shape.left + shape.width / 2,
    }

def link_labels_to_images(slide):
    labels = []
    images = []

    for shape in slide.shapes:
        if shape.shape_type == MSO_SHAPE_TYPE.PICTURE:
            images.append(shape)
        elif getattr(shape, "has_text_frame", False):
            text = shape.text_frame.text.strip()
            if text:
                labels.append(shape)

    associations = []
    for label in labels:
        lb = get_shape_bounds(label)
        best_match = None
        best_distance = float("inf")

        for img in images:
            ib = get_shape_bounds(img)
            horizontal_overlap = ib["left"] - 100000 <= lb["center_x"] <= ib["right"] + 100000
            if not horizontal_overlap:
                continue
            vertical_gap = abs(ib["top"] - lb["top"])
            if vertical_gap < best_distance:
                best_distance = vertical_gap
                best_match = img

        associations.append({
            "label_text": label.text_frame.text.strip(),
            "image_shape_id": best_match.shape_id if best_match else None,
        })

    return associations
```

**2. Right before your shape loop** (just above line 171, `for shape_index, shape in enumerate(slide.shapes, start=1):`), add:

```python
# Build a lookup so each picture shape can find its associated title/label
label_associations = link_labels_to_images(slide)
image_shape_id_to_label = {
    assoc["image_shape_id"]: assoc["label_text"]
    for assoc in label_associations
    if assoc["image_shape_id"] is not None
}
```

**3. Inside your existing `if shape.shape_type == MSO_SHAPE_TYPE.PICTURE:` block** (lines 195–218), add the lookup and thread the label through into both `object_record` and the OCR tagging:

```python
if shape.shape_type == MSO_SHAPE_TYPE.PICTURE:
    image = shape.image
    image_extension = image.ext or "png"
    image_file_name = (
        f"{document_label}_slide_{slide_number:03d}_{slide_label}_image_{shape_index:03d}.{image_extension}"
    )
    local_image_path = os.path.join(local_extracted_images_dir, image_file_name)
    with open(local_image_path, "wb") as image_file:
        image_file.write(image.blob)

    dbfs_image_path = to_dbfs_volume_path(local_image_path)
    image_paths.append(dbfs_image_path)
    written_image_count += 1
    object_record["image_path"] = dbfs_image_path

    # --- NEW: attach the matched label, if any ---
    associated_label = image_shape_id_to_label.get(shape.shape_id)
    if associated_label:
        object_record["associated_label"] = associated_label

    try:
        image_for_ocr = Image.open(BytesIO(image.blob))
        ocr_text = clean_text(pytesseract.image_to_string(image_for_ocr))
        if ocr_text:
            # --- CHANGED: embed the label into the OCR tag ---
            label_attr = f' label="{associated_label}"' if associated_label else ""
            ocr_text = f"<image{label_attr}>\n" + ocr_text + "\n</image>\n"
            ocr_text_parts.append(ocr_text)
            object_record["ocr_preview"] = ocr_text[:250]
    except Exception as exc:
        object_record["ocr_error"] = str(exc)[:250]

extracted_objects.append(object_record)
```

**What this gets you downstream:** your `ocr_text` field per slide will now read like:
```
<image label="Control">
peacock ... Together for one great price ...
</image>
<image label="Variant">
peacock ... Save over 30% on Apple TV and Peacock ...
</image>
```
— so when this feeds into your knowledge-assist agent, it can distinguish which OCR block belongs to Control vs. Variant instead of getting two unlabeled walls of text. `object_record["associated_label"]` also gives you that same info structured, in case you want to query `extracted_objects_json` for it directly (e.g. filter for `object_category == "picture" AND associated_label == "Variant"`) without re-parsing the OCR tags.

One thing worth double-checking on your actual deck: `MSO_SHAPE_TYPE` is already imported in this notebook per your Bronze_ingestion screenshot context — if `link_labels_to_images` throws a `NameError` on it, just make sure `from pptx.enum.shapes import MSO_SHAPE_TYPE` is present above where you define the helper (you already have this import in your other notebook per Image 1).
