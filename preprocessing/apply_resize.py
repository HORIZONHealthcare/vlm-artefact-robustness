import argparse
from pathlib import Path
import pandas as pd
from PIL import Image


# -------------------------
# resize function (no padding)
# -------------------------
def resize_keep_ratio_no_pad(pil_img, max_side=384):
    """
    Keep aspect ratio.
    - If max(w,h) <= max_side: return original size (converted to L).
    - Else: downscale so that max(w,h) == max_side.
    No padding.
    """
    if max_side <= 0:
        raise ValueError("max_side must be positive")
    pil_img = pil_img.convert("L")
    w, h = pil_img.size

    if max(w, h) <= max_side:
        return pil_img

    scale = max_side / float(max(w, h))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    return pil_img.resize((new_w, new_h), Image.BILINEAR)


# -------------------------
# main resize function (no noise)
# -------------------------
def resize_images_from_csv(
    csv_path,
    output_root,
    out_csv_path,
    max_side=384,
    data_root=None,
):

    df = pd.read_csv(csv_path)
    output_root = Path(output_root).expanduser().resolve()

    new_rows = []

    for _, row in df.iterrows():

        img_path = Path(str(row["filepath"])).expanduser()
        if not img_path.is_absolute() and data_root is not None:
            img_path = Path(data_root).expanduser() / img_path
        dataset = str(row["dataset"])
        label = str(row["binarylabel"]).lower()

        if not img_path.exists():
            print("skip:", img_path)
            continue

        # Resize the clean source without adding any corruption.
        with Image.open(img_path) as img:
            img_resized = resize_keep_ratio_no_pad(img, max_side=max_side)

        # save path
        save_dir = output_root / dataset / label / "clean"
        save_dir.mkdir(parents=True, exist_ok=True)

        save_path = save_dir / img_path.name

        # (optional) keep format consistent
        # If you want to force PNG output, uncomment next line and change suffix.
        # save_path = save_path.with_suffix(".png")

        img_resized.save(save_path)

        # ---- new csv row ----
        new_row = row.copy()
        new_row["filepath"] = str(save_path)
        new_row["filename"] = save_path.name
        new_row["corruption"] = "clean"
        new_row["severity"] = "clean"
        new_row["corruption_severity"] = ""
        new_rows.append(new_row)

    new_df = pd.DataFrame(new_rows)

    Path(out_csv_path).parent.mkdir(parents=True, exist_ok=True)
    new_df.to_csv(out_csv_path, index=False, encoding="utf-8-sig")

    print(f"Done. images: {len(new_df)}")
    print(f"CSV saved -> {out_csv_path}")


# -------------------------
# run
# -------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Resize grayscale images without padding or added noise."
    )
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--data_root",
        default=None,
        help="Optional root used to resolve relative image paths in the input CSV",
    )
    parser.add_argument("--max_side", type=int, default=384)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    resize_images_from_csv(
        csv_path=args.input_csv,
        output_root=args.output_dir,
        out_csv_path=args.output_csv,
        max_side=args.max_side,
        data_root=args.data_root,
    )
