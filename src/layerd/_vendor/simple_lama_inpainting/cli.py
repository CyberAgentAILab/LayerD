from .models.model import SimpleLama
from PIL import Image
from pathlib import Path
import fire


def main(image_path: str, mask_path: str, out_path: str | None = None) -> None:
    """Apply lama inpainting using given image and mask.

    Args:
        img_path (str): Path to input image (RGB)
        mask_path (str): Path to input mask (Binary 1-CH Image.
                        Pixels with value 255 will be inpainted)
        out_path (str, optional): Optional output image path.
                        If not provided it will be saved to the same
                            path as input image.
                        Defaults to None.
    """
    image_path_obj = Path(image_path)
    mask_path_obj = Path(mask_path)

    img = Image.open(image_path_obj).convert("RGB")
    mask = Image.open(mask_path_obj).convert("L")

    assert img.mode == "RGB" and mask.mode == "L"

    lama = SimpleLama()
    result = lama(img, mask)
    out_path_obj: Path
    if out_path is None:
        out_path_obj = image_path_obj.with_stem(image_path_obj.stem + "_out")
    else:
        out_path_obj = Path(out_path)

    Path.mkdir(out_path_obj.parent, exist_ok=True, parents=True)
    result.save(out_path_obj)
    print(f"Inpainted image is saved to {out_path_obj}")


def lama_cli() -> None:
    fire.Fire(main)


if __name__ == "__main__":
    fire.Fire(main)
