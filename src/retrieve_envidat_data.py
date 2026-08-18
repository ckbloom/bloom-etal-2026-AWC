import zipfile
from pathlib import Path
from typing import Iterable, Union
import requests
import shutil


def download_from_envidat(
        output_dir: Union[str, Path] = "../data/swiss_soil_maps",
        target_maps: Iterable[str] = (
                "clay_maps",
                "sand_maps",
                "gravel_maps",
                "density_maps",
                "soc_maps",
                "depth_map",
        ),
        package_id: str = "soil-property-maps-for-the-swiss-forest",
) -> None:
    """Downloads and extracts soil property maps of Baltensweiler et al. (2021) and discoloration maps of
    Bloom et al. (2026) both hosted on EnviDat.
    Args:
        output_dir: The directory where the downloaded and extracted files will
          be saved.
        target_maps: A collection of resource names (excluding .zip) to filter
          and download.
        package_id: The EnviDat package ID dataset name.
    """
    # Ensure output directory exists
    out_path = Path(output_dir)
    out_path.mkdir(exist_ok=True, parents=True)
    # Normalize target names for reliable lookup
    targets = {name.strip().lower() for name in target_maps}
    api_url = f"https://www.envidat.ch/api/action/package_show?id={package_id}"
    # Fetch metadata
    print(f"Fetching metadata for package '{package_id}'...")
    resp = requests.get(api_url, timeout=30)
    resp.raise_for_status()
    resources = resp.json()["result"]["resources"]
    # Filter and download
    for resource in resources:
        # Strip extension and lower-case for robust matching
        raw_name = resource["name"]
        clean_name = (
            raw_name.replace(".zip", "").replace(".ZIP", "").strip().lower()
        )
        if clean_name not in targets:
            continue
        url = resource["url"]
        dest = out_path / raw_name
        print(f"\nDownloading {raw_name} ...")
        with requests.get(url, stream=True, timeout=120) as dl:
            dl.raise_for_status()
            total = int(dl.headers.get("content-length", 0))
            downloaded = 0
            with open(dest, "wb") as f:
                for chunk in dl.iter_content(chunk_size=1024 * 256):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        pct = downloaded / total * 100
                        print(f"  {pct:.1f}%", end="\r")
        print(f"  Saved to {dest}")
        # Extracting directly into out_path, flattening any internal folders
        print(f"  Extracting {dest.name} ...")
        with zipfile.ZipFile(dest) as zf:
            for member in zf.namelist():
                # Skip directory entries
                if member.endswith("/"):
                    continue
                member_path = zipfile.Path(zf, member)
                target_file = out_path / Path(member).name
                with zf.open(member) as src, open(target_file, "wb") as f:
                    shutil.copyfileobj(src, f)
        dest.unlink()  # Remove zip after extraction
        print(f"  Extracted to {out_path}/")


if __name__ == "__main__":
    pass
