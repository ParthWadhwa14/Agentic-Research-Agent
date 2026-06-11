import re
import warnings
from pathlib import Path
from typing import Optional

try:
    import requests
except Exception:
    requests = None

try:
    from ddgs import DDGS
except Exception:
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=r".*duckduckgo_search.*renamed.*")
            from duckduckgo_search import DDGS
    except Exception:
        DDGS = None

try:
    from PIL import Image
except Exception:
    Image = None


def clean_filename(text: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9_-]+", "_", str(text or "image"))
    return text[:80].strip("_").lower() or "image"


def clean_search_topic(topic: str, max_words: int = 8) -> str:
    topic = re.sub(r"\s+", " ", str(topic or "")).strip()
    topic = re.sub(r"(?i)\b(?:please|create|make|generate|find|search|presentation|ppt|slides?|deck|try to|tell|give|image|images|picture|pictures|photo|photos)\b", " ", topic)
    topic = re.sub(r"[^a-zA-Z0-9 ,&-]+", " ", topic)
    words = [word for word in topic.split() if len(word) > 2]
    return " ".join(words[:max_words]).strip() or "education"


def is_generic_logo_topic(topic: str) -> bool:
    topic = str(topic or "").lower()
    generic_terms = ["education", "poverty", "india", "country", "statewise", "school", "learning"]
    brand_terms = ["company", "brand", "startup", "logo", "organization", "ngo", "university"]
    return any(term in topic for term in generic_terms) and not any(term in topic for term in brand_terms)


def download_image(url: str, save_path: Path) -> Optional[str]:
    if requests is None or Image is None:
        return None

    try:
        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
        if response.status_code != 200:
            return None
        if "image" not in response.headers.get("content-type", ""):
            return None

        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_bytes(response.content)

        try:
            with Image.open(save_path) as image:
                image.verify()
        except Exception:
            save_path.unlink(missing_ok=True)
            return None

        return str(save_path)
    except Exception:
        save_path.unlink(missing_ok=True)
        return None


def search_and_download_images(query: str, output_dir: Path | str, max_images: int = 3) -> list[str]:
    image_paths: list[str] = []
    if DDGS is None:
        return image_paths

    output = Path(output_dir)
    try:
        with DDGS() as ddgs:
            search_kwargs = {
                "max_results": max_images * 5,
                "safesearch": "moderate",
            }
            try:
                results = ddgs.images(query=query, **search_kwargs)
            except TypeError:
                results = ddgs.images(keywords=query, **search_kwargs)

            for index, result in enumerate(results):
                image_url = result.get("image") or result.get("url") or result.get("thumbnail")
                if not image_url:
                    continue

                save_path = output / f"{clean_filename(query)}_{index}.jpg"
                path = download_image(image_url, save_path)
                if path:
                    image_paths.append(path)
                if len(image_paths) >= max_images:
                    break
    except Exception as exc:
        print(f"Image search skipped: {exc}")

    return image_paths


def collect_logo(topic: str, output_dir: Path | str) -> Optional[str]:
    if is_generic_logo_topic(topic):
        return None

    logo_query = f"{clean_search_topic(topic)} logo transparent png"
    logos = search_and_download_images(logo_query, output_dir=output_dir, max_images=1)
    return logos[0] if logos else None


def collect_slide_images(topic: str, output_dir: Path | str, plan: Optional[str] = None, max_images: int = 6) -> list[str]:
    clean_topic = clean_search_topic(topic)
    query = f"{clean_topic} presentation education classroom" if plan else f"{clean_topic} high quality presentation images"
    return search_and_download_images(query, output_dir=output_dir, max_images=max_images)
