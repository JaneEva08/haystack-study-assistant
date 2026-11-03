import json
from urllib.parse import urlparse, urljoin, urlencode
import requests


def get_public(url, allowed, max_bytes=2_000_000):
    for _ in range(4):
        p = urlparse(url)
        if p.scheme != "https" or p.hostname not in allowed or p.username or p.password or p.port not in (None, 443):
            raise ValueError("Адрес внешнего источника не разрешён.")
        try:
            with requests.get(url, timeout=25, allow_redirects=False, stream=True) as r:
                if r.status_code in (301, 302, 303, 307, 308):
                    url = urljoin(url, r.headers.get("Location", ""))
                    continue
                if r.status_code != 200:
                    raise RuntimeError(f"Источник вернул HTTP {r.status_code}.")
                parts, length = [], 0
                for part in r.iter_content(65536):
                    length += len(part)
                    if length > max_bytes:
                        raise ValueError("Ответ источника слишком большой.")
                    parts.append(part)
                return b"".join(parts), r.headers.get("Content-Type", ""), url
        except requests.RequestException:
            raise RuntimeError("Внешний источник недоступен или истекло время ожидания.") from None
    raise RuntimeError("Слишком много перенаправлений.")


def get_json(url, allowed):
    body, mime, final = get_public(url, allowed)
    if "json" not in mime.lower():
        raise ValueError("Источник вернул не JSON.")
    try:
        return json.loads(body)
    except (ValueError, UnicodeError):
        raise ValueError("Некорректный JSON внешнего источника.") from None


def cat_fact():
    obj = get_json("https://catfact.ninja/fact", {"catfact.ninja"})
    fact = obj.get("fact") if isinstance(obj, dict) else None
    if not isinstance(fact, str) or not fact.strip():
        raise ValueError("API не вернул факт.")
    return {"fact": fact, "source": "https://catfact.ninja/fact", "verification": "unverified_external_claim"}


def weather(city):
    if not isinstance(city, str) or not 1 <= len(city.strip()) <= 100:
        raise ValueError("Укажи город длиной до 100 символов.")
    geocode = "https://geocoding-api.open-meteo.com/v1/search?" + urlencode(
        {"name": city.strip(), "count": 1, "language": "ru", "format": "json"})
    results = get_json(geocode, {"geocoding-api.open-meteo.com"}).get("results", [])
    if not results:
        return {"error": "Город не найден. Уточни название."}
    place = results[0]
    if not isinstance(place, dict) or any(not isinstance(place.get(k), (int, float)) for k in ("latitude", "longitude")):
        raise ValueError("Геокодер не вернул координаты города.")
    url = "https://api.open-meteo.com/v1/forecast?" + urlencode({"latitude": place["latitude"],
        "longitude": place["longitude"], "current": "temperature_2m,apparent_temperature,wind_speed_10m",
        "timezone": "auto"})
    result = get_json(url, {"api.open-meteo.com"})
    current = result.get("current", {})
    if not isinstance(current.get("temperature_2m"), (int, float)) or "time" not in current or "timezone" not in result:
        raise ValueError("Нет текущих данных погоды.")
    return {"city": place["name"], "country": place.get("country", ""), "timezone": result["timezone"],
            "current": current, "units": result.get("current_units", {}), "source": "https://open-meteo.com/"}


def dog_image():
    obj = get_json("https://dog.ceo/api/breeds/image/random", {"dog.ceo"})
    if obj.get("status") != "success" or not isinstance(obj.get("message"), str):
        raise ValueError("API не вернул изображение собаки.")
    body, mime, url = get_public(obj["message"], {"images.dog.ceo"}, 8_000_000)
    mime = mime.split(";", 1)[0].lower()
    if not ((mime == "image/jpeg" and body.startswith(b"\xff\xd8")) or
            (mime == "image/png" and body.startswith(b"\x89PNG\r\n\x1a\n"))):
        raise ValueError("Неподдерживаемое или повреждённое изображение.")
    return body, mime, url
