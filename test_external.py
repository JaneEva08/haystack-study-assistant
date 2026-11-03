import unittest
from unittest.mock import patch
from external_tools import cat_fact, weather, dog_image, get_public


class ExternalTests(unittest.TestCase):
    def test_cat_fact_requires_nonempty_text(self):
        for payload in [{}, {"fact": ""}, {"fact": None}]:
            with patch("external_tools.get_json", return_value=payload):
                with self.assertRaises(ValueError): cat_fact()

    def test_weather_keeps_units_city_and_time(self):
        with patch("external_tools.get_json", side_effect=[
            {"results": [{"name": "Москва", "country": "Россия", "latitude": 55.75, "longitude": 37.62}]},
            {"timezone": "Europe/Moscow", "current": {"time": "2026-11-03T10:00", "temperature_2m": 2.5},
             "current_units": {"temperature_2m": "°C"}}]):
            result = weather("Москва")
            self.assertEqual(result["city"], "Москва")
            self.assertEqual(result["units"]["temperature_2m"], "°C")
            self.assertEqual(result["current"]["time"], "2026-11-03T10:00")

    def test_city_not_found(self):
        with patch("external_tools.get_json", return_value={"results": []}):
            self.assertIn("error", weather("unknown"))

    def test_invalid_image_is_not_sent_to_model(self):
        with patch("external_tools.get_json", return_value={"status": "success", "message": "https://images.dog.ceo/dog.jpg"}), \
             patch("external_tools.get_public", return_value=(b"not an image", "image/jpeg", "https://images.dog.ceo/dog.jpg")):
            with self.assertRaises(ValueError): dog_image()

    def test_unknown_host_rejected(self):
        for url in ["http://dog.ceo/a", "https://localhost/a", "https://dog.ceo.evil.test/a"]:
            with self.assertRaises(ValueError): get_public(url, {"dog.ceo"})

    def test_api_failure_is_not_invented_weather(self):
        with patch("external_tools.get_json", side_effect=RuntimeError("HTTP 503")):
            with self.assertRaises(RuntimeError): weather("Москва")


if __name__ == "__main__":
    unittest.main()
