import os
import io
import json
import zipfile
import requests
import pandas as pd
import holidays
from geopy.geocoders import Nominatim
from pathlib import Path

DATA_DIR = Path(__file__).parent

def download_taxi_zones():
    print("Downloading NYC Taxi Zones shapefile...")
    url = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zones.zip"
    r = requests.get(url)
    extract_dir = DATA_DIR / "taxi_zones"
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        z.extractall(extract_dir)
    print(f"  -> Saved to {extract_dir}")

def download_zone_lookup():
    print("Downloading NYC Taxi Zone Lookup CSV...")
    url = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv"
    df = pd.read_csv(url)
    out_path = DATA_DIR / "taxi_zone_lookup.csv"
    df.to_csv(out_path, index=False)
    print(f"  -> Saved to {out_path}")

def download_weather():
    print("Downloading Historical Weather Data (Open-Meteo)...")
    url = "https://archive-api.open-meteo.com/v1/archive?latitude=40.7831&longitude=-73.9712&start_date=2023-01-01&end_date=2024-12-31&hourly=temperature_2m,precipitation,wind_speed_10m&timezone=America%2FNew_York"
    r = requests.get(url).json()
    
    df = pd.DataFrame({
        "time": pd.to_datetime(r["hourly"]["time"]),
        "temp": r["hourly"]["temperature_2m"],
        "precip": r["hourly"]["precipitation"],
        "wind": r["hourly"]["wind_speed_10m"]
    })
    
    df.ffill(inplace=True)
    df.fillna(0, inplace=True)
    
    out_path = DATA_DIR / "weather_2023_2024.csv"
    df.to_csv(out_path, index=False)
    print(f"  -> Saved to {out_path}")

def generate_hubs():
    print("Geocoding NYC Hubs...")
    geolocator = Nominatim(user_agent="eta_challenge_geocoder")
    queries = {
        "jfk": "JFK Airport, New York",
        "lga": "LaGuardia Airport, New York",
        "ewr": "Newark Liberty International Airport",
        "tsq": "Times Square, Manhattan, New York"
    }
    hubs = {}
    for key, query in queries.items():
        try:
            loc = geolocator.geocode(query)
            if loc:
                hubs[key] = [loc.latitude, loc.longitude]
            else:
                raise ValueError("Not found")
        except Exception:
            print(f"  Warning: Could not geocode {query}. Using fallback.")
            fallbacks = {"jfk": [40.6413, -73.7781], "lga": [40.7769, -73.8740], "ewr": [40.6895, -74.1745], "tsq":[40.7580, -73.9855]}
            hubs[key] = fallbacks[key]
            
    out_path = DATA_DIR / "hubs.json"
    with open(out_path, "w") as f:
        json.dump(hubs, f)
    print(f"  -> Saved to {out_path}")

def generate_holidays():
    print("Generating NY State Holidays...")
    ny_holidays_obj = holidays.US(state='NY', years=[2023, 2024, 2025])
    ny_holidays = list({date.strftime('%Y-%m-%d') for date in ny_holidays_obj.keys()})
    
    out_path = DATA_DIR / "holidays.json"
    with open(out_path, "w") as f:
        json.dump(ny_holidays, f)
    print(f"  -> Saved to {out_path}")

if __name__ == "__main__":
    download_taxi_zones()
    download_zone_lookup()
    download_weather()
    generate_hubs()
    generate_holidays()
    print("\nAll external data downloaded successfully!")