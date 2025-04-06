import requests

def get_location():
    """Fetch user's location using IP"""
    response = requests.get("http://ip-api.com/json/")
    return response.json()

def get_weather(city):
    """Fetch weather data from wttr.in (no API key needed)"""
    url = f"https://wttr.in/{city}?format=%C+%t"
    response = requests.get(url)
    return response.text.strip()

if __name__ == "__main__":
    location = get_location()
    city = location.get("city", "London")  # Default to London if location fails
    weather = get_weather(city)
    
    print(f"📍 Location: {city}, {location.get('country')}")
    print(f"🌤️ Weather: {weather}")
