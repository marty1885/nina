---
name: weather
description: "Fetch weather forecasts using wttr.in"
requires:
  bins: ["curl"]
---

# Weather Lookup

## Quick Weather

```bash
# Current weather for a city
curl -s "wttr.in/London?format=3"

# Detailed forecast
curl -s "wttr.in/London"

# Compact one-line
curl -s "wttr.in/London?format=%l:+%c+%t+%w+%h"
```

## Format Options

```bash
# Custom format placeholders:
# %l - location, %c - condition icon, %C - condition text
# %t - temperature, %f - feels like, %w - wind, %h - humidity
# %p - precipitation, %P - pressure, %D - dawn, %S - sunrise
# %z - zenith, %s - sunset, %d - dusk, %m - moon phase

# One-liner with key info
curl -s "wttr.in/Tokyo?format=%l:+%C+%t+(feels+%f)+wind+%w"

# JSON output for programmatic use
curl -s "wttr.in/Berlin?format=j1"
```

## Location Formats

```bash
# City name
curl -s "wttr.in/Paris"

# Airport code
curl -s "wttr.in/JFK"

# Coordinates
curl -s "wttr.in/48.8566,2.3522"

# IP-based (auto-detect)
curl -s "wttr.in"
```

## Tips

- Always use `-s` (silent) to suppress curl's progress bar.
- Use `format=3` for the most concise output suitable for chat responses.
- For multi-day forecasts, the default output (no format parameter) shows 3 days.
- Add `?lang=XX` for localized output (e.g., `?lang=de` for German).
