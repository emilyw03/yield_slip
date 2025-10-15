# sandbox.py
# for testing

import os
import matplotlib

# Get the cache directory
cache_dir = matplotlib.get_cachedir()
print("Matplotlib cache:", cache_dir)

# Delete the font cache file
font_cache = os.path.join(cache_dir, 'fontlist-v330.json')  # Adjust version if needed

if os.path.exists(font_cache):
    os.remove(font_cache)
    print("Font cache removed.")
else:
    print("Font cache file not found.")



