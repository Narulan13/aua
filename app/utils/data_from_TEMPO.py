import netCDF4 as nc
import pandas as pd
import numpy as np

# Open your TEMPO file
filename = "TEMPO_L2_NO2_20230701T1200Z.nc"
ds = nc.Dataset(filename, mode='r')

# Inspect available variables
print("Variables in dataset:")
for var in ds.variables.keys():
    print(var)

# Example: Collect common coordinate + science variables
lat = ds.variables['latitude'][:]
lon = ds.variables['longitude'][:]
time = ds.variables['time'][:] if "time" in ds.variables else None

# Create dictionary for all science variables
data = {
    "latitude": lat.flatten(),
    "longitude": lon.flatten()
}

if time is not None:
    data["time"] = np.repeat(time, lat.size // time.size)

# Loop over all variables (skip coords)
for var in ds.variables.keys():
    if var not in ["latitude", "longitude", "time"]:
        try:
            vals = ds.variables[var][:]
            data[var] = vals.flatten()
        except:
            print(f"Skipping {var} (not exportable)")

# Build DataFrame
df = pd.DataFrame(data)

# Drop NaNs
df = df.dropna()

# Save as CSV
df.to_csv("tempo_full_dataset.csv", index=False)
print("✅ Saved all variables to tempo_full_dataset.csv")
