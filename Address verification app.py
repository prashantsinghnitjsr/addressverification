import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import numpy as np
from geopy.distance import geodesic
import requests
import folium
import streamlit as st
from streamlit_folium import st_folium

# Function to check if a location is within 15 meters of a road using Google Maps Roads API
def is_near_road_google(lat, lon, max_distance=15):
    try:
        api_key = 'AIzaSyAPYdUAcC5g8InALfbUNc15dVoOC3ShYEE'  # Replace with your Google Maps API key
        url = f"https://roads.googleapis.com/v1/nearestRoads?points={lat},{lon}&key={api_key}"
        response = requests.get(url)
        result = response.json()
        
        if 'snappedPoints' in result:
            road_location = result['snappedPoints'][0]['location']
            road_lat = road_location['latitude']
            road_lon = road_location['longitude']
            distance = geodesic((lat, lon), (road_lat, road_lon)).meters
            return distance <= max_distance
        else:
            return False  # No road found within the search area
    except Exception as e:
        print(f"Error checking location ({lat}, {lon}): {e}")
        return False

st.title("Outlier Detection in Geographic Data")

# File uploader for the CSV file
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Step 1: Load the CSV file
    df = pd.read_csv(uploaded_file, encoding='ISO-8859-1')

    # Step 2: Z-Score method for outliers
    scaler = StandardScaler()
    df[['lat_zscore', 'long_zscore']] = scaler.fit_transform(df[['Latitude', 'Longitude']])

    # Flag points with Z-score > 3 as outliers
    df['z_score_outlier'] = np.abs(df[['lat_zscore', 'long_zscore']]).max(axis=1) > 3

    # Step 3: DBSCAN method for outliers
    clustering = DBSCAN(eps=0.1, min_samples=3).fit(df[['Latitude', 'Longitude']])
    df['dbscan_outlier'] = clustering.labels_ == -1  # Flag points with label -1 as outliers

    # Step 4: Check if locations are within 15 meters of a road using Google Maps Roads API
    df['near_road'] = df.apply(lambda row: is_near_road_google(row['Latitude'], row['Longitude']), axis=1)

    # Treat locations more than 15 meters away from a road as outliers
    df['road_outlier'] = ~df['near_road']

    # Step 5: Combine all outliers
    df['is_outlier'] = df['z_score_outlier'] | df['dbscan_outlier'] | df['road_outlier']

    # Display the results
    st.write("Combined Outliers:")
    st.write(df[df['is_outlier']])

    # Save the results to a new CSV file
    output_file_path = 'outliers_with_road_proximity.csv'
    df.to_csv(output_file_path, index=False)
    st.write(f"\nOutliers saved to {output_file_path}")

    # Step 6: Create an interactive map with Folium
    map_center = [df['Latitude'].mean(), df['Longitude'].mean()]
    mymap = folium.Map(location=map_center, zoom_start=12)

    # Add points to the map
    for idx, row in df.iterrows():
        if row['is_outlier']:
            color = 'red'
        else:
            color = 'blue'
        folium.CircleMarker(
            location=(row['Latitude'], row['Longitude']),
            radius=5,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.6
        ).add_to(mymap)

    # Display the map in Streamlit
    st_folium(mymap, width=700, height=500)
