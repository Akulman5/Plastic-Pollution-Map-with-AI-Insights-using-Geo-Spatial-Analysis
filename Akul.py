import pandas as pd
import folium
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


plastic_waste_path = 'Plastic Waste Around the World.csv'
data = pd.read_csv(plastic_waste_path)


coordinates_path = 'world_country_and_usa_states_latitude_and_longitude_values.csv'
coords = pd.read_csv(coordinates_path)


print("Plastic Waste Dataset:")
print(data.head())
print("\nCountry Coordinates Dataset:")
print(coords.head())


coords = coords[['country', 'latitude', 'longitude']] 
coords.rename(columns={'country': 'Country', 'latitude': 'Latitude', 'longitude': 'Longitude'}, inplace=True)


data = data.merge(coords, how='left', on='Country')


data = data.dropna(subset=['Latitude', 'Longitude'])


numeric_columns = ['Total_Plastic_Waste_MT', 'Recycling_Rate', 'Per_Capita_Waste_KG']
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data[numeric_columns])


kmeans = KMeans(n_clusters=5, random_state=42)
data['Cluster'] = kmeans.fit_predict(data_scaled)


def generate_ai_solution(row):
    country = row['Country']
    total_waste = row['Total_Plastic_Waste_MT']
    main_sources = row['Main_Sources']
    recycling_rate = row['Recycling_Rate']
    per_capita_waste = row['Per_Capita_Waste_KG']
    coastal_risk = row['Coastal_Waste_Risk']
    cluster = row['Cluster']
    
    
    if cluster == 0:
        solution = (
            f"In {country}, focus on increasing the recycling rate of {recycling_rate:.1f}% by improving waste management "
            f"infrastructure and raising awareness about reducing waste from {main_sources}."
        )
    elif cluster == 1:
        solution = (
            f"In {country}, address the high per capita plastic waste of {per_capita_waste:.1f} kg by implementing "
            f"policies targeting {main_sources} and encouraging sustainable practices."
        )
    elif cluster == 2:
        solution = (
            f"In {country}, focus on minimizing coastal waste risk ({coastal_risk}) by improving plastic collection "
            f"and recycling systems for {main_sources}. Reduce the total waste of {total_waste:.2f} MT."
        )
    elif cluster == 3:
        solution = (
            f"In {country}, leverage the existing recycling rate of {recycling_rate:.1f}% to promote circular economy "
            f"models and reduce waste from {main_sources}. Address the coastal waste risk: {coastal_risk}."
        )
    elif cluster == 4:
        solution = (
            f"In {country}, implement region-specific solutions to address the {main_sources} sources and reduce the "
            f"per capita waste of {per_capita_waste:.1f} kg. Focus on community-based recycling initiatives."
        )
    else:
        solution = f"Develop targeted solutions for {country} to reduce its plastic waste impact."
    
    return solution

data['AI Suggested Solution'] = data.apply(generate_ai_solution, axis=1)


m = folium.Map(location=[20, 0], zoom_start=2)


for _, row in data.iterrows():
    folium.Marker(
        location=[row['Latitude'], row['Longitude']],
        popup=(
            f"Country: {row['Country']}<br>"
            f"Total Plastic Waste: {row['Total_Plastic_Waste_MT']} MT<br>"
            f"Main Sources: {row['Main_Sources']}<br>"
            f"Recycling Rate: {row['Recycling_Rate']}%<br>"
            f"Per Capita Waste: {row['Per_Capita_Waste_KG']} kg<br>"
            f"Coastal Waste Risk: {row['Coastal_Waste_Risk']}<br><br>"
            f"AI Suggested Solution: {row['AI Suggested Solution']}"
        ),
        icon=folium.Icon(color='blue' if row['Cluster'] == 0 else 'red' if row['Cluster'] == 1 else 'green')
    ).add_to(m)


m.save('plastic_waste_with_insights_map.html')


data.to_csv('plastic_waste_with_insights.csv', index=False)

print("Interactive map saved as 'plastic_waste_with_insights_map.html'")
print("Updated dataset with AI-suggested insights saved as 'plastic_waste_with_insights.csv'")
