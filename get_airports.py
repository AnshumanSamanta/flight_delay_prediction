import pandas as pd 

airports=pd.read_csv('D:/codding/flight_delay_pred/airports.csv')

# print(len(airports))
# print(airports['IATA_CODE'])
list_of_airports=dict()
for i,j in zip(airports['IATA_CODE'],airports['AIRPORT']):
    list_of_airports[i]=j
# print(list_of_airports)
print(list(airports['IATA_CODE']))


