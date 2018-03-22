import requests

url='http://data.fmi.fi/fmi-apikey/9fdf9977-5d8f-4a1f-9800-d80a007579c9/timeseries?format=ascii&separator=,&precision=double&producer=opendata&timeformat=xml&latlons=64.537118,25.006783,61.291923,22.498185&timestep=10&starttime=2018-01-01T00:00:00&endtime=2018-01-02T00:00:00&param=time,name,wmo,lat,lon,mean(mean_t(pressure:60:0))%20as%20pressure,max(max_t(dewpoint:60:0))%20as%20max_temp,min(min_t(dewpoint:60:0))%20as%20min_temp,mean(mean_t(dewpoint:60:0))%20as%20mean_temp,max(sum_t(precipitation1h:180:0))%20as%20precipitation3h'

r = requests.get(url, allow_redirects=True)
open('/home/daniel/bigdata/trains/text.csv', 'wb').write(r.content)
df=spark.read.csv('/home/daniel/bigdata/trains/text.csv')

df=df.toDF("timestamp", "name", "wmo", "lat", "lon", "pressure", "max_temp", "min_temp", "mean_temp", "precipitation")
df.show()
