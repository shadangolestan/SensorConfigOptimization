class Sensor:
    sensor_type = ""
    sensor_id = ""
    x = float()
    y = float()
    z = float()
    r = float()
    room = str()

    def __str__(self):
        return 'type: %s,\n id: %s,\n x: %s,\n y: %s\n'%(self.sensor_type, self.sensor_id, self.x, self.y)
        
    def Initialize(self, x, y, t, sensor_id):
        self.x = x * 100
        self.y = y * 100
        self.sensor_id = sensor_id
        self.sensor_type = t
        return 0
    
    def GaussianFiredSensor(self, simulated_localization, real_localization):
        import math
        sensor_location = [float(self.x)/10, float(self.y)/10]
        cov = [[self.sensing_area, 0],[0, self.sensing_area]]        
        prob = multivariate_normal(simulated_localization, cov)
        probCdf = prob.cdf(sensor_location)
        return probCdf
    
class MotionSensorBinary(Sensor):
    def __init__(self, x, y, radius, room, sensor_id):
        Sensor.Initialize(self, x, y, "motion_sensor", sensor_id)
        self.sensing_area = radius * 100
        self.room = room
        
class BeaconSensor(Sensor):
    def __init__(self, x, y, radius, room, sensor_id):
        Sensor.Initialize(self, x, y, "beacon_sensor", sensor_id)
        self.sensing_area = radius * 100
        self.measuredPower = -69
        # self.RSSI = self.sensing_area
        self.N = 2 #Constant related with the air
        self.room = room 
        # self.accelerometer = str(element.find('accelerometer').text)
        
    # https://iotandelectronics.wordpress.com/2016/10/07/how-to-calculate-distance-from-the-rssi-value-of-the-ble-beacon/
    def rssiToMeters(self):
        u = (float(self.measuredPower) - float(self.RSSI))/(10 * float(self.N))
        dist = 10 ** u
        
        return dist 
    
    def MetersToRSSI(self, dist):
        import math 
        import numpy as np
        
        rawValue = self.measuredPower - 10*self.N*math.log(dist, 10)
        noise = 0 # np.random.normal(0, dist, 1)
        
        return rawValue + noise

class InteractiveSensitive(Sensor):
    def __init__(self, x, y, radius, room, sensor_id, sensitivity):
        Sensor.Initialize(self, x, y, "IS", sensor_id)
        self.sensing_area = radius * 100
        self.sensitivity = sensitivity
        self.room = room