from breezyslam.vehicles import WheeledVehicle


class pioner(WheeledVehicle):

    def __init__(self):
        WheeledVehicle.__init__(self, 195 / 2, 381 / 2)

        self.ticks_per_cycle = 2000

    def __str__(self):
        return '<%s ticks_per_cycle=%d>' % (WheeledVehicle.__str__(self), self.ticks_per_cycle)

    def computePoseChange(self, timestamp, leftWheelOdometry, rightWheelOdometry):
        return WheeledVehicle.computePoseChange(self, timestamp, leftWheelOdometry, rightWheelOdometry)

    def extractOdometry(self, timestamp, leftWheel, rightWheel):
        # Convert microseconds to seconds, ticks to angles
        return timestamp, \
               self._rad_to_degrees(leftWheel), \
               self._rad_to_degrees(rightWheel)

    def odometryStr(self, odometry):
        return '<timestamp=%d usec leftWheelTicks=%d rightWheelTicks=%d>' % \
               (odometry[0], odometry[1], odometry[2])

    def _ticks_to_degrees(self, ticks):

        return ticks * (180. / self.ticks_per_cycle)

    def _rad_to_degrees(self, rad):
        return rad * 57.2958

