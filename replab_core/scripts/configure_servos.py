from arbotix_python.arbotix import ArbotiX

SERVO_IDS = [1, 2, 3, 4, 5, 6]

arbotix = ArbotiX('/dev/ttyUSB0')
err = arbotix.syncWrite(14, [[servo_id, 255, 1] for servo_id in SERVO_IDS])

if err != -1:
    print('Successfully reconfigured servos')
else:
    print('Error writing to servos -- make sure start.sh is not running in the background')
