/*
Copyright (c) 2012, The Smith-Kettlewell Eye Research Institute
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of the The Smith-Kettlewell Eye Research Institute nor
      the names of its contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE SMITH-KETTLEWELL EYE RESEARCH INSTITUTE BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
package ski.blade;

import java.util.ArrayList;
import java.util.List;

import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.util.Log;

public class SensorHandler {
	/** Sensor manager instance */
	private SensorManager sensorManager_;
	/** Sensor types */
	public static final int SENSOR_LIGHT = Sensor.TYPE_LIGHT;
	public static final int SENSOR_ACCELEROMETER = Sensor.TYPE_ACCELEROMETER;
	public static final int SENSOR_PROXIMITY = Sensor.TYPE_PROXIMITY;
	public static final int SENSOR_TEMPERATURE = Sensor.TYPE_TEMPERATURE;
	public static final int SENSOR_ORIENTATION = Sensor.TYPE_ORIENTATION;	//Is used differently than the deprecated version
	public static final int SENSOR_COMPASS = Sensor.TYPE_MAGNETIC_FIELD;
	public static final int SENSOR_GYROSCOPE = Sensor.TYPE_GYROSCOPE;
	/** Sensor delays */
	public static final int RATE_LOW = SensorManager.SENSOR_DELAY_UI;
	public static final int RATE_MEDIUM = SensorManager.SENSOR_DELAY_GAME;
	public static final int RATE_HIGH = SensorManager.SENSOR_DELAY_FASTEST;
	private static final int RATE_NONE = -1;

	/** List of available sensors */
	private List<SensorWrapper> availableSensors_ = new ArrayList<SensorWrapper>();
	
	/** Interface that receives callbacks from sensors that are activated */
	private SensorCallback sensorListener_ = new SensorCallback();
	/** Callbacks that will be generated */
	private Callback callback_;
	
	private final static String TAG = "BLaDE SensorHandler";
	/**
	 * Constructor
	 * @param sensorMng android's sensor manager instance
	 */
	public SensorHandler(SensorManager sensorMng, Callback callback) {
		//Sensor manager
		if (sensorMng == null) {
			Log.e(TAG, "Cannot access sensors on device");
			return;
		}
		sensorManager_ = sensorMng;
		callback_ = callback;
		//Populate sensors:
		List<Sensor> detectedSensors = sensorManager_.getSensorList(Sensor.TYPE_ALL);
		for (Sensor sensor: detectedSensors)
			availableSensors_.add(new SensorWrapper(sensor)); //all sensors are initially off
	}
	
	/**
	 * Finds the wrapped sensor of a given type
	 * @param sensorType type of sensor to look for
	 * @return wrapper for this sensor type
	 */
	private SensorWrapper findWrapper(int sensorType) {
		for (SensorWrapper sensor: availableSensors_) {
			if (sensor.type() == sensorType)
				return sensor;
		}
		Log.d(TAG, "No sensor of this type available:" + sensorType);
		return null;
	}
	
	/**
	 * Finds the wrapped sensor for the given sensor
	 * @param aSensor sensor to find the wrapper for
	 * @return wrapper for this sensor
	 */
	private SensorWrapper findWrapper(Sensor aSensor) {
		return findWrapper(aSensor.getType());
	}
	
	/**
	 * Checks whether sensor is available
	 * @param sensorType type of sensor to look for
	 * @return true if the device has the desired type of sensor
	 */
	public boolean isSensorAvailable(int sensorType) {
		if (sensorType == SENSOR_ORIENTATION)
			return ( isSensorAvailable(SENSOR_ACCELEROMETER) && isSensorAvailable(SENSOR_COMPASS) );
		return (findWrapper(sensorType) != null);
	}
	
	/**
	 * Activates a given sensor and starts receiving callbacks
	 * @param sensorType sensor type to activate
	 * @param senseRate rate to receive callbacks
	 * @return true if a sensor of the given type is found and activated. False if no such sensor exists
	 */
	public boolean activateSensor(int sensorType, int senseRate) {
		if (sensorType == SENSOR_ORIENTATION)
			return ( activateSensor(SENSOR_ACCELEROMETER, senseRate) && activateSensor(SENSOR_COMPASS, senseRate) );
		SensorWrapper sensor = findWrapper(sensorType);
		if (sensor != null)
		{
			sensor.activate(senseRate);
			return true;
		}
		else
		{
			Log.e(TAG, "Sensor not found of type: " + sensorType);
			return false;
		}
	}
	
	/**
	 * Deactivates a sensor of given type
	 * @param sensorType type of sensor to deactivate
	 * @return true if sensor is deactivated
	 */
	public boolean deactivateSensor(int sensorType) {
		if (sensorType == SENSOR_ORIENTATION)
			return ( deactivateSensor(SENSOR_ACCELEROMETER) && deactivateSensor(SENSOR_COMPASS) );
		SensorWrapper sensor = findWrapper(sensorType);
		if (sensor != null)
		{
			if (sensor.isActive()) {
				sensor.deactivate();
				return true;
			}
			else {
				Log.d(TAG, "Sensor is not active: " + sensorType);
				return false;
			}
		}
		else
		{
			Log.e(TAG, "Sensor not found of type: " + sensorType);
			return false;
		}

	}

	/**
	 * Returns true if a sensor is active
	 * @param sensorType sensor type to check
	 * @return true if the sensor is active
	 */
	public boolean isSensorActive(int sensorType) {
		if (sensorType == SENSOR_ORIENTATION)
			return ( isSensorActive(SENSOR_ACCELEROMETER) && isSensorActive(SENSOR_COMPASS) );
		SensorWrapper sensor = findWrapper(sensorType);
		if (sensor != null)
			return sensor.isActive();
		else
		{
			Log.e(TAG, "Sensor not found of type: " + sensorType);
			return false;
		}
	}
	
	/**
	 * Releases all sensors and clears the resources used bu the sensor handler
	 */
	public void release() {
		//Deactivate all sensors
		for (SensorWrapper sensor: availableSensors_)
			sensor.deactivate();
		availableSensors_.clear();
		availableSensors_ = null;
		sensorListener_ = null;
		sensorManager_ = null;
		callback_ = null;
	}
	
	/**
	 * Cleanup
	 */
	@Override
	protected void finalize() {
		release();
	}
	
	/**
	 * @class Callback listener for the active sensors
	 * @author kamyon
	 *
	 */
	private class SensorCallback implements SensorEventListener {
		/** Rotation matrix */
		float [] rotMat = new float[16];
		/** Inclination matrix */
		float [] inclMat = new float[16];
		
		@Override
		public void onAccuracyChanged(Sensor aSensor, int accuracy) {
			// We only report this event if sensor needs calibration
			if (accuracy <= SensorManager.SENSOR_STATUS_ACCURACY_LOW)
			{
				for (SensorWrapper sensor: availableSensors_) {
					if (sensor.type() == aSensor.getType()) {
						Log.d(TAG, "Calibration needed for: " + aSensor.getType());
						callback_.onCalibrationNeeded(sensor.type());
					}
				}
			}
			else
				Log.v(TAG, "Accuracy of sensor changed: " + aSensor.getType());
		}

		@Override
		public void onSensorChanged(SensorEvent event) {
			//If this is an orientation related sensor, we call both the onSensorUpdated and the onOrientationChanged callbacks
			SensorWrapper sensor = findWrapper(event.sensor); //if we are receiving callbacks, this sensor cannot be null.
			Log.v(TAG, "New data received for sensor:" + sensor.type());
			if (sensor != null)
				sensor.setData(event);
			//Update sensor data
			if (callback_ != null)
				callback_.onSensorDataReceived(sensor.type(), sensor.getData());
			//Determine if this is an orientation change, and calculate new orientation if so
			if (sensor.type() == SENSOR_ACCELEROMETER) {
				SensorData accData = sensor.getData(), orientationData = new SensorData();
				//see if there is compass data as well
				SensorWrapper compass = findWrapper(SENSOR_COMPASS);
				if (compass != null)
				{
					SensorData compassData = compass.getData();
					if (compassData != null) {
						orientationData.accuracy = Math.min(compassData.accuracy, accData.accuracy);
						orientationData.timestamp = accData.timestamp;
						orientationData.data = calculateOrientation(compassData.data, accData.data);
						if (callback_ != null)
							callback_.onSensorDataReceived(SENSOR_ORIENTATION, orientationData);
					}
				}
			}
			else if (sensor.type() == SENSOR_COMPASS) {
				SensorData compassData = sensor.getData(), orientationData = new SensorData();
				//See if there is accelerometer data as well
				SensorWrapper accelerometer = findWrapper(SENSOR_ACCELEROMETER);
				if (accelerometer != null)
				{
					SensorData accData = accelerometer.getData();
					if (accData != null) {
						orientationData.accuracy = Math.min(compassData.accuracy, accData.accuracy);
						orientationData.timestamp = compassData.timestamp;
						orientationData.data = calculateOrientation(compassData.data, accData.data);
						if (callback_ != null)
							callback_.onSensorDataReceived(SENSOR_ORIENTATION, orientationData);
					}
				}
			}
		}
		
		private float[] calculateOrientation(float[] compassData, float[] accData) {
			if ( (compassData == null) || (accData == null) )
			{
				Log.e(TAG, "CalculateOrientation should not be called with no data");
				return null;
			}
			float[] data = new float[4]; //data is returned as pitch/yaw/roll/inclination in radians
			/* From: https://developer.android.com/reference/android/hardware/SensorManager.html
			data[0]: azimuth: rotation around the Z axis.
			data[1]: pitch, rotation around the X axis.
			data[2]: roll, rotation around the Y axis.
			data[3]: inclination, the geomagnetic inclination angle in radians
			The reference coordinate-system used is different from the world coordinate-system defined for the rotation matrix:
			X is defined as the vector product Y.Z (It is tangential to the ground at the device's current location and roughly points West).
			Y is tangential to the ground at the device's current location and points towards the magnetic North Pole.
			Z points towards the center of the Earth and is perpendicular to the ground.
			*/
			//Calculate rotation and inclination matrices
			SensorManager.getRotationMatrix(rotMat, inclMat, accData, compassData);
			//Calculate orientation: pitch/yaw/roll
			SensorManager.getOrientation(rotMat, data);
			//Calculate inclination
			data[3] = SensorManager.getInclination(inclMat);
			return data;
		}
	}
	
	/**
	 * @class Sensor data encapsulation
	 * @author kamyon
	 *
	 */
	public class SensorData {
		/** Current accuracy of the sensor */
		public int accuracy = SensorManager.SENSOR_STATUS_UNRELIABLE;
		/** Timestamp of last data */
		public long timestamp = 0;
		/** Last data received */
		public float[] data = null;
	}
	
	/**
	 * @class Keeps data such as accuracy, etc with the available sensors  
	 * @author kamyon
	 *
	 */
	private class SensorWrapper {
		/** Sensor that we are observing */
		private Sensor sensor;
		/** Callback rate: rate_none implies sensor is not active*/
		private int rate = RATE_NONE;
		/** Data corresponding to this sensor */
		private SensorData data = null;
		
		/**
		 * Constructor
		 * @param aSensor sensor to wrap around
		 */
		public SensorWrapper(Sensor aSensor) {
			sensor = aSensor;
		}
		
		/**
		 * Returns the type of the sensor
		 */
		public int type() {
			return sensor.getType();
		}

		/**
		 * Activates the given sensor, and starts receiving callbacks.
		 * @param rate how often to expect callbacks.
		 */
		public void activate(int senseRate) {
			if (!isActive() && (senseRate != RATE_NONE) ) {
				sensorManager_.registerListener(sensorListener_, sensor, senseRate);
				rate = senseRate;
				data = new SensorData();
			}
			else
				Log.d(TAG, "Sensor already started: " + type());
		}
		
		/**
		 * Stops the sensor
		 */
		public void deactivate() {
			if (isActive()) {
				sensorManager_.unregisterListener(sensorListener_, sensor);
				rate = RATE_NONE;
				data = null;
			}
			else
				Log.d(TAG, "Sensor already deactivated: " + type());
		}
				
		/**
		 * Returns whether sensor is active
		 * @return true if sensor is currently providing callbacks
		 */
		public boolean isActive() {
			return (rate != RATE_NONE);
		}
		
		/**
		 * Returns the last data associated with this sensor
		 * @return last obtained data associated with this sensor
		 */
		public SensorData getData() {
			return data;
		}

		public void setData(SensorEvent eventData) {
			data.accuracy = eventData.accuracy;
			data.timestamp = eventData.timestamp;
			data.data = eventData.values;
		}
		
		/**
		 * Cleanup if we lose reference without deactivating
		 */
		@Override
		protected void finalize() {
			deactivate();
		}
	}
	
	/**
	 * @class Callback interface when a sensor value has changed
	 */
	public interface Callback {
		/**
		 * Callback when a sensor needs to be calibrated
		 * @param sensorType type of sensor that needs calibration
		 */
		public void onCalibrationNeeded(int sensorType);
		
		/**
		 * Called when sensor data is updated
		 * @param sensorType type of sensor data is received from
		 * @param sensorData data received from sensor
		 */
		public void onSensorDataReceived(int sensorType, SensorData sensorData);
	}
}
