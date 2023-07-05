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

import java.util.concurrent.locks.Condition;
import java.util.concurrent.locks.ReentrantLock;

import android.app.Activity;
import android.hardware.Sensor;
import android.hardware.SensorManager;
import android.os.Handler;
import android.os.Message;
import android.os.Vibrator;
import android.util.Log;
import android.view.SurfaceView;

public class FrameProcessor {
	private ReentrantLock lock = new ReentrantLock(false);
	private Condition newFrameAvailable = lock.newCondition();
	private volatile boolean isFinished = false;
	private boolean isRunning = false;
	private Thread backgroundProc;
	private Barcode bc;
	private SoundFeedback feedback;
	private byte[] buffer_;
	private int width_, height_;

	private int decodingAttempts;
	private CameraHandler camera_ = null;
	private CameraCallbacks cameraCallbacks_ = new CameraCallbacks();
	private SensorCallbacks sensorCallbacks_ = new SensorCallbacks();
	private SensorHandler sensorHandler_ = null;
	private ProcessorOptions options_ = null;
	private OnBarcodeFound callback_;
	private OverlayView mOverlaySV;
	
	private Vibrator vibrator_;
	
	private final String TAG = "BLaDE FrameProcessor";

	//Status
	static public class Status {
		static final int NOT_FOUND = 0;
		static final int DETECTED = 1;
		static final int DECODING_FAILED = 2;
		static final int DECODED = 3;
		static final int ERROR = -1;
		static final int ENDED = -2;
	}
	int status = Status.NOT_FOUND;

	/**
	 * Frame processor class
	 * @param w width of the frames that will be received
	 * @param h height of the frames that will be received
	 * @param aCam camera
	 */
	FrameProcessor(int w, int h, OnBarcodeFound callback, Activity act, boolean locationOnly) 
	{
		//Get the preview surfaces and open camera
		camera_ = new CameraHandler(w, h, (SurfaceView) act.findViewById(R.id.preview), cameraCallbacks_);
		callback_ = callback;

		//Set up image processing thread
		backgroundProc = new Thread(new NativeProcessor(locationOnly));
		backgroundProc.setPriority(Thread.MAX_PRIORITY);
		//Audio feedback
		feedback = new SoundFeedback();

		//Overlay surface
		mOverlaySV = (OverlayView) act.findViewById(R.id.overlay);
		mOverlaySV.previewWidth = w;
		mOverlaySV.previewHeight = h;
		
		//Sensors
		sensorHandler_ = new SensorHandler((SensorManager) act.getSystemService(Activity.SENSOR_SERVICE), sensorCallbacks_);
		//Vibrator
		vibrator_ = (Vibrator) act.getSystemService(Activity.VIBRATOR_SERVICE);

		//Options
		setOptions(new ProcessorOptions());
	}

	FrameProcessor(int w, int h, OnBarcodeFound callback, Activity act, boolean locationOnly)
	{
		this(w, h, callback, act, false);
	}

	/**
	 * @class Processor options
	 * @author kamyon
	 * Created with default options
	 */
	public class ProcessorOptions implements Cloneable{
		/** True if audio feedback is on */
		public boolean isAudioFeedbackOn = true;
		/** True if light sensing is on */
		public boolean isLightSensingOn = false;	//TODO: was causing problems when on - debug! see if it is getting too many callbacks
		/** True if tilt sensing is on */
		public boolean isTiltSensingOn = false;
		/** True if barcode decoding is on */
		public boolean isBarcodeDecodingOn = true;
		@Override
		public ProcessorOptions clone() {
			ProcessorOptions clonedOpts = null;
			try {
				clonedOpts = (ProcessorOptions) super.clone();
			} catch (CloneNotSupportedException e) {
				e.printStackTrace();
			}
			return clonedOpts;
		}
	}
	
	/**
	 * Retrieves the current options used by the frame processor
	 * @return current frame processing options
	 */
	public ProcessorOptions getOptions() {
		return options_.clone();
	}
	
	/**
	 * Sets options to be used by the frame processor
	 * @param options new options to set. These are not guaranteed, since they are based on availability of hardware.
	 */
	public void setOptions(ProcessorOptions options) {
		//Set the new options
		if (options_ == null) {
			Log.v(TAG, "Setting initial options");
			options_ = options;
			//Audio feedback:
			feedback.setFeedback(options_.isAudioFeedbackOn);
			//Light sensing
			if (options_.isLightSensingOn)
				options_.isLightSensingOn = startLightSensing();
			//Tilt sensing
			if (options_.isTiltSensingOn)
				options_.isTiltSensingOn = startTiltSensing();
		}
		else { //Change existing options
			Log.v(TAG, "Changing options");
			//Audio feedback
			if (options.isAudioFeedbackOn != options_.isAudioFeedbackOn) {
				feedback.setFeedback(options.isAudioFeedbackOn);
				options_.isAudioFeedbackOn = options.isAudioFeedbackOn;
			}
			//Light sensing
			if (options.isLightSensingOn != options_.isLightSensingOn) {
				if (options.isLightSensingOn) //off->on
					options_.isLightSensingOn = startLightSensing();
				else { //on->off
					stopLightSensing();
					options_.isLightSensingOn = false;  
				}
			}
			//Tilt sensing
			if (options.isTiltSensingOn != options_.isTiltSensingOn) {
				Log.d(TAG, "Setting tilt sensing option " + (options.isTiltSensingOn ? "on" : "off") );
				if (options.isTiltSensingOn) //off->on
					options_.isTiltSensingOn = startTiltSensing();
				else { //on->off
					stopTiltSensing();
					options_.isTiltSensingOn = false;
				}
			}
			//Barcode Decoding
			options_.isBarcodeDecodingOn = options.isBarcodeDecodingOn;
		}
	}
	
	/**
	 * Starts sensing ambient light sensors, and turns on flash if light is low and flash is available
	 * @return true if light sensing has started, false if light sensor or camera flash are not available.
	 */
	private boolean startLightSensing() {
		//Light sensor
		if ( sensorHandler_.isSensorAvailable(SensorHandler.SENSOR_LIGHT) && camera_.hasFlash() )
		{
			//Activate sensor
			sensorHandler_.activateSensor(SensorHandler.SENSOR_LIGHT, SensorHandler.RATE_LOW);
			Log.v(TAG, "Starting light sensor");
			return true;
		}
		else {
			Log.d(TAG, "No light sensor available or camera flash unavailable");
			return false;
		}
	}

	/**
	 * Stops sensing ambient light
	 */
	private void stopLightSensing() {
		if (sensorHandler_.isSensorActive(SensorHandler.SENSOR_LIGHT)) {
			//Deactivate light sensor
			sensorHandler_.deactivateSensor(SensorHandler.SENSOR_LIGHT);
			//Deactivate flash
			if (camera_ != null)
				camera_.setFlashMode(CameraHandler.FM_OFF);
			Log.v(TAG, "Stopping light sensor");
		}
	}
	
	/**
	 * starts tilt sensing
	 * @return true if tilt sensing can be done (requires accelerometer
	 */
	private boolean startTiltSensing() {
		//Tilt sensor
		//TODO: add && vibrator_.hasVibrator() check for icecream sandwich version
		if ( sensorHandler_.isSensorAvailable(SensorHandler.SENSOR_ACCELEROMETER) ){
			//Activate sensor
			sensorHandler_.activateSensor(SensorHandler.SENSOR_ACCELEROMETER, SensorHandler.RATE_MEDIUM);
			Log.v(TAG, "Starting tilt sensor");
			return true;
		}
		else {
			Log.d(TAG, "No tilt sensor available");
			return false;
		}
	}
	
	/**
	 * Stops tilt sensing
	 */
	private void stopTiltSensing() {
		if (sensorHandler_.isSensorActive(SensorHandler.SENSOR_ACCELEROMETER)) {
			sensorHandler_.deactivateSensor(SensorHandler.SENSOR_ACCELEROMETER);
			vibrator_.cancel();
			Log.v(TAG, "Stopping tilt sensor");
		}
	}
	
	/**
	 * Starts the frame processor
	 */
	public void start() {
		if (!isRunning) {
			isRunning = true;
			if (camera_ != null)
			{
				if (camera_.isInitialized())
				{
					camera_.startPreview();
					//Autofocus
					camera_.attemptAutoFocus(true);
				}
			}
			else
			{
				Log.e(TAG, "Camera handler not initialized!");
				throw new RuntimeException("Cannot start frame processor before initializing camera handler!");
			}
			if (feedback != null)
				feedback.start();
		}
	}
	
	/** 
	 * Stops the frame processor
	 */
	public void stop() {
		isFinished = true;
		if (feedback != null)
			feedback.stop();
		if (camera_ != null)
			camera_.stopPreview();
		stopLightSensing();
		isRunning = false;
	}

	/**
	 * Pauses the frame processor
	 */
	public void pause() {
		if (feedback != null)
			feedback.pause();
		if (camera_ != null)
			camera_.stopPreview();
		stopLightSensing();
		isRunning = false;
	}
	
	/**
	 * Checks to see if the frame processor is currently processing a frame
	 * @return true if the processor is running, false if it is paused
	 */
	public boolean isActive() {
		return isRunning;
	}
	

	public void release() {
		stopLightSensing();
		stopTiltSensing();
    	if (feedback != null) {
    		feedback.release();
    		feedback = null;
    	}
    	if (camera_ != null) {
    		camera_.release();
    		camera_ = null;
    	}
    	isFinished = true;
		isRunning = false;
	}
	
	protected void finalize() {
		release();
	}

	/**
	 * @class Handles camera handler callbacks
	 * @author kamyon
	 *
	 */
	private class CameraCallbacks implements CameraHandler.Callback {
		@Override
		public void onCameraInitialized() {
			//start preview & processing
			start();
		}
		
		@Override
		public void onFrameReceived(byte[] buf, int w, int h) {
			//We only have a callback if the background processor has finished 
			//processing the previous frame and released the buffer;
			//Start processing if not already started
			buffer_ = buf;
			width_ = w;
			height_ = h;
			if (backgroundProc.isAlive())
			{
				try{
					//Signal that a new frame has arrived
					lock.lock();
					newFrameAvailable.signal();
				}
				catch (IllegalMonitorStateException e){
					Log.e(TAG, "FrameProcessor UI Thread does not have control of the lock");
					e.printStackTrace();
				}
				finally {
					lock.unlock();
				}
			}
			else
			{
				//Start processing thread
				try {
					backgroundProc.start();
				}
				catch (IllegalThreadStateException e) {
					Log.e(TAG, "Background process is already started");
					e.printStackTrace();
				}
			}
			//Attempt autofocus if not focused
			camera_.attemptAutoFocus(false);
		}
	
		@Override
		public void onCameraReleased() {
			//Implies that camera reference is no longer initialized - release the processor. 
			camera_ = null;
			release();
		}
	}
	
	/**
	 * @class Handles sensor callbacks
	 * @author kamyon
	 *
	 */
	private class SensorCallbacks implements SensorHandler.Callback{
		@Override
		public void onSensorDataReceived(int sensorType, SensorHandler.SensorData data) {
			switch (sensorType) {
			case Sensor.TYPE_LIGHT:
			   Log.i(TAG, "LightSensor Change :" + data.data[0]);
			   if (camera_.getFlashMode() != CameraHandler.FM_NOT_AVAILABLE)
			   {
				   if (data.data[0] < SensorManager.LIGHT_CLOUDY)
					   camera_.setFlashMode(CameraHandler.FM_TORCH);
				   else
					   camera_.setFlashMode(CameraHandler.FM_OFF);
			   }
			   break;
			case Sensor.TYPE_ACCELEROMETER:
				Log.i(TAG, "Accelerometer change: " + data.data[0] + "," + data.data[1] + "," + data.data[2]);
				//If off-horizontal, and tilt sensing is active, start vibrator
				float x = data.data[0], y = data.data[1], z = data.data[2];
				//double angleOffHorizontal= Math.atan2(Math.sqrt(y * y + z * z), x) / Math.PI * 180.0; //convert to degrees
				//double g = SensorManager.GRAVITY_EARTH;
				double angleOffHorizontal= Math.acos(z / android.util.FloatMath.sqrt(x * x + y * y + z * z)) / Math.PI * 180.0; //convert to degrees 
				Log.d(TAG, "Phone off horizontal by" + angleOffHorizontal);
				if (angleOffHorizontal > 10.0)//if not vertical by more than 10 degrees
					vibrator_.vibrate(30);
				break;
			default:
				Log.i(TAG, "Unhandled sensor:" + sensorType);
				break;
			}
		}
		
		@Override
		public void onCalibrationNeeded(int sensorType) {
			  Log.v(TAG, "Calibration needed for sensor type: " + sensorType);		
		}
	}
	
	/**
	 * @class Native processing thread encapsulation
	 * @author kamyon
	 *
	 */
	private class NativeProcessor implements Runnable {
		/** Message handler*/
		private MsgHandler msgHandler = new MsgHandler();
		private boolean locationOnly = false;
		
		/**
		 * Constructor
		 */
		public NativeProcessor() {
			bc = new Barcode();
		}

		public NativeProcessor( boolean locationOnly ) {
			this();
			this.locationOnly = locationOnly;
		}
		
		/** Barcode detection function 
		 * @param[in] yuv420 input image in YUV420 format
		 * @param[in] width width of the image
		 * @param[in] height height of the image
		 * @param[out] aBC barcode structure if found.
		 */
		private native int blade(byte[] yuv420, int height, int width, Barcode aBc, boolean locationOnly);

		@Override
		public void run() {
			while (!isFinished && !Thread.currentThread().isInterrupted()) {
				try {
					//Acquire the lock
					if (lock.isHeldByCurrentThread())
						Log.e(TAG, "Why is lock held by this thread already??");
					else if (lock.isLocked())
						Log.d(TAG, "Lock held by someone who is not us...");
					lock.lock();
					//Do the processing if not finished or interrupted
					Log.d(TAG, "Received frame for processing with height = " + height_ + ", width = " + width_);
					int result = blade(buffer_, height_, width_, bc, locationOnly);
					//Signal end of processing for this frame
					msgHandler.sendEmptyMessage((result < 0 ? Status.ERROR : result));
					//Finished processing, wait for new frame
					newFrameAvailable.await(); //lock is released
				}
				catch (InterruptedException e) {
					Log.d(TAG, "Processing thread interrupted.");
					e.printStackTrace();
					Thread.currentThread().interrupt();
					break;
				}
				finally {
					lock.unlock();
				}
			}
		}
		
		/**
		 * @class Define a message handler for the UI Thread
		 * @author kamyon
		 *
		 */
		private class MsgHandler extends Handler{
			@Override
			public void handleMessage(Message msg) 
			{ 
				int status;
				// this code runs in the main UI thread
				// so you can update the UI elements here
				if (!isFinished)
				{
					status = msg.what;
					switch (status) {
					case Status.ERROR: 
						isFinished = true;
						throw new RuntimeException("ENDER: JNI error");
					case Status.NOT_FOUND:
						if (feedback != null)
							feedback.set(null, width_, height_);
						//Add back the preview buffer
						camera_.releasePreviewBuffer();
						break;
					case Status.DETECTED:
						if (feedback != null)
							feedback.set(bc, width_, height_);
						callback_.onBarcodeDetected(bc);
						//Add back the preview buffer
						camera_.releasePreviewBuffer();
						//See if we should reattempt autofocus
						decodingAttempts++;
						if (decodingAttempts > 5) {
							camera_.attemptAutoFocus(true);
							decodingAttempts = 0;
						}
						break;
					case Status.DECODING_FAILED:
						if (feedback != null)
							feedback.set(bc, width_, height_);
						callback_.onBarcodeDecodingFailed(bc);
						//Add back the preview buffer
						camera_.releasePreviewBuffer();
						//See if we should reattempt autofocus
						decodingAttempts++; 
						if (decodingAttempts > 3) {
							camera_.attemptAutoFocus(true);
							decodingAttempts = 0;
						}
						break;
					case Status.DECODED:
						if (options_.isBarcodeDecodingOn) {
							if (feedback != null)
								feedback.pause();
							callback_.onBarcodeDecoded(bc);
						}
						else {
							if (feedback != null)
								feedback.set(bc, width_, height_);
							callback_.onBarcodeDecodingFailed(bc);
							//Add back the preview buffer
							camera_.releasePreviewBuffer();
							//See if we should reattempt autofocus
							decodingAttempts++; 
							if (decodingAttempts > 3) {
								camera_.attemptAutoFocus(true);
								decodingAttempts = 0;
							}							
						}
						break;
					default:
						super.handleMessage(msg);
					}
					mOverlaySV.updateOverlay(bc, status);
				}
			}
		}
	}
	
	/**
	 * @class Interface used to return barcode detection results, as well as sensor-related results.
	 * @author kamyon
	 *
	 */
	public interface OnBarcodeFound {
		//Barcode results:
		/**
		 * Called when a barcode is detected, but cannot yet be decoded
		 * @param bc detected barcode
		 */
		void onBarcodeDetected(Barcode bc);
		/**
		 * Called when a barcode is detected, but the decoding attempt has failed
		 * @param bc detected barcode
		 */
		void onBarcodeDecodingFailed(Barcode bc);
		/**
		 * Called when a barcode is successfully decoded
		 * @param bc decoded barcode
		 */
		void onBarcodeDecoded(Barcode bc);
	}
}