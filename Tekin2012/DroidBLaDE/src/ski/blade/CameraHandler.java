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

import java.util.List;

import android.hardware.Camera;
import android.util.Log;
import android.view.SurfaceHolder;
import android.view.SurfaceView;

/**
 * @class class that handles camera operations and the preview surface
 * @author kamyon
 *
 */
public class CameraHandler {
	/** Width and height of requested preview */
	private int width_, height_;
	/** Signals whether camera is initialized */
	private boolean isInitialized_ = false;
	/** Actual camera device */
	private Camera camera_ = null;
	/** Camera buffer */
	private byte [] buffer = null;
	/** Callback interface */
	private CameraHandler.Callback callback_ = null;
	/** Autofocus handler  - created after camera is opened*/
	private FocusHandler focusHandler_ = null;
	/** Preview Surface Handler */
	private PreviewSurfaceHandler surfaceHandler_ = new PreviewSurfaceHandler();
	/** Camera Callback handler - created after camera is opened*/
	private CameraCallbackHandler cameraListener_ = null;

	/** Current flash mode */
	private String flashMode_ = null;
	/** Flash modes */
	public static final String FM_NOT_AVAILABLE = null;
	public static final String FM_ON = Camera.Parameters.FLASH_MODE_ON;
	public static final String FM_OFF = Camera.Parameters.FLASH_MODE_OFF;
	public static final String FM_TORCH = Camera.Parameters.FLASH_MODE_TORCH;
	public static final String FM_AUTO = Camera.Parameters.FLASH_MODE_AUTO;
	public static final String FM_RED_EYE = Camera.Parameters.FLASH_MODE_RED_EYE;
	//Logging tag
	private final String TAG = "BLaDE CameraHandler";

	/**
	 * Constructor
	 * @param w requested width of the camera preview
	 * @param h requested height of the camera preview
	 * @param previewSurface preview surface to use
	 * @param cameraCallback class that will receive callbacks from the camera handler
	 * 
	 */
	public CameraHandler(int w, int h, SurfaceView previewSurface, CameraHandler.Callback cameraCallback) {
		width_ = w;
		height_ = h;
		
		SurfaceHolder holder = previewSurface.getHolder();
		holder.setSizeFromLayout();
		//holder.setType(SurfaceHolder.SURFACE_TYPE_PUSH_BUFFERS); //TODO: see if removing this crashed the code - it was earlier
		holder.setKeepScreenOn(true);
		holder.addCallback(surfaceHandler_);
		
		//Set callback handle
		callback_ = cameraCallback;
	}

	/**
	 * First stage camera initialization, called when the preview surface is created.
	 */
	private void initialize1() {
		//Open camera
		camera_ = Camera.open();
		//Check that there is a camera, and if not, quit
		if (camera_ == null) {
			Log.e(TAG, "Cannot open camera");
			release();
		}
		//Start receiving callbacks from camera
		cameraListener_ = new CameraCallbackHandler();
		
		//See whether camera has flash
		Camera.Parameters param = camera_.getParameters();
		//Flash mode
		flashMode_ = param.getFlashMode();
		if (flashMode_ == null)
			Log.d(TAG,"Flash not available");
		
		//Set the focus handler
		focusHandler_ = new FocusHandler();
	}
	
	/**
	 * Seconds stage initialization - called when the preview surface is changed
	 * @param holder surface holder that holds the preview surface
	 */
	private void initialize2(SurfaceHolder holder) {
		// The Surface has been created, tell the camera where to draw.
		try {
			//Set the preview display
			camera_.setPreviewDisplay(holder);
			//Set up camera parameters
			Camera.Parameters param = camera_.getParameters();
			param.setPreviewSize(width_, height_);	//TODO: change this so we make sure it is an appropriate size
			camera_.setParameters(param);
			buffer = new byte[width_ * height_ * 4]; //TODO: change this based on actual preview size obtained
			camera_.setDisplayOrientation(90);
			//Camera is now initialized - signal this to the callback interface.
			isInitialized_ = true;
			if (callback_ != null)
				callback_.onCameraInitialized();
		} 
		catch (Exception exception) {
			Log.e(TAG, exception.getMessage());
			exception.printStackTrace();
			release();
		}
	}
	
	/**
	 * Returns whether camera is initialized
	 * @return true if camera is successfully initialized.
	 */
	public boolean isInitialized() {
		return isInitialized_;
	}
	
	/**
	 * Starts the preview and callbacks
	 */
	public void startPreview() {
		//If not initialized, initialize camera
		if (camera_ != null)
		{
			//Set scene mode to "barcode" if it is supported
			Camera.Parameters param = camera_.getParameters();
			List<String> sceneModes = param.getSupportedSceneModes();
			if (sceneModes.contains(Camera.Parameters.SCENE_MODE_BARCODE)) {
				Log.v(TAG, "Setting scene mode to barcode");
				param.setSceneMode(Camera.Parameters.SCENE_MODE_BARCODE);
				camera_.setParameters(param);
			}
			
			//Add preview callbacks
			camera_.setPreviewCallbackWithBuffer(cameraListener_);
			camera_.addCallbackBuffer(buffer);
			
			//Start preview
			try{
				camera_.startPreview();
			}
			catch (RuntimeException e) {
				Log.e(TAG, "Camera preview failed to start");
				e.printStackTrace();
				release();
			}
		}
		else
		{
			throw new RuntimeException(TAG + ": Camera not initialized");
		}
	}
	
	/**
	 * Stops the preview and the callbacks
	 */
	public void stopPreview() {
		if (camera_ != null)
		{
			camera_.setPreviewCallbackWithBuffer(null);
			camera_.stopPreview();
		}
	}
		
	/**
	 * Stops the previews and callbacks, and releases the camera
	 */
	public void release() {
		//signal that we are releasing camera resources
		if (callback_ != null)
			callback_.onCameraReleased();
		if (camera_ != null)
		{
			//Stop the preview
			stopPreview();
			//Release camera
			camera_.release();
			camera_ = null;
		}
		//Release the autofocus handler
		focusHandler_ = null;
		//Release camera callback handler
		cameraListener_ = null;
		//Release callback handle
		callback_ = null;
		isInitialized_ = false;
	}
	
	protected void finalize() {
		release();
	}

	/**
	 * Attempts to autofocus the camera
	 * @param doForceRefocus whether we should try another autofocus even if the camera is focused
	 */
	public void attemptAutoFocus(boolean doForceRefocus) {
		focusHandler_.attemptAutoFocus(doForceRefocus);
	}
	
	/**
	 * Returns camera flash mode
	 * @return current flash mode;
	 */
	public String getFlashMode() {
		return flashMode_;
	}
	
	/**
	 * Returns whether camera has flash
	 * @return true if camera is determined to have flash
	 */
	public boolean hasFlash() {
		return (flashMode_ != FM_NOT_AVAILABLE);
	}
	
	/**
	 * Sets flash mode if it is supported
	 * @param flashMode flash mode to set
	 * @return returns the current flash mode
	 */
	public String setFlashMode(String flashMode) {
		if (flashMode_ != null)
		{
			if (camera_ != null) {
				Camera.Parameters param = camera_.getParameters();
				if (param.getSupportedFlashModes().contains(flashMode))
				{
					param.setFlashMode(flashMode);
					camera_.setParameters(param);
					flashMode_ = flashMode;
				}
			}
		}
		return flashMode_;
	}
	
	/**
	 * Releases the preview buffer, so that we start getting callbacks from the camera
	 */
	public void releasePreviewBuffer() {
		camera_.addCallbackBuffer(buffer);
	}
	
	/**
	 * @class Focus handler - wraps around Camera.AutoFocusCallback
	 * @author kamyon
	 *
	 */
	private class FocusHandler implements Camera.AutoFocusCallback {
		//Autofocus status flags
		private final int FS_STATIC = 0;	//for static modes
		private final int FS_FOCUSING = 1;	//currently attempting autofocus
		private final int FS_FOCUSED = 2;	//camera focused
		private final int FS_OUT_OF_FOCUS = 3;	//camera out of focus
		//AF status
		private int focusStatus_ = FS_OUT_OF_FOCUS;
		
		/** Constructor */
		public FocusHandler() {
			if (camera_ == null)
				throw new RuntimeException(TAG + ": Camera is not yet open");
			Camera.Parameters param = camera_.getParameters();
			List<String> fModes = param.getSupportedFocusModes();
			//TODO: In IceCream Sandwich, try Camera.Parameters.FOCUS_MODE_CONTINUOUS_PICTURE which should be faster
			if (fModes.contains(Camera.Parameters.FOCUS_MODE_CONTINUOUS_PICTURE))
			{
				param.setFocusMode(Camera.Parameters.FOCUS_MODE_CONTINUOUS_PICTURE);
				focusStatus_ = FS_STATIC;	//Will not attempt autofocus
			}
			else if (fModes.contains(Camera.Parameters.FOCUS_MODE_CONTINUOUS_VIDEO))
			{
				param.setFocusMode(Camera.Parameters.FOCUS_MODE_CONTINUOUS_VIDEO);
				focusStatus_ = FS_STATIC;	//Will not attempt autofocus
				Log.d(TAG, "Continuous Focus mode available");
			}
			else if (fModes.contains(Camera.Parameters.FOCUS_MODE_MACRO))
				param.setFocusMode(Camera.Parameters.FOCUS_MODE_MACRO);
			else if (fModes.contains(Camera.Parameters.FOCUS_MODE_AUTO))
				param.setFocusMode(Camera.Parameters.FOCUS_MODE_AUTO);
			else
				focusStatus_ = FS_STATIC;	//no need to keep trying
			//Set focus mode
			camera_.setParameters(param);
		}
		
		//Camera.AutoFocusCallback methods
		@Override
		public void onAutoFocus(boolean success, Camera camera) {
			if (success)
			{
				Log.d(TAG, "Camera Focused");
				focusStatus_ = FS_FOCUSED;
			}
			else
			{
				Log.d(TAG, "Autofocus Failed");
				focusStatus_ = FS_OUT_OF_FOCUS;
			}
		}
		
		//Other methods
		/**
		 * Attempts autofocus if it is supported.
		 * @param doForceRefocus if true, will attempt autofocus even if camera is focused. If false, autofocus is only attempted if the camera is declared out of focus
		 */
		public void attemptAutoFocus(boolean doForceRefocus) {
			if ( (focusStatus_ == FS_OUT_OF_FOCUS) || (doForceRefocus && (focusStatus_ == FS_FOCUSED)) )
			{
				camera_.autoFocus(this);
				Log.d(TAG, "Attempting autofocus");
				focusStatus_ = FS_FOCUSING;
			}
		}
	}

	/**
	 * @class PreviewHandler handles the preview setup when video frames are available
	 * @author kamyon
	 *
	 */
	private class PreviewSurfaceHandler implements SurfaceHolder.Callback {

		@Override
		public void surfaceCreated(SurfaceHolder holder) {
			Log.d(TAG, "Preview surface created");
			initialize1();
		}

		@Override
		public void surfaceChanged(SurfaceHolder holder, int format, int width, int height) {
			Log.d(TAG, "Preview surface initialized - width = " + width + ", height = " + height);
			initialize2(holder);
		}

		@Override
		public void surfaceDestroyed(SurfaceHolder holder) {
			Log.d(TAG, "Preview surface destroyed");
			release();
		}

	}
	
	/**
	 * @class Provides information about camera errors:
	 * @author kamyon
	 *
	 */
	private class CameraCallbackHandler implements Camera.PreviewCallback, Camera.ErrorCallback {

		/**
		 * Constructor
		 */
		public CameraCallbackHandler() {
			if (camera_ != null) {
				camera_.setErrorCallback(this);
			}
		}
		
		//Camera.PreviewCallback methods
		@Override
		public void onPreviewFrame(byte[] data, Camera camera) {
			if (callback_ != null)
				callback_.onFrameReceived(data, width_, height_);
		}

		//Camera.ErrorCallback methods
		@Override
		public void onError(int error, Camera camera) {
			switch (error) {
			case Camera.CAMERA_ERROR_SERVER_DIED:
				Log.e(TAG, "Camera served died. Releasing camera handler.");
				release();
				break;
			case Camera.CAMERA_ERROR_UNKNOWN:
				Log.e(TAG, "Unspecified camera error");
				break;
			default:
				Log.e(TAG, "Other Camera Errors?");
			}
		}
	}
	
	//Interface
	public interface Callback {
		/**
		 * Called every time a video frame is available for processing. 
		 * After processing the received buffer, it must be released by calling releasePreviewBuffer() on the camera handler;
		 * @param buf video frame data
		 * @param w width of provided frame
		 * @param h height of provided frame
		 */
		void onFrameReceived(byte [] buf, int w, int h);
		
		/**
		 * Called when the camera is initialized and preview surface is set up.
		 */
		void onCameraInitialized();
		
		/**
		 * Called when the camera is about to be released.
		 */
		void onCameraReleased();
	}
}
