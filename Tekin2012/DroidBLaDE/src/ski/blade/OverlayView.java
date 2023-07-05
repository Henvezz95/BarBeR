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

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.Paint.Align;
import android.graphics.PixelFormat;
import android.util.AttributeSet;
import android.util.Log;
import android.view.GestureDetector;
import android.view.MotionEvent;
import android.view.SurfaceHolder;
import android.view.SurfaceHolder.Callback;
import android.view.SurfaceView;

public class OverlayView extends SurfaceView implements Callback {
	Paint segmentPaint = new Paint();
	Paint textPaint = new Paint();
	Barcode bc = null;
	int status_ = FrameProcessor.Status.NOT_FOUND;
	boolean isReady = false;
	private SurfaceHolder mOverlaySH = getHolder();
	private float scaleX=0, scaleY=0;
	private int width=0, height=0;
	public int previewWidth = 640, previewHeight = 480;
	GestureDetector gestureScanner_;
	
	private final String TAG = "BLaDe OverlayView";
	
	public OverlayView(Context context, AttributeSet attributeSet) {
		super(context, attributeSet);
		mOverlaySH.addCallback(this);
		textPaint.setTextAlign(Align.CENTER);
		textPaint.setTextSize(20);
		textPaint.setColor(Color.BLUE);
		setBackgroundColor(Color.TRANSPARENT);
		mOverlaySH.setSizeFromLayout();
	}

	public void setGestureListener(GestureDetector aDetector) {
		gestureScanner_ = aDetector;
	}
	
	@Override
	public void onDraw(Canvas canvas) {
		if (isReady && (bc != null))
		{
			switch (status_) {
			case FrameProcessor.Status.DECODED:
				segmentPaint.setColor(Color.GREEN);
				break;
			case FrameProcessor.Status.DECODING_FAILED:
				segmentPaint.setColor(Color.GREEN);
				break;
			case FrameProcessor.Status.DETECTED:
				segmentPaint.setColor(Color.RED);
				break;
			default:
				Log.e(TAG, "onDraw should not be called without a detected barcode");
			}
			canvas.drawColor(Color.TRANSPARENT);
			canvas.drawLine(scaleX * (previewHeight - bc.y1), scaleY * bc.x1, scaleX * (previewHeight - bc.y2), scaleY * bc.x2, segmentPaint);
			if (bc.UPC != null)
				canvas.drawText(bc.UPC, height/3, width/2 + 20, textPaint);
		}
	}
	
	public void updateOverlay(Barcode barcode, int status) {
		if (status > FrameProcessor.Status.NOT_FOUND) {
			bc = barcode;
			status_ = status;
			if (isReady && (barcode != null)) {
				Canvas c = mOverlaySH.lockCanvas();
				draw(c);
				mOverlaySH.unlockCanvasAndPost(c);
				//invalidate();
			}
		}
	}
	
	@Override
	public void surfaceCreated(SurfaceHolder holder) {
		mOverlaySH.setFormat(PixelFormat.TRANSLUCENT);
	}

	@Override
	public void surfaceChanged(SurfaceHolder holder, int format, int w, int h) {
		//Surface is rotated 90 degrees
		height = w;
		width = h;
		scaleX = (float) width / previewWidth;
		scaleY = (float) height / previewHeight;
		isReady = true;
	}

	@Override
	public void surfaceDestroyed(SurfaceHolder holder) {
		isReady = false;
	}
	
	@Override
	public boolean onTouchEvent(MotionEvent event) {
		int action = event.getAction();
		if (action > MotionEvent.ACTION_POINTER_UP)
			return false;
		final String names[] = { "DOWN" , "UP" , "MOVE" , "CANCEL" , "OUTSIDE" ,
			      "POINTER_DOWN" , "POINTER_UP"};
		StringBuilder actionName = new StringBuilder();
		//Log touch event
		actionName.append("Action_").append(names[action]);
		if (action == MotionEvent.ACTION_POINTER_DOWN)
			actionName.append("(#").append(event.getActionIndex()).append(")");
		actionName.append("[");
		for (int i = 0; i < event.getPointerCount(); i++) {
			int ptr = event.getPointerId(i);
			actionName.append("(#").append(ptr).append(":").append(event.getX(ptr)).append(",").append(event.getY(ptr)).append(")");				
		}
		actionName.append("]");
		Log.d(TAG, "Touch event:" + actionName);
		return (gestureScanner_ == null ? false : gestureScanner_.onTouchEvent(event) );
	}
}
