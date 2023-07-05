LOCAL_PATH := $(call my-dir)

include $(CLEAR_VARS)

LOCAL_MODULE    := BLaDE_JNI
### Add all source file names to be included in lib separated by a whitespace
LOCAL_SRC_FILES := BLaDE_jni.cpp
LOCAL_CFLAGS := -O3 -I$(LOCAL_PATH)/../../include
LOCAL_LDLIBS := -llog -latomic
LOCAL_ARM_MODE := arm
LOCAL_STATIC_LIBRARIES := BLaDE
include $(BUILD_SHARED_LIBRARY)
include $(LOCAL_PATH)/../../BLaDE/src/Android.mk
