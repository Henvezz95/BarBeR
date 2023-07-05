################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../src/AlsaAccess.cpp \
../src/BarcodeEngine.cpp \
../src/ProductSearch.cpp \
../src/ProductSearch_Google.cpp \
../src/SoundManager.cpp \
../src/iohandler.cpp \
../src/main.cpp \
../src/misc.cpp \
../src/opts.cpp 

OBJS += \
./src/AlsaAccess.o \
./src/BarcodeEngine.o \
./src/ProductSearch.o \
./src/ProductSearch_Google.o \
./src/SoundManager.o \
./src/iohandler.o \
./src/main.o \
./src/misc.o \
./src/opts.o 

CPP_DEPS += \
./src/AlsaAccess.d \
./src/BarcodeEngine.d \
./src/ProductSearch.d \
./src/ProductSearch_Google.d \
./src/SoundManager.d \
./src/iohandler.d \
./src/main.d \
./src/misc.d \
./src/opts.d 


# Each subdirectory must supply rules for building sources it contributes
src/%.o: ../src/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C++ Compiler'
	g++ -DDEBUG_ -DUSE_OPENCV -I"/home/kamyon/Projects/BLaDE_released/include" -O0 -Wall -Wextra -c -fmessage-length=0 -std=c++0x -v -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)"  -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


