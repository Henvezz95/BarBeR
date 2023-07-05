################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../src/BLaDE.cpp \
../src/BLaDE_Impl.cpp \
../src/Decoder.cpp \
../src/Locator.cpp \
../src/Symbology.cpp \
../src/UPCASymbology.cpp 

OBJS += \
./src/BLaDE.o \
./src/BLaDE_Impl.o \
./src/Decoder.o \
./src/Locator.o \
./src/Symbology.o \
./src/UPCASymbology.o 

CPP_DEPS += \
./src/BLaDE.d \
./src/BLaDE_Impl.d \
./src/Decoder.d \
./src/Locator.d \
./src/Symbology.d \
./src/UPCASymbology.d 


# Each subdirectory must supply rules for building sources it contributes
src/%.o: ../src/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C++ Compiler'
	g++ -DDEBUG -I/home/kamyon/Projects/BLaDE_released/include -O0 -g3 -Wall -c -fmessage-length=0 -std=c++0x -fPIC -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


