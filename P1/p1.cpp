#include <stdio.h>
#include <CL/cl.h>
#include <windows.h>
#include <string>
#include <iostream>
#include <thread>
#include <glew.h>
#include <freeglut.h>

#include <stdlib.h>
#include <tchar.h>
#include <memory.h>
#include <vector>

#include <fstream>
#include <sstream>

#if !defined(CL_CALLBACK)
#define CL_CALLBACK
#endif

// Macros for OpenCL versions
#define OPENCL_VERSION_1_2  1.2f
#define OPENCL_VERSION_2_0  2.0f

// Suppress a compiler warning about undefined CL_TARGET_OPENCL_VERSION
// Khronos ICD supports only latest OpenCL version
#define CL_TARGET_OPENCL_VERSION 220

// Suppress a compiler warning about 'clCreateCommandQueue': was declared deprecated
// for OpenCL 1.2
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

//#include "CL\cl.h"
#include "utils.h"
#pragma warning( disable : 4996 )

//for perf. counters
#include <Windows.h>

using namespace std;

GLfloat r = 0.0f, g = 0.0f, b = 0.0f;
const int WIDTH = 1024, HEIGHT = 768;
int cell[WIDTH][HEIGHT];
// injection circle points x, y
int cx[9];
int cy[9];
int cx1[256];  //up
int ccx1[256];
int cy1[256];
int ccy1[256]; //y-1,y-2,... y--
int cx2[256]; //down
int ccx2[256];
int cy2[256];
int ccy2[256];
int cx3[256]; //left GPU
int ccx3[256];
int cy3[256];
int ccy3[256];
int cx4[256]; //right
int ccx4[256];
int cy4[256];
int ccy4[256];
int cx5[256];
int ccx5[256];
int cy5[256];
int ccy5[256];
int cx6[256]; //injectParallel() moveParallel() moveWithCuda() moveKernel()
int ccx6[256];
int cy6[256];
int ccy6[256];
int cx10[256];  //up
int ccx10[256];
int cy10[256];
int ccy10[256]; 
int cx20[256]; //down
int ccx20[256];
int cy20[256];
int ccy20[256];
int cx30[256]; //left CPU
int ccx30[256];
int cy30[256];
int ccy30[256];
int cx301[256]; //left uhdGPU num=3
int ccx301[256];
int cy301[256];
int ccy301[256];
int cx40[256]; //right
int ccx40[256];
int cy40[256];
int ccy40[256];
int cx401[256]; //right uhdGPU num=3
int ccx401[256];
int cy401[256];
int ccy401[256];
int cx11[256]; //up num=5 GPU
int ccx11[256];
int cy11[256];
int ccy11[256];
int cx21[256]; //down
int ccx21[256];
int cy21[256];
int ccy21[256];
int cx31[256]; //left
int ccx31[256]; 
int cy31[256];
int ccy31[256];
int cx41[256]; //right
int ccx41[256];
int cy41[256];
int ccy41[256];
int cx110[256]; //up num=5 CPU
int ccx110[256];
int cy110[256];
int ccy110[256];
int m[9];
int m1[9]; //up
int m2[9]; //down
int m3[9]; //left GPU
int m4[9]; //right GPU
int m5[9];
int m6[9]; //injectParallel() moveParallel() moveWithCuda() moveKernel()
int m10[9]; //up 
int m20[9]; //down 
int m30[9]; //left CPU
int m301[9]; //left uhdGPU num=3
int m40[9]; //right CPU
int m401[9]; //right uhdGPU
int m11[9]; //up
int m21[9]; //down
int m31[9]; //left
int m41[9]; //right
int m110[9]; //up
int q;
int q1; //up
int q2; //down
int q3; //left GPU
int q4; //right GPU
int q10; //up
int q20; //down
int q30; //left CPU
int q301; //left uhdGPU num=3
int q40; //right CPU
int q401; //right uhdGPU num=3
int q11; //up
int q21; //down
int q31; //left
int q41; //right
int q110; //up

//======================== from opencl book ===================
// Function to check and handle OpenCL errors
inline void
checkErr(cl_int err, const char* name)
{
	if (err != CL_SUCCESS) {
		std::cerr << "ERROR: " << name << " (" << err << ")" << std::endl;
		exit(EXIT_FAILURE);
	}
}

void CL_CALLBACK contextCallback(const char* errInfo, const void* private_info,	size_t cb,void* user_data)
{
	std::cout << "Error occured during context use: " << errInfo << std::endl;
	// should really perform any clearup and so on at this point
	// but for simplicitly just exit.
	exit(1);
}
//===================== ^ from opencl book==========================

const char* TranslateOpenCLError(cl_int errorCode)
{
	switch (errorCode)
	{
	case CL_SUCCESS:                            return "CL_SUCCESS";
	case CL_DEVICE_NOT_FOUND:                   return "CL_DEVICE_NOT_FOUND";
	case CL_DEVICE_NOT_AVAILABLE:               return "CL_DEVICE_NOT_AVAILABLE";
	case CL_COMPILER_NOT_AVAILABLE:             return "CL_COMPILER_NOT_AVAILABLE";
	case CL_MEM_OBJECT_ALLOCATION_FAILURE:      return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
	case CL_OUT_OF_RESOURCES:                   return "CL_OUT_OF_RESOURCES";
	case CL_OUT_OF_HOST_MEMORY:                 return "CL_OUT_OF_HOST_MEMORY";
	case CL_PROFILING_INFO_NOT_AVAILABLE:       return "CL_PROFILING_INFO_NOT_AVAILABLE";
	case CL_MEM_COPY_OVERLAP:                   return "CL_MEM_COPY_OVERLAP";
	case CL_IMAGE_FORMAT_MISMATCH:              return "CL_IMAGE_FORMAT_MISMATCH";
	case CL_IMAGE_FORMAT_NOT_SUPPORTED:         return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
	case CL_BUILD_PROGRAM_FAILURE:              return "CL_BUILD_PROGRAM_FAILURE";
	case CL_MAP_FAILURE:                        return "CL_MAP_FAILURE";
	case CL_MISALIGNED_SUB_BUFFER_OFFSET:       return "CL_MISALIGNED_SUB_BUFFER_OFFSET";                          //-13
	case CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST:    return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";   //-14
	case CL_COMPILE_PROGRAM_FAILURE:            return "CL_COMPILE_PROGRAM_FAILURE";                               //-15
	case CL_LINKER_NOT_AVAILABLE:               return "CL_LINKER_NOT_AVAILABLE";                                  //-16
	case CL_LINK_PROGRAM_FAILURE:               return "CL_LINK_PROGRAM_FAILURE";                                  //-17
	case CL_DEVICE_PARTITION_FAILED:            return "CL_DEVICE_PARTITION_FAILED";                               //-18
	case CL_KERNEL_ARG_INFO_NOT_AVAILABLE:      return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";                         //-19
	case CL_INVALID_VALUE:                      return "CL_INVALID_VALUE";
	case CL_INVALID_DEVICE_TYPE:                return "CL_INVALID_DEVICE_TYPE";
	case CL_INVALID_PLATFORM:                   return "CL_INVALID_PLATFORM";
	case CL_INVALID_DEVICE:                     return "CL_INVALID_DEVICE";
	case CL_INVALID_CONTEXT:                    return "CL_INVALID_CONTEXT";
	case CL_INVALID_QUEUE_PROPERTIES:           return "CL_INVALID_QUEUE_PROPERTIES";
	case CL_INVALID_COMMAND_QUEUE:              return "CL_INVALID_COMMAND_QUEUE";
	case CL_INVALID_HOST_PTR:                   return "CL_INVALID_HOST_PTR";
	case CL_INVALID_MEM_OBJECT:                 return "CL_INVALID_MEM_OBJECT";
	case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:    return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
	case CL_INVALID_IMAGE_SIZE:                 return "CL_INVALID_IMAGE_SIZE";
	case CL_INVALID_SAMPLER:                    return "CL_INVALID_SAMPLER";
	case CL_INVALID_BINARY:                     return "CL_INVALID_BINARY";
	case CL_INVALID_BUILD_OPTIONS:              return "CL_INVALID_BUILD_OPTIONS";
	case CL_INVALID_PROGRAM:                    return "CL_INVALID_PROGRAM";
	case CL_INVALID_PROGRAM_EXECUTABLE:         return "CL_INVALID_PROGRAM_EXECUTABLE";
	case CL_INVALID_KERNEL_NAME:                return "CL_INVALID_KERNEL_NAME";
	case CL_INVALID_KERNEL_DEFINITION:          return "CL_INVALID_KERNEL_DEFINITION";
	case CL_INVALID_KERNEL:                     return "CL_INVALID_KERNEL";
	case CL_INVALID_ARG_INDEX:                  return "CL_INVALID_ARG_INDEX";
	case CL_INVALID_ARG_VALUE:                  return "CL_INVALID_ARG_VALUE";
	case CL_INVALID_ARG_SIZE:                   return "CL_INVALID_ARG_SIZE";
	case CL_INVALID_KERNEL_ARGS:                return "CL_INVALID_KERNEL_ARGS";
	case CL_INVALID_WORK_DIMENSION:             return "CL_INVALID_WORK_DIMENSION";
	case CL_INVALID_WORK_GROUP_SIZE:            return "CL_INVALID_WORK_GROUP_SIZE";
	case CL_INVALID_WORK_ITEM_SIZE:             return "CL_INVALID_WORK_ITEM_SIZE";
	case CL_INVALID_GLOBAL_OFFSET:              return "CL_INVALID_GLOBAL_OFFSET";
	case CL_INVALID_EVENT_WAIT_LIST:            return "CL_INVALID_EVENT_WAIT_LIST";
	case CL_INVALID_EVENT:                      return "CL_INVALID_EVENT";
	case CL_INVALID_OPERATION:                  return "CL_INVALID_OPERATION";
	case CL_INVALID_GL_OBJECT:                  return "CL_INVALID_GL_OBJECT";
	case CL_INVALID_BUFFER_SIZE:                return "CL_INVALID_BUFFER_SIZE";
	case CL_INVALID_MIP_LEVEL:                  return "CL_INVALID_MIP_LEVEL";
	case CL_INVALID_GLOBAL_WORK_SIZE:           return "CL_INVALID_GLOBAL_WORK_SIZE";                           //-63
	case CL_INVALID_PROPERTY:                   return "CL_INVALID_PROPERTY";                                   //-64
	case CL_INVALID_IMAGE_DESCRIPTOR:           return "CL_INVALID_IMAGE_DESCRIPTOR";                           //-65
	case CL_INVALID_COMPILER_OPTIONS:           return "CL_INVALID_COMPILER_OPTIONS";                           //-66
	case CL_INVALID_LINKER_OPTIONS:             return "CL_INVALID_LINKER_OPTIONS";                             //-67
	case CL_INVALID_DEVICE_PARTITION_COUNT:     return "CL_INVALID_DEVICE_PARTITION_COUNT";                     //-68
//    case CL_INVALID_PIPE_SIZE:                  return "CL_INVALID_PIPE_SIZE";                                  //-69
//    case CL_INVALID_DEVICE_QUEUE:               return "CL_INVALID_DEVICE_QUEUE";                               //-70    

	default:
		return "UNKNOWN ERROR CODE";
	}
}

struct ocl_args_d_t
{
	ocl_args_d_t();
	~ocl_args_d_t();

	// Regular OpenCL objects:
	cl_context       context;           // hold the context handler
	cl_device_id     device;            // hold the selected device handler
	cl_command_queue commandQueue;      // hold the commands-queue handler
	cl_program       program;           // hold the program handler
	cl_kernel        kernel1;            // hold the kernel handler
	cl_kernel        kernel2;            // hold the kernel handler
	cl_kernel        kernel3;            // hold the kernel handler
	cl_kernel        kernel4;            // hold the kernel handler
	float            platformVersion;   // hold the OpenCL platform version (default 1.2)
	float            deviceVersion;     // hold the OpenCL device version (default. 1.2)
	float            compilerVersion;   // hold the device OpenCL C version (default. 1.2)

	// Objects that are specific for algorithm implemented in this sample
	cl_mem           srcA;              // hold first source buffer
	//cl_mem           srcB;              // hold second source buffer
	cl_mem           dstMemA;            // hold destination buffer
	cl_mem           dstMemB;            // hold destination buffer
	cl_mem           dstMemC;            // hold destination buffer
	cl_mem           dstMemD;            // hold destination buffer
};

ocl_args_d_t::ocl_args_d_t() :
	context(NULL),
	device(NULL),
	commandQueue(NULL),
	program(NULL),
	kernel1(NULL),
	kernel2(NULL),
	kernel3(NULL),
	kernel4(NULL),
	platformVersion(OPENCL_VERSION_1_2),
	deviceVersion(OPENCL_VERSION_1_2),
	compilerVersion(OPENCL_VERSION_1_2),
	srcA(NULL),
	//srcB(NULL),
	dstMemA(NULL),
	dstMemB(NULL),
	dstMemC(NULL),
	dstMemD(NULL)
{
}

ocl_args_d_t::~ocl_args_d_t()
{
	cl_int err = CL_SUCCESS;

	if (kernel1)
	{
		err = clReleaseKernel(kernel1);
		if (CL_SUCCESS != err)
		{
			LogError("Error: clReleaseKernel returned '%s'.\n", TranslateOpenCLError(err));
		}
	}
	if (kernel2)
	{
		err = clReleaseKernel(kernel2);
		if (CL_SUCCESS != err)
		{
			LogError("Error: clReleaseKernel returned '%s'.\n", TranslateOpenCLError(err));
		}
	}
	if (kernel3)
	{
		err = clReleaseKernel(kernel3);
		if (CL_SUCCESS != err)
		{
			LogError("Error: clReleaseKernel returned '%s'.\n", TranslateOpenCLError(err));
		}
	}
	if (kernel4)
	{
		err = clReleaseKernel(kernel4);
		if (CL_SUCCESS != err)
		{
			LogError("Error: clReleaseKernel returned '%s'.\n", TranslateOpenCLError(err));
		}
	}
	if (program)
	{
		err = clReleaseProgram(program);
		if (CL_SUCCESS != err)
		{
			LogError("Error: clReleaseProgram returned '%s'.\n", TranslateOpenCLError(err));
		}
	}
	if (srcA)
	{
		err = clReleaseMemObject(srcA);
		if (CL_SUCCESS != err)
		{
			LogError("Error: clReleaseMemObject returned '%s'.\n", TranslateOpenCLError(err));
		}
	}
	/*if (srcB)
	{
		err = clReleaseMemObject(srcB);
		if (CL_SUCCESS != err)
		{
			LogError("Error: clReleaseMemObject returned '%s'.\n", TranslateOpenCLError(err));
		}
	}*/
	if (dstMemA)
	{
		err = clReleaseMemObject(dstMemA);
		if (CL_SUCCESS != err)
		{
			LogError("Error: clReleaseMemObject returned '%s'.\n", TranslateOpenCLError(err));
		}
	}
	if (dstMemB)
	{
		err = clReleaseMemObject(dstMemB);
		if (CL_SUCCESS != err)
		{
			LogError("Error: clReleaseMemObject returned '%s'.\n", TranslateOpenCLError(err));
		}
	}
	if (dstMemC)
	{
		err = clReleaseMemObject(dstMemC);
		if (CL_SUCCESS != err)
		{
			LogError("Error: clReleaseMemObject returned '%s'.\n", TranslateOpenCLError(err));
		}
	}
	if (dstMemD)
	{
		err = clReleaseMemObject(dstMemD);
		if (CL_SUCCESS != err)
		{
			LogError("Error: clReleaseMemObject returned '%s'.\n", TranslateOpenCLError(err));
		}
	}
	if (commandQueue)
	{
		err = clReleaseCommandQueue(commandQueue);
		if (CL_SUCCESS != err)
		{
			LogError("Error: clReleaseCommandQueue returned '%s'.\n", TranslateOpenCLError(err));
		}
	}
	if (device)
	{
		err = clReleaseDevice(device);
		if (CL_SUCCESS != err)
		{
			LogError("Error: clReleaseDevice returned '%s'.\n", TranslateOpenCLError(err));
		}
	}
	if (context)
	{
		err = clReleaseContext(context);
		if (CL_SUCCESS != err)
		{
			LogError("Error: clReleaseContext returned '%s'.\n", TranslateOpenCLError(err));
		}
	}

	/*
	 * Note there is no procedure to deallocate platform
	 * because it was not created at the startup,
	 * but just queried from OpenCL runtime.
	 */
}

bool CheckPreferredPlatformMatch(cl_platform_id platform, const char* preferredPlatform)
{
	size_t stringLength = 0;
	cl_int err = CL_SUCCESS;
	bool match = false;

	// In order to read the platform's name, we first read the platform's name string length (param_value is NULL).
	// The value returned in stringLength
	err = clGetPlatformInfo(platform, CL_PLATFORM_NAME, 0, NULL, &stringLength);
	if (CL_SUCCESS != err)
	{
		LogError("Error: clGetPlatformInfo() to get CL_PLATFORM_NAME length returned '%s'.\n", TranslateOpenCLError(err));
		return false;
	}

	// Now, that we know the platform's name string length, we can allocate enough space before read it
	std::vector<char> platformName(stringLength);

	// Read the platform's name string
	// The read value returned in platformName
	err = clGetPlatformInfo(platform, CL_PLATFORM_NAME, stringLength, &platformName[0], NULL);
	if (CL_SUCCESS != err)
	{
		LogError("Error: clGetplatform_ids() to get CL_PLATFORM_NAME returned %s.\n", TranslateOpenCLError(err));
		return false;
	}

	// Now check if the platform's name is the required one
	if (strstr(&platformName[0], preferredPlatform) != 0)
	{
		// The checked platform is the one we're looking for
		LogInfo("Platform: %s\n", &platformName[0]);
		match = true;
	}

	return match;
}

cl_platform_id FindOpenCLPlatform(const char* preferredPlatform, cl_device_type deviceType)
{
	cl_uint numPlatforms = 0;
	cl_int err = CL_SUCCESS;

	// Get (in numPlatforms) the number of OpenCL platforms available
	// No platform ID will be return, since platforms is NULL
	err = clGetPlatformIDs(0, NULL, &numPlatforms);
	if (CL_SUCCESS != err)
	{
		LogError("Error: clGetplatform_ids() to get num platforms returned %s.\n", TranslateOpenCLError(err));
		return NULL;
	}
	LogInfo("Number of available platforms: %u\n", numPlatforms);

	if (0 == numPlatforms)
	{
		LogError("Error: No platforms found!\n");
		return NULL;
	}

	std::vector<cl_platform_id> platforms(numPlatforms);

	// Now, obtains a list of numPlatforms OpenCL platforms available
	// The list of platforms available will be returned in platforms
	err = clGetPlatformIDs(numPlatforms, &platforms[0], NULL);
	if (CL_SUCCESS != err)
	{
		LogError("Error: clGetplatform_ids() to get platforms returned %s.\n", TranslateOpenCLError(err));
		return NULL;
	}

	// Check if one of the available platform matches the preferred requirements
	for (cl_uint i = 0; i < numPlatforms; i++)
	{
		bool match = true;
		cl_uint numDevices = 0;

		// If the preferredPlatform is not NULL then check if platforms[i] is the required one
		// Otherwise, continue the check with platforms[i]
		if ((NULL != preferredPlatform) && (strlen(preferredPlatform) > 0))
		{
			// In case we're looking for a specific platform
			match = CheckPreferredPlatformMatch(platforms[i], preferredPlatform);
		}

		// match is true if the platform's name is the required one or don't care (NULL)
		if (match)
		{
			// Obtains the number of deviceType devices available on platform
			// When the function failed we expect numDevices to be zero.
			// We ignore the function return value since a non-zero error code
			// could happen if this platform doesn't support the specified device type.
			err = clGetDeviceIDs(platforms[i], deviceType, 0, NULL, &numDevices);
			if (CL_SUCCESS != err)
			{
				LogInfo("   Required device was not found on this platform.\n");
			}

			if (0 != numDevices)
			{
				// There is at list one device that answer the requirements
				LogInfo("   Required device was found.\n");
				return platforms[i];
			}
		}
	}

	LogError("Error: Required device was not found on any platform.\n");
	return NULL;
}

int GetPlatformAndDeviceVersion(cl_platform_id platformId, ocl_args_d_t* ocl)
{
	cl_int err = CL_SUCCESS;

	// Read the platform's version string length (param_value is NULL).
	// The value returned in stringLength
	size_t stringLength = 0;
	err = clGetPlatformInfo(platformId, CL_PLATFORM_VERSION, 0, NULL, &stringLength);
	if (CL_SUCCESS != err)
	{
		LogError("Error: clGetPlatformInfo() to get CL_PLATFORM_VERSION length returned '%s'.\n", TranslateOpenCLError(err));
		return err;
	}

	// Now, that we know the platform's version string length, we can allocate enough space before read it
	std::vector<char> platformVersion(stringLength);

	// Read the platform's version string
	// The read value returned in platformVersion
	err = clGetPlatformInfo(platformId, CL_PLATFORM_VERSION, stringLength, &platformVersion[0], NULL);
	if (CL_SUCCESS != err)
	{
		LogError("Error: clGetplatform_ids() to get CL_PLATFORM_VERSION returned %s.\n", TranslateOpenCLError(err));
		return err;
	}

	if (strstr(&platformVersion[0], "OpenCL 2.0") != NULL)
	{
		ocl->platformVersion = OPENCL_VERSION_2_0;
	}

	// Read the device's version string length (param_value is NULL).
	err = clGetDeviceInfo(ocl->device, CL_DEVICE_VERSION, 0, NULL, &stringLength);
	if (CL_SUCCESS != err)
	{
		LogError("Error: clGetDeviceInfo() to get CL_DEVICE_VERSION length returned '%s'.\n", TranslateOpenCLError(err));
		return err;
	}

	// Now, that we know the device's version string length, we can allocate enough space before read it
	std::vector<char> deviceVersion(stringLength);

	// Read the device's version string
	// The read value returned in deviceVersion
	err = clGetDeviceInfo(ocl->device, CL_DEVICE_VERSION, stringLength, &deviceVersion[0], NULL);
	if (CL_SUCCESS != err)
	{
		LogError("Error: clGetDeviceInfo() to get CL_DEVICE_VERSION returned %s.\n", TranslateOpenCLError(err));
		return err;
	}

	if (strstr(&deviceVersion[0], "OpenCL 2.0") != NULL)
	{
		ocl->deviceVersion = OPENCL_VERSION_2_0;
	}

	// Read the device's OpenCL C version string length (param_value is NULL).
	err = clGetDeviceInfo(ocl->device, CL_DEVICE_OPENCL_C_VERSION, 0, NULL, &stringLength);
	if (CL_SUCCESS != err)
	{
		LogError("Error: clGetDeviceInfo() to get CL_DEVICE_OPENCL_C_VERSION length returned '%s'.\n", TranslateOpenCLError(err));
		return err;
	}

	// Now, that we know the device's OpenCL C version string length, we can allocate enough space before read it
	std::vector<char> compilerVersion(stringLength);

	// Read the device's OpenCL C version string
	// The read value returned in compilerVersion
	err = clGetDeviceInfo(ocl->device, CL_DEVICE_OPENCL_C_VERSION, stringLength, &compilerVersion[0], NULL);
	if (CL_SUCCESS != err)
	{
		LogError("Error: clGetDeviceInfo() to get CL_DEVICE_OPENCL_C_VERSION returned %s.\n", TranslateOpenCLError(err));
		return err;
	}

	else if (strstr(&compilerVersion[0], "OpenCL C 2.0") != NULL)
	{
		ocl->compilerVersion = OPENCL_VERSION_2_0;
	}

	return err;
}

// change here
void generateInput(cl_int* inputArray, cl_uint arrayWidth, cl_uint arrayHeight)
{
	// initialization of input
	cl_uint array_size = arrayWidth * arrayHeight;
	for (cl_uint i = 0; i < array_size; ++i)
	{
		inputArray[i] = i;
	}
}

int SetupOpenCL(ocl_args_d_t* ocl, cl_device_type deviceType)
{
	// The following variable stores return codes for all OpenCL calls.
	cl_int err = CL_SUCCESS;

	// Query for all available OpenCL platforms on the system
	// Here you enumerate all platforms and pick one which name has preferredPlatform as a sub-string
	cl_platform_id platformId = FindOpenCLPlatform("Intel", deviceType);
	if (NULL == platformId)
	{
		LogError("Error: Failed to find OpenCL platform.\n");
		return CL_INVALID_VALUE;
	}

	// Create context with device of specified type.
	// Required device type is passed as function argument deviceType.
	// So you may use this function to create context for any CPU or GPU OpenCL device.
	// The creation is synchronized (pfn_notify is NULL) and NULL user_data
	cl_context_properties contextProperties[] = { CL_CONTEXT_PLATFORM, (cl_context_properties)platformId, 0 };
	ocl->context = clCreateContextFromType(contextProperties, deviceType, NULL, NULL, &err);
	if ((CL_SUCCESS != err) || (NULL == ocl->context))
	{
		LogError("Couldn't create a context, clCreateContextFromType() returned '%s'.\n", TranslateOpenCLError(err));
		return err;
	}

	// Query for OpenCL device which was used for context creation
	err = clGetContextInfo(ocl->context, CL_CONTEXT_DEVICES, sizeof(cl_device_id), &ocl->device, NULL);
	if (CL_SUCCESS != err)
	{
		LogError("Error: clGetContextInfo() to get list of devices returned %s.\n", TranslateOpenCLError(err));
		return err;
	}

	// Read the OpenCL platform's version and the device OpenCL and OpenCL C versions
	GetPlatformAndDeviceVersion(platformId, ocl);

	// Create command queue.
	// OpenCL kernels are enqueued for execution to a particular device through special objects called command queues.
	// Command queue guarantees some ordering between calls and other OpenCL commands.
	// Here you create a simple in-order OpenCL command queue that doesn't allow execution of two kernels in parallel on a target device.
#ifdef CL_VERSION_2_0
	if (OPENCL_VERSION_2_0 == ocl->deviceVersion)
	{
		const cl_command_queue_properties properties[] = { CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0 };
		ocl->commandQueue = clCreateCommandQueueWithProperties(ocl->context, ocl->device, properties, &err);
	}
	else {
		// default behavior: OpenCL 1.2
		cl_command_queue_properties properties = CL_QUEUE_PROFILING_ENABLE;
		ocl->commandQueue = clCreateCommandQueue(ocl->context, ocl->device, properties, &err);
	}
#else
	// default behavior: OpenCL 1.2
	cl_command_queue_properties properties = CL_QUEUE_PROFILING_ENABLE;
	ocl->commandQueue = clCreateCommandQueue(ocl->context, ocl->device, properties, &err);
#endif
	if (CL_SUCCESS != err)
	{
		LogError("Error: clCreateCommandQueue() returned %s.\n", TranslateOpenCLError(err));
		return err;
	}

	return CL_SUCCESS;
}

int CreateAndBuildProgram(ocl_args_d_t* ocl)
{
	cl_int err = CL_SUCCESS;

	// Upload the OpenCL C source code from the input file to source
	// The size of the C program is returned in sourceSize
	char* source = NULL;
	size_t src_size = 0;
	err = ReadSourceFromFile("Template.cl", &source, &src_size);
	if (CL_SUCCESS != err)
	{
		LogError("Error: ReadSourceFromFile returned %s.\n", TranslateOpenCLError(err));
		goto Finish;
	}

	// And now after you obtained a regular C string call clCreateProgramWithSource to create OpenCL program object.
	ocl->program = clCreateProgramWithSource(ocl->context, 1, (const char**)&source, &src_size, &err);
	if (CL_SUCCESS != err)
	{
		LogError("Error: clCreateProgramWithSource returned %s.\n", TranslateOpenCLError(err));
		goto Finish;
	}

	// Build the program
	// During creation a program is not built. You need to explicitly call build function.
	// Here you just use create-build sequence,
	// but there are also other possibilities when program consist of several parts,
	// some of which are libraries, and you may want to consider using clCompileProgram and clLinkProgram as
	// alternatives.
	err = clBuildProgram(ocl->program, 1, &ocl->device, "", NULL, NULL);
	if (CL_SUCCESS != err)
	{
		LogError("Error: clBuildProgram() for source program returned %s.\n", TranslateOpenCLError(err));

		// In case of error print the build log to the standard output
		// First check the size of the log
		// Then allocate the memory and obtain the log from the program
		if (err == CL_BUILD_PROGRAM_FAILURE)
		{
			size_t log_size = 0;
			clGetProgramBuildInfo(ocl->program, ocl->device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);

			std::vector<char> build_log(log_size);
			clGetProgramBuildInfo(ocl->program, ocl->device, CL_PROGRAM_BUILD_LOG, log_size, &build_log[0], NULL);

			LogError("Error happened during the build of OpenCL program.\nBuild log:%s", &build_log[0]);
		}
	}

Finish:
	if (source)
	{
		delete[] source;
		source = NULL;
	}

	return err;
}

//Create OpenCL buffers from host memory. These buffers will be used later by the OpenCL kernel. change here
int CreateBufferArguments(ocl_args_d_t* ocl, cl_int* inputA, cl_int* outputB, cl_int* outputC, cl_int* outputD, cl_int* outputE, cl_uint arrayWidth, cl_uint arrayHeight)
{
	cl_int err = CL_SUCCESS;

	cl_image_format format;
	cl_image_desc desc;

	// Define the image data-type and order -
	// one channel (R) with unit values
	format.image_channel_data_type = CL_UNSIGNED_INT32;
	format.image_channel_order = CL_R;

	// Define the image properties (descriptor)
	desc.image_type = CL_MEM_OBJECT_IMAGE2D;
	desc.image_width = arrayWidth;
	desc.image_height = arrayHeight;
	desc.image_depth = 0;
	desc.image_array_size = 1;
	desc.image_row_pitch = 0;
	desc.image_slice_pitch = 0;
	desc.num_mip_levels = 0;
	desc.num_samples = 0;
#ifdef CL_VERSION_2_0
	desc.mem_object = NULL;
#else
	desc.buffer = NULL;
#endif

	// Create first image based on host memory inputA
	ocl->srcA = clCreateImage(ocl->context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, &format, &desc, inputA, &err);
	if (CL_SUCCESS != err)
	{
		LogError("Error: clCreateImage for srcA returned %s\n", TranslateOpenCLError(err));
		return err;
	}

	// Create second image based on host memory inputB
	/*ocl->srcB = clCreateImage(ocl->context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, &format, &desc, inputB, &err);
	if (CL_SUCCESS != err)
	{
		LogError("Error: clCreateImage for srcB returned %s\n", TranslateOpenCLError(err));
		return err;
	}*/

	// Create third (output) image based on host memory outputB
	ocl->dstMemA = clCreateImage(ocl->context, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR, &format, &desc, outputC, &err);
	if (CL_SUCCESS != err)
	{
		LogError("Error: clCreateImage for dstMem returned %s\n", TranslateOpenCLError(err));
		return err;
	}

	// Create third (output) image based on host memory outputC
	ocl->dstMemB = clCreateImage(ocl->context, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR, &format, &desc, outputC, &err);
	if (CL_SUCCESS != err)
	{
		LogError("Error: clCreateImage for dstMem returned %s\n", TranslateOpenCLError(err));
		return err;
	}

	// Create third (output) image based on host memory outputD
	ocl->dstMemC = clCreateImage(ocl->context, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR, &format, &desc, outputC, &err);
	if (CL_SUCCESS != err)
	{
		LogError("Error: clCreateImage for dstMem returned %s\n", TranslateOpenCLError(err));
		return err;
	}

	// Create third (output) image based on host memory outputE
	ocl->dstMemD = clCreateImage(ocl->context, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR, &format, &desc, outputC, &err);
	if (CL_SUCCESS != err)
	{
		LogError("Error: clCreateImage for dstMem returned %s\n", TranslateOpenCLError(err));
		return err;
	}

	return CL_SUCCESS;
}

// change here
cl_uint SetKernelArguments(ocl_args_d_t* ocl, int ix, int iy, int choice)
{
	cl_int err = CL_SUCCESS;

	/*err = clSetKernelArg(ocl->kernel, 0, sizeof(cl_mem), (void*)&ocl->srcA);
	if (CL_SUCCESS != err)
	{
		LogError("error: Failed to set argument srcA, returned %s\n", TranslateOpenCLError(err));
		return err;
	}*/

	/*err = clSetKernelArg(ocl->kernel, 1, sizeof(cl_mem), (void*)&ocl->srcB);
	if (CL_SUCCESS != err)
	{
		LogError("Error: Failed to set argument srcB, returned %s\n", TranslateOpenCLError(err));
		return err;
	}*/

	switch (choice) {
	case 1: {
		err = clSetKernelArg(ocl->kernel1, 0, sizeof(cl_mem), (void*)&ocl->srcA);
		if (CL_SUCCESS != err)
		{
			LogError("error: Failed to set argument srcA kernel1, returned %s\n", TranslateOpenCLError(err));
			return err;
		}

		err = clSetKernelArg(ocl->kernel1, 1, sizeof(cl_mem), (void*)&ocl->dstMemA);
		if (CL_SUCCESS != err)
		{
			LogError("Error: Failed to set argument dstMemA kernel1, returned %s\n", TranslateOpenCLError(err));
			return err;
		}

		err = clSetKernelArg(ocl->kernel1, 2, sizeof(cl_mem), &ix);
		if (CL_SUCCESS != err)
		{
			LogError("Error: Failed to set argument ix, returned %s\n", TranslateOpenCLError(err));
			return err;
		}

		err = clSetKernelArg(ocl->kernel1, 3, sizeof(cl_mem), &iy);
		if (CL_SUCCESS != err)
		{
			LogError("Error: Failed to set argument iy, returned %s\n", TranslateOpenCLError(err));
			return err;
		}
	}
		  break;
	case 2: {
		err = clSetKernelArg(ocl->kernel2, 0, sizeof(cl_mem), (void*)&ocl->srcA);
		if (CL_SUCCESS != err)
		{
			LogError("error: Failed to set argument srcA kernel2, returned %s\n", TranslateOpenCLError(err));
			return err;
		}

		err = clSetKernelArg(ocl->kernel2, 1, sizeof(cl_mem), (void*)&ocl->dstMemB);
		if (CL_SUCCESS != err)
		{
			LogError("Error: Failed to set argument dstMemB, returned %s\n", TranslateOpenCLError(err));
			return err;
		}

		err = clSetKernelArg(ocl->kernel2, 2, sizeof(cl_mem), &ix);
		if (CL_SUCCESS != err)
		{
			LogError("Error: Failed to set argument ix, returned %s\n", TranslateOpenCLError(err));
			return err;
		}

		err = clSetKernelArg(ocl->kernel2, 3, sizeof(cl_mem), &iy);
		if (CL_SUCCESS != err)
		{
			LogError("Error: Failed to set argument iy, returned %s\n", TranslateOpenCLError(err));
			return err;
		}
	}
		  break;
	case 3: {
		err = clSetKernelArg(ocl->kernel3, 0, sizeof(cl_mem), (void*)&ocl->srcA);
		if (CL_SUCCESS != err)
		{
			LogError("error: Failed to set argument srcA kernel3, returned %s\n", TranslateOpenCLError(err));
			return err;
		}

		err = clSetKernelArg(ocl->kernel3, 1, sizeof(cl_mem), (void*)&ocl->dstMemC);
		if (CL_SUCCESS != err)
		{
			LogError("Error: Failed to set argument dstMemC, returned %s\n", TranslateOpenCLError(err));
			return err;
		}

		err = clSetKernelArg(ocl->kernel3, 2, sizeof(cl_mem), &ix);
		if (CL_SUCCESS != err)
		{
			LogError("Error: Failed to set argument ix, returned %s\n", TranslateOpenCLError(err));
			return err;
		}

		err = clSetKernelArg(ocl->kernel3, 3, sizeof(cl_mem), &iy);
		if (CL_SUCCESS != err)
		{
			LogError("Error: Failed to set argument iy, returned %s\n", TranslateOpenCLError(err));
			return err;
		}
	}
		  break;
	case 4: {
		err = clSetKernelArg(ocl->kernel4, 0, sizeof(cl_mem), (void*)&ocl->srcA);
		if (CL_SUCCESS != err)
		{
			LogError("error: Failed to set argument srcA kernel4, returned %s\n", TranslateOpenCLError(err));
			return err;
		}

		err = clSetKernelArg(ocl->kernel4, 1, sizeof(cl_mem), (void*)&ocl->dstMemD);
		if (CL_SUCCESS != err)
		{
			LogError("Error: Failed to set argument dstMemD, returned %s\n", TranslateOpenCLError(err));
			return err;
		}

		err = clSetKernelArg(ocl->kernel4, 2, sizeof(cl_mem), &ix);
		if (CL_SUCCESS != err)
		{
			LogError("Error: Failed to set argument ix, returned %s\n", TranslateOpenCLError(err));
			return err;
		}

		err = clSetKernelArg(ocl->kernel4, 3, sizeof(cl_mem), &iy);
		if (CL_SUCCESS != err)
		{
			LogError("Error: Failed to set argument iy, returned %s\n", TranslateOpenCLError(err));
			return err;
		}
	}
		  break;
	}//end switch

	return err;
}

//same as before
cl_uint ExecuteMoveKernel(ocl_args_d_t* ocl, cl_uint width, cl_uint height, int choice)
{
	cl_int err = CL_SUCCESS;

	// Define global iteration space for clEnqueueNDRangeKernel.
	size_t globalWorkSize[2] = { width, height };


	// execute kernel
	/*err = clEnqueueNDRangeKernel(ocl->commandQueue, ocl->kernel, 2, NULL, globalWorkSize, NULL, 0, NULL, NULL);
	if (CL_SUCCESS != err)
	{
		LogError("Error: Failed to run kernel, return %s\n", TranslateOpenCLError(err));
		return err;
	}*/

	switch (choice) {
	case 1: {
		err = clEnqueueNDRangeKernel(ocl->commandQueue, ocl->kernel1, 2, NULL, globalWorkSize, NULL, 0, NULL, NULL);
		if (CL_SUCCESS != err)
		{
			LogError("Error: Failed to run kernel1, return %s\n", TranslateOpenCLError(err));
			return err;
		}
	}
		  break;
	case 2: {
		err = clEnqueueNDRangeKernel(ocl->commandQueue, ocl->kernel2, 2, NULL, globalWorkSize, NULL, 0, NULL, NULL);
		if (CL_SUCCESS != err)
		{
			LogError("Error: Failed to run kernel2, return %s\n", TranslateOpenCLError(err));
			return err;
		}
	}
		  break;
	case 3: {
		err = clEnqueueNDRangeKernel(ocl->commandQueue, ocl->kernel3, 2, NULL, globalWorkSize, NULL, 0, NULL, NULL);
		if (CL_SUCCESS != err)
		{
			LogError("Error: Failed to run kernel3, return %s\n", TranslateOpenCLError(err));
			return err;
		}
	}
		  break;
	case 4: {
		err = clEnqueueNDRangeKernel(ocl->commandQueue, ocl->kernel4, 2, NULL, globalWorkSize, NULL, 0, NULL, NULL);
		if (CL_SUCCESS != err)
		{
			LogError("Error: Failed to run kernel4, return %s\n", TranslateOpenCLError(err));
			return err;
		}
	}
		  break;
	}//end switch

	// Wait until the queued kernel is completed by the device
	err = clFinish(ocl->commandQueue);
	if (CL_SUCCESS != err)
	{
		LogError("Error: clFinish return %s\n", TranslateOpenCLError(err));
		return err;
	}

	return CL_SUCCESS;
}

//"Read" the result buffer (mapping the buffer to the host memory address). change here
// cx?[] x++
bool ReadAndVerify1(ocl_args_d_t* ocl, cl_uint width, cl_uint height, cl_int* inputA, int* cx6)
{
	cl_int err = CL_SUCCESS;
	bool result = true;

	// Enqueue a command to map the buffer object (ocl->dstMem) into the host address space and returns a pointer to it
	// The map operation is blocking
	size_t origin[] = { 0, 0, 0 };
	size_t region[] = { width, height, 1 };
	size_t image_row_pitch;
	size_t image_slice_pitch;
	cl_int* resultPtrA = (cl_int*)clEnqueueMapImage(ocl->commandQueue, ocl->dstMemA, true, CL_MAP_READ, origin, region, &image_row_pitch, &image_slice_pitch, 0, NULL, NULL, &err);
	//cl_int* resultPtrB = (cl_int*)clEnqueueMapImage(ocl->commandQueue, ocl->dstMemB, true, CL_MAP_READ, origin, region, &image_row_pitch, &image_slice_pitch, 0, NULL, NULL, &err);
	//cl_int* resultPtrC = (cl_int*)clEnqueueMapImage(ocl->commandQueue, ocl->dstMemC, true, CL_MAP_READ, origin, region, &image_row_pitch, &image_slice_pitch, 0, NULL, NULL, &err);

	if (CL_SUCCESS != err)
	{
		LogError("Error: clEnqueueMapBuffer returned %s\n", TranslateOpenCLError(err));
		return false;
	}

	// Call clFinish to guarantee that output region is updated
	err = clFinish(ocl->commandQueue);
	if (CL_SUCCESS != err)
	{
		LogError("Error: clFinish returned %s\n", TranslateOpenCLError(err));
	}

	// We mapped dstMem to resultPtr, so resultPtr is ready and includes the kernel output !!!
	// Verify the results
	unsigned int size = width * height;
	for (unsigned int k = 0; k < size; ++k)
	{
		cx6[k] = resultPtrA[k];
		//cy6[k] = resultPtrB[k];
		//ccy6[k] = resultPtrC[k];
	}

	// Unmapped the output buffer before releasing it
	err = clEnqueueUnmapMemObject(ocl->commandQueue, ocl->dstMemA, resultPtrA, 0, NULL, NULL);
	if (CL_SUCCESS != err)
	{
		LogError("Error: clEnqueueUnmapMemObject returned %s\n", TranslateOpenCLError(err));
	}
	/*err = clEnqueueUnmapMemObject(ocl->commandQueue, ocl->dstMemB, resultPtrB, 0, NULL, NULL);
	if (CL_SUCCESS != err)
	{
		LogError("Error: clEnqueueUnmapMemObject returned %s\n", TranslateOpenCLError(err));
	}
	err = clEnqueueUnmapMemObject(ocl->commandQueue, ocl->dstMemC, resultPtrC, 0, NULL, NULL);
	if (CL_SUCCESS != err)
	{
		LogError("Error: clEnqueueUnmapMemObject returned %s\n", TranslateOpenCLError(err));
	}*/

	return result;
}

// cy?[] y++
bool ReadAndVerify2(ocl_args_d_t* ocl, cl_uint width, cl_uint height, cl_int* inputA, int* cy6)
{
	cl_int err = CL_SUCCESS;
	bool result = true;

	// Enqueue a command to map the buffer object (ocl->dstMem) into the host address space and returns a pointer to it
	// The map operation is blocking
	size_t origin[] = { 0, 0, 0 };
	size_t region[] = { width, height, 1 };
	size_t image_row_pitch;
	size_t image_slice_pitch;
	//cl_int* resultPtrA = (cl_int*)clEnqueueMapImage(ocl->commandQueue, ocl->dstMemA, true, CL_MAP_READ, origin, region, &image_row_pitch, &image_slice_pitch, 0, NULL, NULL, &err);
	cl_int* resultPtrB = (cl_int*)clEnqueueMapImage(ocl->commandQueue, ocl->dstMemB, true, CL_MAP_READ, origin, region, &image_row_pitch, &image_slice_pitch, 0, NULL, NULL, &err);
	//cl_int* resultPtrC = (cl_int*)clEnqueueMapImage(ocl->commandQueue, ocl->dstMemC, true, CL_MAP_READ, origin, region, &image_row_pitch, &image_slice_pitch, 0, NULL, NULL, &err);

	if (CL_SUCCESS != err)
	{
		LogError("Error: clEnqueueMapBuffer returned %s\n", TranslateOpenCLError(err));
		return false;
	}

	// Call clFinish to guarantee that output region is updated
	err = clFinish(ocl->commandQueue);
	if (CL_SUCCESS != err)
	{
		LogError("Error: clFinish returned %s\n", TranslateOpenCLError(err));
	}

	// We mapped dstMem to resultPtr, so resultPtr is ready and includes the kernel output !!!
	// Verify the results
	unsigned int size = width * height;
	for (unsigned int k = 0; k < size; ++k)
	{
		//cx6[k] = resultPtrA[k];
		cy6[k] = resultPtrB[k];
		//ccy6[k] = resultPtrC[k];
	}

	// Unmapped the output buffer before releasing it
	/*err = clEnqueueUnmapMemObject(ocl->commandQueue, ocl->dstMemA, resultPtrA, 0, NULL, NULL);
	if (CL_SUCCESS != err)
	{
		LogError("Error: clEnqueueUnmapMemObject returned %s\n", TranslateOpenCLError(err));
	}*/
	err = clEnqueueUnmapMemObject(ocl->commandQueue, ocl->dstMemB, resultPtrB, 0, NULL, NULL);
	if (CL_SUCCESS != err)
	{
		LogError("Error: clEnqueueUnmapMemObject returned %s\n", TranslateOpenCLError(err));
	}
	/*err = clEnqueueUnmapMemObject(ocl->commandQueue, ocl->dstMemC, resultPtrC, 0, NULL, NULL);
	if (CL_SUCCESS != err)
	{
		LogError("Error: clEnqueueUnmapMemObject returned %s\n", TranslateOpenCLError(err));
	}*/

	return result;
}

// ccy?[] y--
bool ReadAndVerify3(ocl_args_d_t* ocl, cl_uint width, cl_uint height, cl_int* inputA, int* ccy6)
{
	cl_int err = CL_SUCCESS;
	bool result = true;

	// Enqueue a command to map the buffer object (ocl->dstMem) into the host address space and returns a pointer to it
	// The map operation is blocking
	size_t origin[] = { 0, 0, 0 };
	size_t region[] = { width, height, 1 };
	size_t image_row_pitch;
	size_t image_slice_pitch;
	//cl_int* resultPtrA = (cl_int*)clEnqueueMapImage(ocl->commandQueue, ocl->dstMemA, true, CL_MAP_READ, origin, region, &image_row_pitch, &image_slice_pitch, 0, NULL, NULL, &err);
	//cl_int* resultPtrB = (cl_int*)clEnqueueMapImage(ocl->commandQueue, ocl->dstMemB, true, CL_MAP_READ, origin, region, &image_row_pitch, &image_slice_pitch, 0, NULL, NULL, &err);
	cl_int* resultPtrC = (cl_int*)clEnqueueMapImage(ocl->commandQueue, ocl->dstMemC, true, CL_MAP_READ, origin, region, &image_row_pitch, &image_slice_pitch, 0, NULL, NULL, &err);

	if (CL_SUCCESS != err)
	{
		LogError("Error: clEnqueueMapBuffer returned %s\n", TranslateOpenCLError(err));
		return false;
	}

	// Call clFinish to guarantee that output region is updated
	err = clFinish(ocl->commandQueue);
	if (CL_SUCCESS != err)
	{
		LogError("Error: clFinish returned %s\n", TranslateOpenCLError(err));
	}

	// We mapped dstMem to resultPtr, so resultPtr is ready and includes the kernel output !!!
	// Verify the results
	unsigned int size = width * height;
	for (unsigned int k = 0; k < size; ++k)
	{
		//cx6[k] = resultPtrA[k];
		//cy6[k] = resultPtrB[k];
		ccy6[k] = resultPtrC[k];
	}

	// Unmapped the output buffer before releasing it
	/*err = clEnqueueUnmapMemObject(ocl->commandQueue, ocl->dstMemA, resultPtrA, 0, NULL, NULL);
	if (CL_SUCCESS != err)
	{
		LogError("Error: clEnqueueUnmapMemObject returned %s\n", TranslateOpenCLError(err));
	}
	err = clEnqueueUnmapMemObject(ocl->commandQueue, ocl->dstMemB, resultPtrB, 0, NULL, NULL);
	if (CL_SUCCESS != err)
	{
		LogError("Error: clEnqueueUnmapMemObject returned %s\n", TranslateOpenCLError(err));
	}*/
	err = clEnqueueUnmapMemObject(ocl->commandQueue, ocl->dstMemC, resultPtrC, 0, NULL, NULL);
	if (CL_SUCCESS != err)
	{
		LogError("Error: clEnqueueUnmapMemObject returned %s\n", TranslateOpenCLError(err));
	}

	return result;
}

// ccx?[] x--
bool ReadAndVerify4(ocl_args_d_t* ocl, cl_uint width, cl_uint height, cl_int* inputA, int* ccx6)
{
	cl_int err = CL_SUCCESS;
	bool result = true;

	// Enqueue a command to map the buffer object (ocl->dstMem) into the host address space and returns a pointer to it
	// The map operation is blocking
	size_t origin[] = { 0, 0, 0 };
	size_t region[] = { width, height, 1 };
	size_t image_row_pitch;
	size_t image_slice_pitch;
	//cl_int* resultPtrA = (cl_int*)clEnqueueMapImage(ocl->commandQueue, ocl->dstMemA, true, CL_MAP_READ, origin, region, &image_row_pitch, &image_slice_pitch, 0, NULL, NULL, &err);
	//cl_int* resultPtrB = (cl_int*)clEnqueueMapImage(ocl->commandQueue, ocl->dstMemB, true, CL_MAP_READ, origin, region, &image_row_pitch, &image_slice_pitch, 0, NULL, NULL, &err);
	//cl_int* resultPtrC = (cl_int*)clEnqueueMapImage(ocl->commandQueue, ocl->dstMemC, true, CL_MAP_READ, origin, region, &image_row_pitch, &image_slice_pitch, 0, NULL, NULL, &err);
	cl_int* resultPtrD = (cl_int*)clEnqueueMapImage(ocl->commandQueue, ocl->dstMemD, true, CL_MAP_READ, origin, region, &image_row_pitch, &image_slice_pitch, 0, NULL, NULL, &err);

	if (CL_SUCCESS != err)
	{
		LogError("Error: clEnqueueMapBuffer returned %s\n", TranslateOpenCLError(err));
		return false;
	}

	// Call clFinish to guarantee that output region is updated
	err = clFinish(ocl->commandQueue);
	if (CL_SUCCESS != err)
	{
		LogError("Error: clFinish returned %s\n", TranslateOpenCLError(err));
	}

	// We mapped dstMem to resultPtr, so resultPtr is ready and includes the kernel output !!!
	// Verify the results
	unsigned int size = width * height;
	for (unsigned int k = 0; k < size; ++k)
	{
		//cx6[k] = resultPtrA[k];
		//cy6[k] = resultPtrB[k];
		//ccy6[k] = resultPtrC[k];
		ccx6[k] = resultPtrD[k];
	}

	// Unmapped the output buffer before releasing it
	err = clEnqueueUnmapMemObject(ocl->commandQueue, ocl->dstMemD, resultPtrD, 0, NULL, NULL);
	if (CL_SUCCESS != err)
	{
		LogError("Error: clEnqueueUnmapMemObject returned %s\n", TranslateOpenCLError(err));
	}

	return result;
}

//inject() Wrapup() moveInGPU() cx6 x++ cy6 y++ ccx6 x-- ccy6 y-- m6[]
int Wrapup(int ix, int iy) {
	cl_int err;
	ocl_args_d_t ocl;
	cl_device_type deviceType = CL_DEVICE_TYPE_GPU;
	//cl_device_type deviceType = CL_DEVICE_TYPE_CPU;

	/*LARGE_INTEGER perfFrequency;
	LARGE_INTEGER performanceCountNDRangeStart;
	LARGE_INTEGER performanceCountNDRangeStop;*/

	cl_uint arrayWidth = 16;
	cl_uint arrayHeight = 16;

	//initialize Open CL objects (context, queue, etc.)
	if (CL_SUCCESS != SetupOpenCL(&ocl, deviceType))
	{
		return -1;
	}

	// allocate working buffers. 
	// the buffer should be aligned with 4K page and size should fit 64-byte cached line
	cl_uint optimizedSize = ((sizeof(cl_int) * arrayWidth * arrayHeight - 1) / 64 + 1) * 64;
	cl_int* inputA = (cl_int*)_aligned_malloc(optimizedSize, 4096);
	cl_int* outputB = (cl_int*)_aligned_malloc(optimizedSize, 4096);
	cl_int* outputC = (cl_int*)_aligned_malloc(optimizedSize, 4096);
	cl_int* outputD = (cl_int*)_aligned_malloc(optimizedSize, 4096);
	cl_int* outputE = (cl_int*)_aligned_malloc(optimizedSize, 4096);
	if (NULL == inputA || NULL == outputB || NULL == outputC || NULL == outputD || NULL == outputE)
	{
		LogError("Error: _aligned_malloc failed to allocate buffers.\n");
		return -1;
	}

	//input
	generateInput(inputA, arrayWidth, arrayHeight);
	////generateInput(inputB, arrayWidth, arrayHeight);

	// Create OpenCL buffers from host memory
	// These buffers will be used later by the OpenCL kernel
	if (CL_SUCCESS != CreateBufferArguments(&ocl, inputA, outputB, outputC, outputD, outputE, arrayWidth, arrayHeight))
	{
		return -1;
	}

	// Create and build the OpenCL program
	if (CL_SUCCESS != CreateAndBuildProgram(&ocl))
	{
		return -1;
	}

	// Program consists of kernels.
	// Each kernel can be called (enqueued) from the host part of OpenCL application.
	// To call the kernel, you need to create it from existing program.
	ocl.kernel1 = clCreateKernel(ocl.program, "moveA", &err);
	if (CL_SUCCESS != err)
	{
		LogError("Error: clCreateKernel 1 returned %s\n", TranslateOpenCLError(err));
		return -1;
	}
	ocl.kernel2 = clCreateKernel(ocl.program, "moveB", &err);
	if (CL_SUCCESS != err)
	{
		LogError("Error: clCreateKernel 2 returned %s\n", TranslateOpenCLError(err));
		return -1;
	}
	ocl.kernel3 = clCreateKernel(ocl.program, "moveC", &err);
	if (CL_SUCCESS != err)
	{
		LogError("Error: clCreateKernel 3 returned %s\n", TranslateOpenCLError(err));
		return -1;
	}
	ocl.kernel4 = clCreateKernel(ocl.program, "moveD", &err);
	if (CL_SUCCESS != err)
	{
		LogError("Error: clCreateKernel 4 returned %s\n", TranslateOpenCLError(err));
		return -1;
	}

	int choice = 1;
	// Passing arguments into OpenCL kernel.
	if (CL_SUCCESS != SetKernelArguments(&ocl, ix, iy, choice))
	{
		return -1;
	}
	if (CL_SUCCESS != ExecuteMoveKernel(&ocl, arrayWidth, arrayHeight, choice))
	{
		return -1;
	}

	choice = 2;
	// Passing arguments into OpenCL kernel.
	if (CL_SUCCESS != SetKernelArguments(&ocl, ix, iy, choice))
	{
		return -1;
	}
	if (CL_SUCCESS != ExecuteMoveKernel(&ocl, arrayWidth, arrayHeight, choice))
	{
		return -1;
	}

	choice = 3;
	// Passing arguments into OpenCL kernel.
	if (CL_SUCCESS != SetKernelArguments(&ocl, ix, iy, choice))
	{
		return -1;
	}
	if (CL_SUCCESS != ExecuteMoveKernel(&ocl, arrayWidth, arrayHeight, choice))
	{
		return -1;
	}

	choice = 4;
	// Passing arguments into OpenCL kernel.
	if (CL_SUCCESS != SetKernelArguments(&ocl, ix, iy, choice))
	{
		return -1;
	}
	if (CL_SUCCESS != ExecuteMoveKernel(&ocl, arrayWidth, arrayHeight, choice))
	{
		return -1;
	}

	/*bool queueProfilingEnable = true;
	if (queueProfilingEnable)
		QueryPerformanceCounter(&performanceCountNDRangeStart);*/
		// Execute (enqueue) the kernel
		/*if (CL_SUCCESS != ExecuteMoveAKernel(&ocl, arrayWidth, arrayHeight, choice))
		{
			return -1;
		}*/
		/*if (queueProfilingEnable)
			QueryPerformanceCounter(&performanceCountNDRangeStop);*/

			// The last part of this function: getting processed results back.
			// use map-unmap sequence to update original memory area with output buffer.
	ReadAndVerify1(&ocl, arrayWidth, arrayHeight, inputA, cx6);
	ReadAndVerify2(&ocl, arrayWidth, arrayHeight, inputA, cy6);
	ReadAndVerify3(&ocl, arrayWidth, arrayHeight, inputA, ccy6);
	ReadAndVerify4(&ocl, arrayWidth, arrayHeight, inputA, ccx6);

	// retrieve performance counter frequency
	/*if (queueProfilingEnable)
	{
		QueryPerformanceFrequency(&perfFrequency);
		LogInfo("NDRange performance counter time %f ms.\n",
			1000.0f * (float)(performanceCountNDRangeStop.QuadPart - performanceCountNDRangeStart.QuadPart) / (float)perfFrequency.QuadPart);
	}*/

	_aligned_free(inputA);
	_aligned_free(outputB);
	_aligned_free(outputC);
	_aligned_free(outputD);
	_aligned_free(outputE);
}

//inject2() Wrapup2() moveRight() cx5 x++ cy5 y++ ccx5 x-- ccy5 y-- m5[]
int Wrapup2(int ix, int iy) {
	cl_int err;
	ocl_args_d_t ocl;
	cl_device_type deviceType = CL_DEVICE_TYPE_GPU;
	//cl_device_type deviceType = CL_DEVICE_TYPE_CPU;

	cl_uint arrayWidth = 16;
	cl_uint arrayHeight = 16;

	//initialize Open CL objects (context, queue, etc.)
	if (CL_SUCCESS != SetupOpenCL(&ocl, deviceType))
	{
		return -1;
	}

	// allocate working buffers. 
	// the buffer should be aligned with 4K page and size should fit 64-byte cached line
	cl_uint optimizedSize = ((sizeof(cl_int) * arrayWidth * arrayHeight - 1) / 64 + 1) * 64;
	cl_int* inputA = (cl_int*)_aligned_malloc(optimizedSize, 4096);
	cl_int* outputB = (cl_int*)_aligned_malloc(optimizedSize, 4096);
	cl_int* outputC = (cl_int*)_aligned_malloc(optimizedSize, 4096);
	cl_int* outputD = (cl_int*)_aligned_malloc(optimizedSize, 4096);
	cl_int* outputE = (cl_int*)_aligned_malloc(optimizedSize, 4096);
	if (NULL == inputA || NULL == outputB || NULL == outputC || NULL == outputD || NULL == outputE)
	{
		LogError("Error: _aligned_malloc failed to allocate buffers.\n");
		return -1;
	}

	//input
	generateInput(inputA, arrayWidth, arrayHeight);
	////generateInput(inputB, arrayWidth, arrayHeight);

	// Create OpenCL buffers from host memory
	// These buffers will be used later by the OpenCL kernel
	if (CL_SUCCESS != CreateBufferArguments(&ocl, inputA, outputB, outputC, outputD, outputE, arrayWidth, arrayHeight))
	{
		return -1;
	}

	// Create and build the OpenCL program
	if (CL_SUCCESS != CreateAndBuildProgram(&ocl))
	{
		return -1;
	}

	// Program consists of kernels.
	// Each kernel can be called (enqueued) from the host part of OpenCL application.
	// To call the kernel, you need to create it from existing program.
	ocl.kernel1 = clCreateKernel(ocl.program, "moveA", &err);
	if (CL_SUCCESS != err)
	{
		LogError("Error: clCreateKernel 1 returned %s\n", TranslateOpenCLError(err));
		return -1;
	}
	ocl.kernel2 = clCreateKernel(ocl.program, "moveB", &err);
	if (CL_SUCCESS != err)
	{
		LogError("Error: clCreateKernel 2 returned %s\n", TranslateOpenCLError(err));
		return -1;
	}
	ocl.kernel3 = clCreateKernel(ocl.program, "moveC", &err);
	if (CL_SUCCESS != err)
	{
		LogError("Error: clCreateKernel 3 returned %s\n", TranslateOpenCLError(err));
		return -1;
	}
	ocl.kernel4 = clCreateKernel(ocl.program, "moveD", &err);
	if (CL_SUCCESS != err)
	{
		LogError("Error: clCreateKernel 4 returned %s\n", TranslateOpenCLError(err));
		return -1;
	}

	int choice = 1;
	// Passing arguments into OpenCL kernel.
	if (CL_SUCCESS != SetKernelArguments(&ocl, ix, iy, choice))
	{
		return -1;
	}
	if (CL_SUCCESS != ExecuteMoveKernel(&ocl, arrayWidth, arrayHeight, choice))
	{
		return -1;
	}

	choice = 2;
	// Passing arguments into OpenCL kernel.
	if (CL_SUCCESS != SetKernelArguments(&ocl, ix, iy, choice))
	{
		return -1;
	}
	if (CL_SUCCESS != ExecuteMoveKernel(&ocl, arrayWidth, arrayHeight, choice))
	{
		return -1;
	}

	choice = 3;
	// Passing arguments into OpenCL kernel.
	if (CL_SUCCESS != SetKernelArguments(&ocl, ix, iy, choice))
	{
		return -1;
	}
	if (CL_SUCCESS != ExecuteMoveKernel(&ocl, arrayWidth, arrayHeight, choice))
	{
		return -1;
	}

	choice = 4;
	// Passing arguments into OpenCL kernel.
	if (CL_SUCCESS != SetKernelArguments(&ocl, ix, iy, choice))
	{
		return -1;
	}
	if (CL_SUCCESS != ExecuteMoveKernel(&ocl, arrayWidth, arrayHeight, choice))
	{
		return -1;
	}

			// The last part of this function: getting processed results back.
			// use map-unmap sequence to update original memory area with output buffer.
	ReadAndVerify1(&ocl, arrayWidth, arrayHeight, inputA, cx5);
	ReadAndVerify2(&ocl, arrayWidth, arrayHeight, inputA, cy5);
	ReadAndVerify3(&ocl, arrayWidth, arrayHeight, inputA, ccy5);
	ReadAndVerify4(&ocl, arrayWidth, arrayHeight, inputA, ccx5);

	_aligned_free(inputA);
	_aligned_free(outputB);
	_aligned_free(outputC);
	_aligned_free(outputD);
	_aligned_free(outputE);
}

//inject3() Wrapup3() moveLeft() cx4 x++ cy4 y++ ccx4 x-- ccy4 y-- m4[]
int Wrapup3(int ix, int iy) {
	cl_int err;
	ocl_args_d_t ocl;
	cl_device_type deviceType = CL_DEVICE_TYPE_GPU;
	//cl_device_type deviceType = CL_DEVICE_TYPE_CPU;

	cl_uint arrayWidth = 16;
	cl_uint arrayHeight = 16;

	//initialize Open CL objects (context, queue, etc.)
	if (CL_SUCCESS != SetupOpenCL(&ocl, deviceType))
	{
		return -1;
	}

	// allocate working buffers. 
	// the buffer should be aligned with 4K page and size should fit 64-byte cached line
	cl_uint optimizedSize = ((sizeof(cl_int) * arrayWidth * arrayHeight - 1) / 64 + 1) * 64;
	cl_int* inputA = (cl_int*)_aligned_malloc(optimizedSize, 4096);
	cl_int* outputB = (cl_int*)_aligned_malloc(optimizedSize, 4096);
	cl_int* outputC = (cl_int*)_aligned_malloc(optimizedSize, 4096);
	cl_int* outputD = (cl_int*)_aligned_malloc(optimizedSize, 4096);
	cl_int* outputE = (cl_int*)_aligned_malloc(optimizedSize, 4096);
	if (NULL == inputA || NULL == outputB || NULL == outputC || NULL == outputD || NULL == outputE)
	{
		LogError("Error: _aligned_malloc failed to allocate buffers.\n");
		return -1;
	}

	//input
	generateInput(inputA, arrayWidth, arrayHeight);
	////generateInput(inputB, arrayWidth, arrayHeight);

	// Create OpenCL buffers from host memory
	// These buffers will be used later by the OpenCL kernel
	if (CL_SUCCESS != CreateBufferArguments(&ocl, inputA, outputB, outputC, outputD, outputE, arrayWidth, arrayHeight))
	{
		return -1;
	}

	// Create and build the OpenCL program
	if (CL_SUCCESS != CreateAndBuildProgram(&ocl))
	{
		return -1;
	}

	// Program consists of kernels.
	// Each kernel can be called (enqueued) from the host part of OpenCL application.
	// To call the kernel, you need to create it from existing program.
	ocl.kernel1 = clCreateKernel(ocl.program, "moveA", &err);
	if (CL_SUCCESS != err)
	{
		LogError("Error: clCreateKernel 1 returned %s\n", TranslateOpenCLError(err));
		return -1;
	}
	ocl.kernel2 = clCreateKernel(ocl.program, "moveB", &err);
	if (CL_SUCCESS != err)
	{
		LogError("Error: clCreateKernel 2 returned %s\n", TranslateOpenCLError(err));
		return -1;
	}
	ocl.kernel3 = clCreateKernel(ocl.program, "moveC", &err);
	if (CL_SUCCESS != err)
	{
		LogError("Error: clCreateKernel 3 returned %s\n", TranslateOpenCLError(err));
		return -1;
	}
	ocl.kernel4 = clCreateKernel(ocl.program, "moveD", &err);
	if (CL_SUCCESS != err)
	{
		LogError("Error: clCreateKernel 4 returned %s\n", TranslateOpenCLError(err));
		return -1;
	}

	int choice = 1;
	// Passing arguments into OpenCL kernel.
	if (CL_SUCCESS != SetKernelArguments(&ocl, ix, iy, choice))
	{
		return -1;
	}
	if (CL_SUCCESS != ExecuteMoveKernel(&ocl, arrayWidth, arrayHeight, choice))
	{
		return -1;
	}

	choice = 2;
	// Passing arguments into OpenCL kernel.
	if (CL_SUCCESS != SetKernelArguments(&ocl, ix, iy, choice))
	{
		return -1;
	}
	if (CL_SUCCESS != ExecuteMoveKernel(&ocl, arrayWidth, arrayHeight, choice))
	{
		return -1;
	}

	choice = 3;
	// Passing arguments into OpenCL kernel.
	if (CL_SUCCESS != SetKernelArguments(&ocl, ix, iy, choice))
	{
		return -1;
	}
	if (CL_SUCCESS != ExecuteMoveKernel(&ocl, arrayWidth, arrayHeight, choice))
	{
		return -1;
	}

	choice = 4;
	// Passing arguments into OpenCL kernel.
	if (CL_SUCCESS != SetKernelArguments(&ocl, ix, iy, choice))
	{
		return -1;
	}
	if (CL_SUCCESS != ExecuteMoveKernel(&ocl, arrayWidth, arrayHeight, choice))
	{
		return -1;
	}

	// The last part of this function: getting processed results back.
	// use map-unmap sequence to update original memory area with output buffer.
	ReadAndVerify1(&ocl, arrayWidth, arrayHeight, inputA, cx4);
	ReadAndVerify2(&ocl, arrayWidth, arrayHeight, inputA, cy4);
	ReadAndVerify3(&ocl, arrayWidth, arrayHeight, inputA, ccy4);
	ReadAndVerify4(&ocl, arrayWidth, arrayHeight, inputA, ccx4);

	_aligned_free(inputA);
	_aligned_free(outputB);
	_aligned_free(outputC);
	_aligned_free(outputD);
	_aligned_free(outputE);
}
//image moveA to moveD
int wrapupNew(int ix, int iy, int* x, int* y, int* _x, int* _y) {
	cl_int err;
	ocl_args_d_t ocl;
	cl_device_type deviceType = CL_DEVICE_TYPE_GPU;
	//cl_device_type deviceType = CL_DEVICE_TYPE_CPU;

	cl_uint arrayWidth = 16;
	cl_uint arrayHeight = 16;

	//initialize Open CL objects (context, queue, etc.)
	if (CL_SUCCESS != SetupOpenCL(&ocl, deviceType))
	{
		return -1;
	}

	// allocate working buffers. 
	// the buffer should be aligned with 4K page and size should fit 64-byte cached line
	cl_uint optimizedSize = ((sizeof(cl_int) * arrayWidth * arrayHeight - 1) / 64 + 1) * 64;
	cl_int* inputA = (cl_int*)_aligned_malloc(optimizedSize, 4096);
	cl_int* outputB = (cl_int*)_aligned_malloc(optimizedSize, 4096);
	cl_int* outputC = (cl_int*)_aligned_malloc(optimizedSize, 4096);
	cl_int* outputD = (cl_int*)_aligned_malloc(optimizedSize, 4096);
	cl_int* outputE = (cl_int*)_aligned_malloc(optimizedSize, 4096);
	if (NULL == inputA || NULL == outputB || NULL == outputC || NULL == outputD || NULL == outputE)
	{
		LogError("Error: _aligned_malloc failed to allocate buffers.\n");
		return -1;
	}

	//input
	generateInput(inputA, arrayWidth, arrayHeight);
	////generateInput(inputB, arrayWidth, arrayHeight);

	// Create OpenCL buffers from host memory
	// These buffers will be used later by the OpenCL kernel
	if (CL_SUCCESS != CreateBufferArguments(&ocl, inputA, outputB, outputC, outputD, outputE, arrayWidth, arrayHeight))
	{
		return -1;
	}

	// Create and build the OpenCL program
	if (CL_SUCCESS != CreateAndBuildProgram(&ocl))
	{
		return -1;
	}

	// Program consists of kernels.
	// Each kernel can be called (enqueued) from the host part of OpenCL application.
	// To call the kernel, you need to create it from existing program.
	ocl.kernel1 = clCreateKernel(ocl.program, "moveA", &err);
	if (CL_SUCCESS != err)
	{
		LogError("Error: clCreateKernel 1 returned %s\n", TranslateOpenCLError(err));
		return -1;
	}
	ocl.kernel2 = clCreateKernel(ocl.program, "moveB", &err);
	if (CL_SUCCESS != err)
	{
		LogError("Error: clCreateKernel 2 returned %s\n", TranslateOpenCLError(err));
		return -1;
	}
	ocl.kernel3 = clCreateKernel(ocl.program, "moveC", &err);
	if (CL_SUCCESS != err)
	{
		LogError("Error: clCreateKernel 3 returned %s\n", TranslateOpenCLError(err));
		return -1;
	}
	ocl.kernel4 = clCreateKernel(ocl.program, "moveD", &err);
	if (CL_SUCCESS != err)
	{
		LogError("Error: clCreateKernel 4 returned %s\n", TranslateOpenCLError(err));
		return -1;
	}

	int choice = 1;
	// Passing arguments into OpenCL kernel.
	if (CL_SUCCESS != SetKernelArguments(&ocl, ix, iy, choice))
	{
		return -1;
	}
	if (CL_SUCCESS != ExecuteMoveKernel(&ocl, arrayWidth, arrayHeight, choice))
	{
		return -1;
	}

	choice = 2;
	// Passing arguments into OpenCL kernel.
	if (CL_SUCCESS != SetKernelArguments(&ocl, ix, iy, choice))
	{
		return -1;
	}
	if (CL_SUCCESS != ExecuteMoveKernel(&ocl, arrayWidth, arrayHeight, choice))
	{
		return -1;
	}

	choice = 3;
	// Passing arguments into OpenCL kernel.
	if (CL_SUCCESS != SetKernelArguments(&ocl, ix, iy, choice))
	{
		return -1;
	}
	if (CL_SUCCESS != ExecuteMoveKernel(&ocl, arrayWidth, arrayHeight, choice))
	{
		return -1;
	}

	choice = 4;
	// Passing arguments into OpenCL kernel.
	if (CL_SUCCESS != SetKernelArguments(&ocl, ix, iy, choice))
	{
		return -1;
	}
	if (CL_SUCCESS != ExecuteMoveKernel(&ocl, arrayWidth, arrayHeight, choice))
	{
		return -1;
	}

	// The last part of this function: getting processed results back.
	// use map-unmap sequence to update original memory area with output buffer.
	ReadAndVerify1(&ocl, arrayWidth, arrayHeight, inputA, x);
	ReadAndVerify2(&ocl, arrayWidth, arrayHeight, inputA, y);
	ReadAndVerify3(&ocl, arrayWidth, arrayHeight, inputA, _y);
	ReadAndVerify4(&ocl, arrayWidth, arrayHeight, inputA, _x);

	_aligned_free(inputA);
	_aligned_free(outputB);
	_aligned_free(outputC);
	_aligned_free(outputD);
	_aligned_free(outputE);
}

// kernel move256() run for all type device
void Wrapup4All(int ix, int iy, int* x, int* y, int* _x, int* _y) {
	int* A = NULL;
	int* C = NULL;  // Output array
	int* D = NULL;  // Output array
	int* E = NULL;  // Output array
	int* F = NULL;  // Output array

	const int elements = 256;
	size_t datasize = sizeof(int) * elements;
	A = (int*)malloc(datasize);
	C = (int*)malloc(datasize);
	D = (int*)malloc(datasize);
	E = (int*)malloc(datasize);
	F = (int*)malloc(datasize);
	for (int i = 0; i < elements; i++) {
		A[i] = i;
	}
	cl_int status;
	// STEP 1: Discover and initialize the platforms
	cl_uint numPlatforms = 0;
	cl_platform_id* platforms = NULL;
	status = clGetPlatformIDs(0, NULL, &numPlatforms);
	platforms = (cl_platform_id*)malloc(numPlatforms * sizeof(cl_platform_id));
	status = clGetPlatformIDs(numPlatforms, platforms, NULL);
	// STEP 2: Discover and initialize the devices
	cl_uint numDevices = 0;
	cl_device_id* devices = NULL;
	status = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_ALL, 0, NULL, &numDevices);
	devices = (cl_device_id*)malloc(numDevices * sizeof(cl_device_id));
	status = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_ALL, numDevices, devices, NULL);
	// STEP 3: Create a context
	cl_context context = NULL;
	context = clCreateContext(NULL, numDevices, devices, NULL, NULL, &status);
	// STEP 4: Create a command queue
	cl_command_queue cmdQueue;
	cmdQueue = clCreateCommandQueue(context, devices[0], 0, &status);
	// STEP 5: Create device buffers
	cl_mem bufferA;
	cl_mem bufferC;
	cl_mem bufferD;
	cl_mem bufferE;
	cl_mem bufferF;
	bufferA = clCreateBuffer(context, CL_MEM_READ_ONLY, datasize, NULL, &status);
	bufferC = clCreateBuffer(context, CL_MEM_WRITE_ONLY, datasize, NULL, &status);
	bufferD = clCreateBuffer(context, CL_MEM_WRITE_ONLY, datasize, NULL, &status);
	bufferE = clCreateBuffer(context, CL_MEM_WRITE_ONLY, datasize, NULL, &status);
	bufferF = clCreateBuffer(context, CL_MEM_WRITE_ONLY, datasize, NULL, &status);
	// STEP 6: Write host data to device buffers
	status = clEnqueueWriteBuffer(cmdQueue, bufferA, CL_FALSE, 0, datasize, A, 0, NULL, NULL);
	// STEP 7: Create and compile the program
	std::ifstream srcFile("Template.cl");
	checkErr(srcFile.is_open() ? CL_SUCCESS : -1, "reading Template.cl");
	std::string srcProg(
		std::istreambuf_iterator<char>(srcFile),
		(std::istreambuf_iterator<char>()));

	const char* src = srcProg.c_str();
	size_t length = srcProg.length();
	cl_int errNum;
	cl_program program = clCreateProgramWithSource(context, 1, &src, &length, &errNum);
	checkErr(errNum, "clCreateProgramWithSource");
	status = clBuildProgram(program, numDevices, devices, NULL, NULL, NULL);
	//cl_program program = clCreateProgramWithSource(context,1,(const char**)&programSource,NULL,&status);
	// STEP 8: Create the kernel
	cl_kernel kernel = NULL;
	kernel = clCreateKernel(program, "move256", &status);
	// STEP 9: Set the kernel arguments
	status = clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufferA);
	status |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufferC);
	status |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufferD);
	status |= clSetKernelArg(kernel, 3, sizeof(cl_mem), &bufferE);
	status |= clSetKernelArg(kernel, 4, sizeof(cl_mem), &bufferF);
	status |= clSetKernelArg(kernel, 5, sizeof(cl_mem), &ix);
	status |= clSetKernelArg(kernel, 6, sizeof(cl_mem), &iy);
	// STEP 10: Configure the work-item structure
	size_t globalWorkSize[1];
	globalWorkSize[0] = elements; // size
	// STEP 11: Enqueue the kernel for execution
	status = clEnqueueNDRangeKernel(cmdQueue, kernel, 1, NULL, globalWorkSize, NULL, 0, NULL, NULL);
	// STEP 12: Read the output buffer back to the host
	clEnqueueReadBuffer(cmdQueue, bufferC, CL_TRUE, 0, datasize, C, 0, NULL, NULL);
	clEnqueueReadBuffer(cmdQueue, bufferD, CL_TRUE, 0, datasize, D, 0, NULL, NULL);
	clEnqueueReadBuffer(cmdQueue, bufferE, CL_TRUE, 0, datasize, E, 0, NULL, NULL);
	clEnqueueReadBuffer(cmdQueue, bufferF, CL_TRUE, 0, datasize, F, 0, NULL, NULL);
	//Verify the output
   /*cout << "C[]=>>";
   for (int i = 0; i < elements; i++) {
	   cout << C[i] << " ";
   }
   cout << endl;
   cout << "D[]=>>";
   for (int i = 0; i < elements; i++) {
	   cout << D[i] << " ";
   }
   cout << endl;
   cout << "E[]=>>";
   for (int i = 0; i < elements; i++) {
	   cout << E[i] << " ";
   }
   cout << endl;
   cout << "F[]=>>";
   for (int i = 0; i < elements; i++) {
	   cout << F[i] << " ";
   }
   cout << endl;*/

	for (int k = 0; k < elements; ++k)
	{
		x[k] = C[k];
		y[k] = D[k];
		_x[k] = E[k];
		_y[k] = F[k];
	}

	// STEP 13: Release OpenCL resources
	clReleaseKernel(kernel);
	clReleaseProgram(program);
	clReleaseCommandQueue(cmdQueue);
	clReleaseMemObject(bufferA);
	clReleaseMemObject(bufferC);
	clReleaseMemObject(bufferD);
	clReleaseMemObject(bufferE);
	clReleaseMemObject(bufferF);
	clReleaseContext(context);
	free(A);
	free(C);
	free(D);
	free(E);
	free(F);
	free(platforms);
	free(devices);
}
// kernel move256() nvidia GPU platform[0]
void Wrapup4nvGPU(int ix, int iy, int* x, int* y, int* _x, int* _y) {
	int* A = NULL;
	int* C = NULL;  // Output array
	int* D = NULL;  // Output array
	int* E = NULL;  // Output array
	int* F = NULL;  // Output array

	const int elements = 256;
	size_t datasize = sizeof(int) * elements;
	A = (int*)malloc(datasize);
	C = (int*)malloc(datasize);
	D = (int*)malloc(datasize);
	E = (int*)malloc(datasize);
	F = (int*)malloc(datasize);
	for (int i = 0; i < elements; i++) {
		A[i] = i;
	}
	cl_int status;
	// STEP 1: Discover and initialize the platforms
	cl_uint numPlatforms = 0;
	cl_platform_id* platforms = NULL;
	status = clGetPlatformIDs(0, NULL, &numPlatforms);
	platforms = (cl_platform_id*)malloc(numPlatforms * sizeof(cl_platform_id));
	status = clGetPlatformIDs(numPlatforms, platforms, NULL);
	// STEP 2: Discover and initialize the devices
	cl_uint numDevices = 0;
	cl_device_id* devices = NULL;
	status = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, 0, NULL, &numDevices);
	devices = (cl_device_id*)malloc(numDevices * sizeof(cl_device_id));
	status = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, numDevices, devices, NULL);
	// STEP 3: Create a context
	cl_context context = NULL;
	context = clCreateContext(NULL, numDevices, devices, NULL, NULL, &status);
	// STEP 4: Create a command queue
	cl_command_queue cmdQueue;
	cmdQueue = clCreateCommandQueue(context, devices[0], 0, &status);
	// STEP 5: Create device buffers
	cl_mem bufferA;
	cl_mem bufferC;
	cl_mem bufferD;
	cl_mem bufferE;
	cl_mem bufferF;
	bufferA = clCreateBuffer(context, CL_MEM_READ_ONLY, datasize, NULL, &status);
	bufferC = clCreateBuffer(context, CL_MEM_WRITE_ONLY, datasize, NULL, &status);
	bufferD = clCreateBuffer(context, CL_MEM_WRITE_ONLY, datasize, NULL, &status);
	bufferE = clCreateBuffer(context, CL_MEM_WRITE_ONLY, datasize, NULL, &status);
	bufferF = clCreateBuffer(context, CL_MEM_WRITE_ONLY, datasize, NULL, &status);
	// STEP 6: Write host data to device buffers
	status = clEnqueueWriteBuffer(cmdQueue, bufferA, CL_FALSE, 0, datasize, A, 0, NULL, NULL);
	// STEP 7: Create and compile the program
	std::ifstream srcFile("Template.cl");
	checkErr(srcFile.is_open() ? CL_SUCCESS : -1, "reading Template.cl");
	std::string srcProg(
		std::istreambuf_iterator<char>(srcFile),
		(std::istreambuf_iterator<char>()));

	const char* src = srcProg.c_str();
	size_t length = srcProg.length();
	cl_int errNum;
	cl_program program = clCreateProgramWithSource(context, 1, &src, &length, &errNum);
	checkErr(errNum, "clCreateProgramWithSource");
	status = clBuildProgram(program, numDevices, devices, NULL, NULL, NULL);
	//cl_program program = clCreateProgramWithSource(context,1,(const char**)&programSource,NULL,&status);
	// STEP 8: Create the kernel
	cl_kernel kernel = NULL;
	kernel = clCreateKernel(program, "move256", &status);
	// STEP 9: Set the kernel arguments
	status = clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufferA);
	status |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufferC);
	status |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufferD);
	status |= clSetKernelArg(kernel, 3, sizeof(cl_mem), &bufferE);
	status |= clSetKernelArg(kernel, 4, sizeof(cl_mem), &bufferF);
	status |= clSetKernelArg(kernel, 5, sizeof(cl_mem), &ix);
	status |= clSetKernelArg(kernel, 6, sizeof(cl_mem), &iy);
	// STEP 10: Configure the work-item structure
	size_t globalWorkSize[1];
	globalWorkSize[0] = elements; // size
	// STEP 11: Enqueue the kernel for execution
	status = clEnqueueNDRangeKernel(cmdQueue, kernel, 1, NULL, globalWorkSize, NULL, 0, NULL, NULL);
	// STEP 12: Read the output buffer back to the host
	clEnqueueReadBuffer(cmdQueue, bufferC, CL_TRUE, 0, datasize, C, 0, NULL, NULL);
	clEnqueueReadBuffer(cmdQueue, bufferD, CL_TRUE, 0, datasize, D, 0, NULL, NULL);
	clEnqueueReadBuffer(cmdQueue, bufferE, CL_TRUE, 0, datasize, E, 0, NULL, NULL);
	clEnqueueReadBuffer(cmdQueue, bufferF, CL_TRUE, 0, datasize, F, 0, NULL, NULL);
	 //Verify the output
	/*cout << "C[]=>>";
	for (int i = 0; i < elements; i++) {
		cout << C[i] << " ";
	}
	cout << endl;
	cout << "D[]=>>";
	for (int i = 0; i < elements; i++) {
		cout << D[i] << " ";
	}
	cout << endl;
	cout << "E[]=>>";
	for (int i = 0; i < elements; i++) {
		cout << E[i] << " ";
	}
	cout << endl;
	cout << "F[]=>>";
	for (int i = 0; i < elements; i++) {
		cout << F[i] << " ";
	}
	cout << endl;*/

	for (int k = 0; k < elements; ++k)
	{
		x[k] = C[k];
		y[k] = D[k];
		_x[k] = E[k];
		_y[k] = F[k];
	}

	// STEP 13: Release OpenCL resources
	clReleaseKernel(kernel);
	clReleaseProgram(program);
	clReleaseCommandQueue(cmdQueue);
	clReleaseMemObject(bufferA);
	clReleaseMemObject(bufferC);
	clReleaseMemObject(bufferD);
	clReleaseMemObject(bufferE);
	clReleaseMemObject(bufferF);
	clReleaseContext(context);
	free(A);
	free(C);
	free(D);
	free(E);
	free(F);
	free(platforms);
	free(devices);
}
// kernel move256() intel UHD GPU platform[1]
void Wrapup4uhdGPU(int ix, int iy, int* x, int* y, int* _x, int* _y) {
	int* A = NULL;
	int* C = NULL;  // Output array
	int* D = NULL;  // Output array
	int* E = NULL;  // Output array
	int* F = NULL;  // Output array

	const int elements = 256;
	size_t datasize = sizeof(int) * elements;
	A = (int*)malloc(datasize);
	C = (int*)malloc(datasize);
	D = (int*)malloc(datasize);
	E = (int*)malloc(datasize);
	F = (int*)malloc(datasize);
	for (int i = 0; i < elements; i++) {
		A[i] = i;
	}
	cl_int status;
	// STEP 1: Discover and initialize the platforms
	cl_uint numPlatforms = 0;
	cl_platform_id* platforms = NULL;
	status = clGetPlatformIDs(0, NULL, &numPlatforms);
	platforms = (cl_platform_id*)malloc(numPlatforms * sizeof(cl_platform_id));
	status = clGetPlatformIDs(numPlatforms, platforms, NULL);
	// STEP 2: Discover and initialize the devices
	cl_uint numDevices = 0;
	cl_device_id* devices = NULL;
	status = clGetDeviceIDs(platforms[1], CL_DEVICE_TYPE_GPU, 0, NULL, &numDevices);
	devices = (cl_device_id*)malloc(numDevices * sizeof(cl_device_id));
	status = clGetDeviceIDs(platforms[1], CL_DEVICE_TYPE_GPU, numDevices, devices, NULL);
	// STEP 3: Create a context
	cl_context context = NULL;
	context = clCreateContext(NULL, numDevices, devices, NULL, NULL, &status);
	// STEP 4: Create a command queue
	cl_command_queue cmdQueue;
	cmdQueue = clCreateCommandQueue(context, devices[0], 0, &status);
	// STEP 5: Create device buffers
	cl_mem bufferA;
	cl_mem bufferC;
	cl_mem bufferD;
	cl_mem bufferE;
	cl_mem bufferF;
	bufferA = clCreateBuffer(context, CL_MEM_READ_ONLY, datasize, NULL, &status);
	bufferC = clCreateBuffer(context, CL_MEM_WRITE_ONLY, datasize, NULL, &status);
	bufferD = clCreateBuffer(context, CL_MEM_WRITE_ONLY, datasize, NULL, &status);
	bufferE = clCreateBuffer(context, CL_MEM_WRITE_ONLY, datasize, NULL, &status);
	bufferF = clCreateBuffer(context, CL_MEM_WRITE_ONLY, datasize, NULL, &status);
	// STEP 6: Write host data to device buffers
	status = clEnqueueWriteBuffer(cmdQueue, bufferA, CL_FALSE, 0, datasize, A, 0, NULL, NULL);
	// STEP 7: Create and compile the program
	std::ifstream srcFile("Template.cl");
	checkErr(srcFile.is_open() ? CL_SUCCESS : -1, "reading Template.cl");
	std::string srcProg(
		std::istreambuf_iterator<char>(srcFile),
		(std::istreambuf_iterator<char>()));

	const char* src = srcProg.c_str();
	size_t length = srcProg.length();
	cl_int errNum;
	cl_program program = clCreateProgramWithSource(context, 1, &src, &length, &errNum);
	checkErr(errNum, "clCreateProgramWithSource");
	status = clBuildProgram(program, numDevices, devices, NULL, NULL, NULL);
	//cl_program program = clCreateProgramWithSource(context,1,(const char**)&programSource,NULL,&status);
	// STEP 8: Create the kernel
	cl_kernel kernel = NULL;
	kernel = clCreateKernel(program, "move256", &status);
	// STEP 9: Set the kernel arguments
	status = clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufferA);
	status |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufferC);
	status |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufferD);
	status |= clSetKernelArg(kernel, 3, sizeof(cl_mem), &bufferE);
	status |= clSetKernelArg(kernel, 4, sizeof(cl_mem), &bufferF);
	status |= clSetKernelArg(kernel, 5, sizeof(cl_mem), &ix);
	status |= clSetKernelArg(kernel, 6, sizeof(cl_mem), &iy);
	// STEP 10: Configure the work-item structure
	size_t globalWorkSize[1];
	globalWorkSize[0] = elements; // size
	// STEP 11: Enqueue the kernel for execution
	status = clEnqueueNDRangeKernel(cmdQueue, kernel, 1, NULL, globalWorkSize, NULL, 0, NULL, NULL);
	// STEP 12: Read the output buffer back to the host
	clEnqueueReadBuffer(cmdQueue, bufferC, CL_TRUE, 0, datasize, C, 0, NULL, NULL);
	clEnqueueReadBuffer(cmdQueue, bufferD, CL_TRUE, 0, datasize, D, 0, NULL, NULL);
	clEnqueueReadBuffer(cmdQueue, bufferE, CL_TRUE, 0, datasize, E, 0, NULL, NULL);
	clEnqueueReadBuffer(cmdQueue, bufferF, CL_TRUE, 0, datasize, F, 0, NULL, NULL);
	//Verify the output
   /*cout << "C[]=>>";
   for (int i = 0; i < elements; i++) {
	   cout << C[i] << " ";
   }
   cout << endl;
   cout << "D[]=>>";
   for (int i = 0; i < elements; i++) {
	   cout << D[i] << " ";
   }
   cout << endl;
   cout << "E[]=>>";
   for (int i = 0; i < elements; i++) {
	   cout << E[i] << " ";
   }
   cout << endl;
   cout << "F[]=>>";
   for (int i = 0; i < elements; i++) {
	   cout << F[i] << " ";
   }
   cout << endl;*/

	for (int k = 0; k < elements; ++k)
	{
		x[k] = C[k];
		y[k] = D[k];
		_x[k] = E[k];
		_y[k] = F[k];
	}

	// STEP 13: Release OpenCL resources
	clReleaseKernel(kernel);
	clReleaseProgram(program);
	clReleaseCommandQueue(cmdQueue);
	clReleaseMemObject(bufferA);
	clReleaseMemObject(bufferC);
	clReleaseMemObject(bufferD);
	clReleaseMemObject(bufferE);
	clReleaseMemObject(bufferF);
	clReleaseContext(context);
	free(A);
	free(C);
	free(D);
	free(E);
	free(F);
	free(platforms);
	free(devices);
}
// kernel move256() intel CPU platform[1]
void Wrapup4CPU(int ix, int iy, int* x, int* y, int* _x, int* _y) {
	int* A = NULL;
	int* C = NULL;  // Output array
	int* D = NULL;  // Output array
	int* E = NULL;  // Output array
	int* F = NULL;  // Output array

	const int elements = 256;
	size_t datasize = sizeof(int) * elements;
	A = (int*)malloc(datasize);
	C = (int*)malloc(datasize);
	D = (int*)malloc(datasize);
	E = (int*)malloc(datasize);
	F = (int*)malloc(datasize);
	for (int i = 0; i < elements; i++) {
		A[i] = i;
	}
	cl_int status;
	// STEP 1: Discover and initialize the platforms
	cl_uint numPlatforms = 0;
	cl_platform_id* platforms = NULL;
	status = clGetPlatformIDs(0, NULL, &numPlatforms);
	platforms = (cl_platform_id*)malloc(numPlatforms * sizeof(cl_platform_id));
	status = clGetPlatformIDs(numPlatforms, platforms, NULL);
	// STEP 2: Discover and initialize the devices
	cl_uint numDevices = 0;
	cl_device_id* devices = NULL;
	status = clGetDeviceIDs(platforms[1], CL_DEVICE_TYPE_CPU, 0, NULL, &numDevices);
	devices = (cl_device_id*)malloc(numDevices * sizeof(cl_device_id));
	status = clGetDeviceIDs(platforms[1], CL_DEVICE_TYPE_CPU, numDevices, devices, NULL);
	// STEP 3: Create a context
	cl_context context = NULL;
	context = clCreateContext(NULL, numDevices, devices, NULL, NULL, &status);
	// STEP 4: Create a command queue
	cl_command_queue cmdQueue;
	cmdQueue = clCreateCommandQueue(context, devices[0], 0, &status);
	// STEP 5: Create device buffers
	cl_mem bufferA;
	cl_mem bufferC;
	cl_mem bufferD;
	cl_mem bufferE;
	cl_mem bufferF;
	bufferA = clCreateBuffer(context, CL_MEM_READ_ONLY, datasize, NULL, &status);
	bufferC = clCreateBuffer(context, CL_MEM_WRITE_ONLY, datasize, NULL, &status);
	bufferD = clCreateBuffer(context, CL_MEM_WRITE_ONLY, datasize, NULL, &status);
	bufferE = clCreateBuffer(context, CL_MEM_WRITE_ONLY, datasize, NULL, &status);
	bufferF = clCreateBuffer(context, CL_MEM_WRITE_ONLY, datasize, NULL, &status);
	// STEP 6: Write host data to device buffers
	status = clEnqueueWriteBuffer(cmdQueue, bufferA, CL_FALSE, 0, datasize, A, 0, NULL, NULL);
	// STEP 7: Create and compile the program
	std::ifstream srcFile("Template.cl");
	checkErr(srcFile.is_open() ? CL_SUCCESS : -1, "reading Template.cl");
	std::string srcProg(
		std::istreambuf_iterator<char>(srcFile),
		(std::istreambuf_iterator<char>()));

	const char* src = srcProg.c_str();
	size_t length = srcProg.length();
	cl_int errNum;
	cl_program program = clCreateProgramWithSource(context, 1, &src, &length, &errNum);
	checkErr(errNum, "clCreateProgramWithSource");
	status = clBuildProgram(program, numDevices, devices, NULL, NULL, NULL);
	//cl_program program = clCreateProgramWithSource(context,1,(const char**)&programSource,NULL,&status);
	// STEP 8: Create the kernel
	cl_kernel kernel = NULL;
	kernel = clCreateKernel(program, "move256", &status);
	// STEP 9: Set the kernel arguments
	status = clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufferA);
	status |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufferC);
	status |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufferD);
	status |= clSetKernelArg(kernel, 3, sizeof(cl_mem), &bufferE);
	status |= clSetKernelArg(kernel, 4, sizeof(cl_mem), &bufferF);
	status |= clSetKernelArg(kernel, 5, sizeof(cl_mem), &ix);
	status |= clSetKernelArg(kernel, 6, sizeof(cl_mem), &iy);
	// STEP 10: Configure the work-item structure
	size_t globalWorkSize[1];
	globalWorkSize[0] = elements; // size
	// STEP 11: Enqueue the kernel for execution
	status = clEnqueueNDRangeKernel(cmdQueue, kernel, 1, NULL, globalWorkSize, NULL, 0, NULL, NULL);
	// STEP 12: Read the output buffer back to the host
	clEnqueueReadBuffer(cmdQueue, bufferC, CL_TRUE, 0, datasize, C, 0, NULL, NULL);
	clEnqueueReadBuffer(cmdQueue, bufferD, CL_TRUE, 0, datasize, D, 0, NULL, NULL);
	clEnqueueReadBuffer(cmdQueue, bufferE, CL_TRUE, 0, datasize, E, 0, NULL, NULL);
	clEnqueueReadBuffer(cmdQueue, bufferF, CL_TRUE, 0, datasize, F, 0, NULL, NULL);
	//Verify the output
   /*cout << "C[]=>>";
   for (int i = 0; i < elements; i++) {
	   cout << C[i] << " ";
   }
   cout << endl;
   cout << "D[]=>>";
   for (int i = 0; i < elements; i++) {
	   cout << D[i] << " ";
   }
   cout << endl;
   cout << "E[]=>>";
   for (int i = 0; i < elements; i++) {
	   cout << E[i] << " ";
   }
   cout << endl;
   cout << "F[]=>>";
   for (int i = 0; i < elements; i++) {
	   cout << F[i] << " ";
   }
   cout << endl;*/

	for (int k = 0; k < elements; ++k)
	{
		x[k] = C[k];
		y[k] = D[k];
		_x[k] = E[k];
		_y[k] = F[k];
	}

	// STEP 13: Release OpenCL resources
	clReleaseKernel(kernel);
	clReleaseProgram(program);
	clReleaseCommandQueue(cmdQueue);
	clReleaseMemObject(bufferA);
	clReleaseMemObject(bufferC);
	clReleaseMemObject(bufferD);
	clReleaseMemObject(bufferE);
	clReleaseMemObject(bufferF);
	clReleaseContext(context);
	free(A);
	free(C);
	free(D);
	free(E);
	free(F);
	free(platforms);
	free(devices);
}

//=======================================
//    point positions
//    8 1 2
//    7 0 3
//    6 5 4
//========= openGL ======================

void init() {
	glClearColor(0.0, 0.0, 0.0, 0.0);
	glMatrixMode(GL_PROJECTION);
	gluOrtho2D(-0.5f, WIDTH - 0.5f, -0.5f, HEIGHT - 0.5f);
}

//inject() Wrapup() moveInGPU() cx6 x++ cy6 y++ ccx6 x-- ccy6 y-- m6[]
void inject(int x, int y, int num) {
	cx6[0] = x;
	cy6[0] = y;
	m6[0] = cell[x][y];
	Wrapup(x, y);
}

//inject2() Wrapup2() moveRight() cx5 x++ cy5 y++ ccx5 x-- ccy5 y-- m5[]
void inject2(int x, int y, int num) {
	cx5[0] = x;
	cy5[0] = y;
	m5[0] = cell[x][y];
	Wrapup2(x, y);
}

//inject3() Wrapup3() moveLeft() cx4 x++ cy4 y++ ccx4 x-- ccy4 y-- m4[]
void inject3(int x, int y, int num) {
	cx4[0] = x;
	cy4[0] = y;
	m4[0] = cell[x][y];
	Wrapup3(x, y);
}

//inject() Wrapup() moveInGPU() cx6 x++ cy6 y++ ccx6 x-- ccy6 y-- m6[]
void moveInGPU() {
	if (cell[cx6[0]][cy6[q]] == 4)
	cell[cx6[0]][cy6[q]] = m6[1];
	if(cell[cx6[q]][cy6[q]] == 4)
	cell[cx6[q]][cy6[q]] = m6[2];
	if(cell[cx6[q]][cy6[0]] == 4)
	cell[cx6[q]][cy6[0]] = m6[3];
	if(cell[cx6[q]][ccy6[q]] == 4)
	cell[cx6[q]][ccy6[q]] = m6[4];
	if(cell[cx6[0]][ccy6[q]] ==4)
	cell[cx6[0]][ccy6[q]] = m6[5];

	q++;
	m6[1] = cell[cx6[0]][cy6[q]];
	cell[cx6[0]][cy6[q]] = 4;
	m6[2] = cell[cx6[q]][cy6[q]];
	cell[cx6[q]][cy6[q]] = 4;
	m6[3] = cell[cx6[q]][cy6[0]];
	cell[cx6[q]][cy6[0]] = 4;
	m6[4] = cell[cx6[q]][ccy6[q]];
	cell[cx6[q]][ccy6[q]] = 4;
	m6[5] = cell[cx6[0]][ccy6[q]];
	cell[cx6[0]][ccy6[q]] = 4;
}

//inject2() Wrapup2() moveRight() cx5 x++ cy5 y++ ccx5 x-- ccy5 y-- m5[]
void moveInGPU2() {
	if (cell[cx5[0]][cy5[q1]] == 4)
		cell[cx5[0]][cy5[q1]] = m5[1];
	if (cell[cx5[q1]][cy5[q1]] == 4)
		cell[cx5[q1]][cy5[q1]] = m5[2];
	if (cell[cx5[q1]][cy5[0]] == 4)
		cell[cx5[q1]][cy5[0]] = m5[3];
	if (cell[cx5[q1]][ccy5[q1]] == 4)
		cell[cx5[q1]][ccy5[q1]] = m5[4];
	if (cell[cx5[0]][ccy5[q1]] == 4)
		cell[cx5[0]][ccy5[q1]] = m5[5];

	q1++;
	m5[1] = cell[cx5[0]][cy5[q1]];
	cell[cx5[0]][cy5[q1]] = 4;
	m5[2] = cell[cx5[q1]][cy5[q1]];
	cell[cx5[q1]][cy5[q1]] = 4;
	m5[3] = cell[cx5[q1]][cy5[0]];
	cell[cx5[q1]][cy5[0]] = 4;
	m5[4] = cell[cx5[q1]][ccy5[q1]];
	cell[cx5[q1]][ccy5[q1]] = 4;
	m5[5] = cell[cx5[0]][ccy5[q1]];
	cell[cx5[0]][ccy5[q1]] = 4;
}

//inject3() Wrapup3() moveLeft() cx4 x++ cy4 y++ ccx4 x-- ccy4 y-- m4[]
void moveLeft() {
	if (cell[ccx4[q2]][ccy4[q2]] == 4)
		cell[ccx4[q2]][ccy4[q2]] = m4[6];
	if (cell[ccx4[q2]][cy4[0]] == 4)
		cell[ccx4[q2]][cy4[0]] = m4[7];
	if (cell[ccx4[q2]][cy4[q2]] == 4)
		cell[ccx4[q2]][cy4[q2]] = m4[8];

	q2++;
	m4[6] = cell[ccx4[q2]][ccy4[q2]];
	cell[ccx4[q2]][ccy4[q2]] = 4;
	m4[7] = cell[ccx4[q2]][cy4[0]];
	cell[ccx4[q2]][cy4[0]] = 4;
	m4[8] = cell[ccx4[q2]][cy4[q2]];
	cell[ccx4[q2]][cy4[q2]] = 4;
}

//============== helper methods ==================
//x[] x++ y[] y++ _x[] x-- _y[] y-- m?[] Wrapup4(px, py, x, y, _x, _y) default for all device type 1=>nvGPU 2=>CPU 3=>uhdGPU
void injectHelper(int* x, int* y, int* m, int px, int py, int* _x, int* _y, int deviceType) {
	x[0] = px;
	y[0] = py;
	m[0] = cell[px][py];
	switch (deviceType) {
	case 1:
		Wrapup4nvGPU(px, py, x, y, _x, _y);
		break;
	case 2:
		Wrapup4CPU(px, py, x, y, _x, _y);
		break;
	case 3:
		Wrapup4uhdGPU(px, py, x, y, _x, _y);
		break;
	default:
		Wrapup4All(px, py, x, y, _x, _y);
	}
}
//inject move up 3 or 5 points moveUp()
void inject_u(int px, int py, int num, int deviceType) {
	if (num == 3) {
		switch (deviceType) {
		case 1:
			injectHelper(cx1, cy1, m1, px, py, ccx1, ccy1, 1);
			break;
		case 2:
			injectHelper(cx10, cy10, m10, px, py, ccx10, ccy10, 2);
			break;
		default:
			injectHelper(cx1, cy1, m1, px, py, ccx1, ccy1, 1);
		}
	}
	else { // num == 5
		switch (deviceType) {
		case 1:
			injectHelper(cx11, cy11, m11, px, py, ccx11, ccy11, 1);
			break;
		case 2:
			injectHelper(cx110, cy110, m110, px, py, ccx110, ccy110, 2);
			break;
		default:
			injectHelper(cx11, cy11, m11, px, py, ccx11, ccy11, 1);
		}
	}
}
//inject move down 3 points moveDown()
void inject_d(int px, int py, int num, int deviceType) {
	switch (deviceType) {
	case 1:
		injectHelper(cx2, cy2, m2, px, py, ccx2, ccy2, 1);
		break;
	case 2:
		injectHelper(cx20, cy20, m20, px, py, ccx20, ccy20, 2);
		break;
	}
}
//inject move left 3 points moveLf()
void inject_l(int px, int py, int num, int deviceType) {
	switch (deviceType) {
	case 1:
		injectHelper(cx3, cy3, m3, px, py, ccx3, ccy3, 1);
		break;
	case 2:
		injectHelper(cx30, cy30, m30, px, py, ccx30, ccy30, 2);
		break;
	case 3:
		injectHelper(cx301, cy301, m301, px, py, ccx301, ccy301, 3);
		break;
	default:
		injectHelper(cx3, cy3, m3, px, py, ccx3, ccy3, 1);
	}
}
//inject move right 3 points moveRt()
void inject_r(int px, int py, int num, int deviceType) {
	switch (deviceType) {
	case 1:
		injectHelper(cx4, cy4, m4, px, py, ccx4, ccy4, 1);
		break;
	case 2:
		injectHelper(cx40, cy40, m40, px, py, ccx40, ccy40, 2);
		break;
	case 3:
		injectHelper(cx401, cy401, m401, px, py, ccx401, ccy401, 3);
		break;
	default:
		injectHelper(cx4, cy4, m4, px, py, ccx4, ccy4, 1);
	}
}
// inject 4 points with different move directions, default for all device type
void multipleInject(int px1, int py1, int px2, int py2, int px3, int py3, int px4, int py4, int num) {
	inject_l(px1, py1, num, 4); // default for all device type
	inject_u(px2, py2, num, 4);
	inject_d(px3, py3, num, 4);
	inject_r(px4, py4, num, 4);
}

// test GPU and CPU
void multipleInject_GPUCPU(int px1, int py1, int px2, int py2, int px3, int py3, int px4, int py4, int px5, int py5, int px10, int py10, int px20, int py20, int px30, int py30, int px40, int py40, int px50, int py50, int num1, int num2) {
	inject_l(px1, py1, num1, 1); // nvGPU
	inject_u(px2, py2, num1, 1);
	inject_d(px3, py3, num1, 1);
	inject_r(px4, py4, num1, 1);
	inject_u(px5, py5, num2, 1);

	inject_l(px10, py10, num1, 2); // CPU
	inject_u(px20, py20, num1, 2);
	inject_d(px30, py30, num1, 2);
	inject_r(px40, py40, num1, 2);
	inject_u(px50, py50, num2, 2);
}

//x[] x++ y[] y++ _x[] x-- _y[] y-- m?[] q?
void moveUpHelper(int* x, int* y, int* _x, int* _y, int* m, int q) {
	if (cell[x[0]][y[q]] == 4)
		cell[x[0]][y[q]] = m[1];
	if (cell[x[q]][y[q]] == 4)
		cell[x[q]][y[q]] = m[2];
	if (cell[_x[q]][y[q]] == 4)
		cell[_x[q]][y[q]] = m[8];

	q++;
	m[1] = cell[x[0]][y[q]];
	cell[x[0]][y[q]] = 4;
	m[2] = cell[x[q]][y[q]];
	cell[x[q]][y[q]] = 4;
	m[8] = cell[_x[q]][y[q]];
	cell[_x[q]][y[q]] = 4;
}
void moveUpHelper1(int* x, int* y, int* _x, int* _y, int* m, int q) {
	if (cell[x[0]][y[q]] == 4)
		cell[x[0]][y[q]] = m[1];
	if (cell[x[q]][y[q]] == 4)
		cell[x[q]][y[q]] = m[2];
	if (cell[_x[q]][y[q]] == 4)
		cell[_x[q]][y[q]] = m[8];
}
void moveUpHelper2(int* x, int* y, int* _x, int* _y, int* m, int q) {
	m[1] = cell[x[0]][y[q]];
	cell[x[0]][y[q]] = 4;
	m[2] = cell[x[q]][y[q]];
	cell[x[q]][y[q]] = 4;
	m[8] = cell[_x[q]][y[q]];
	cell[_x[q]][y[q]] = 4;
}
void moveUp5Helper1(int* x, int* y, int* _x, int* _y, int* m, int q) {
	if (cell[x[0]][y[q]] == 4)
		cell[x[0]][y[q]] = m[1];
	if (cell[x[q]][y[q]] == 4)
		cell[x[q]][y[q]] = m[2];
	if (cell[x[q]][y[0]] == 4)
		cell[x[q]][y[0]] = m[3];
	if (cell[_x[q]][y[0]] == 4)
		cell[_x[q]][y[0]] = m[7];
	if (cell[_x[q]][y[q]] == 4)
		cell[_x[q]][y[q]] = m[8];
}
void moveUp5Helper2(int* x, int* y, int* _x, int* _y, int* m, int q) {
	m[1] = cell[x[0]][y[q]];
	cell[x[0]][y[q]] = 4;
	m[2] = cell[x[q]][y[q]];
	cell[x[q]][y[q]] = 4;
	m[3] = cell[x[q]][y[0]];
	cell[x[q]][y[0]] = 4;
	m[7] = cell[_x[q]][y[0]];
	cell[_x[q]][y[0]] = 4;
	m[8] = cell[_x[q]][y[q]];
	cell[_x[q]][y[q]] = 4;
}
// cx11[] m11[] q11[] num=5 GPU
void moveUp5() {
	moveUp5Helper1(cx11, cy11, ccx11, ccy11, m11, q11);
	q11++;
	moveUp5Helper2(cx11, cy11, ccx11, ccy11, m11, q11);
}
// cx11[] m11[] q11[] num=5 CPU
void moveUp50() {
	moveUp5Helper1(cx110, cy110, ccx110, ccy110, m110, q110);
	q110++;
	moveUp5Helper2(cx110, cy110, ccx110, ccy110, m110, q110);
}
// cx1[] m1[], q1 num=3 GPU
void moveUp() {
	moveUpHelper1(cx1, cy1, ccx1, ccy1, m1, q1);
	q1++;
	moveUpHelper2(cx1, cy1, ccx1, ccy1, m1, q1);
}
// cx10[] m10[] q10 num=3 CPU
void moveUp0() {
	moveUpHelper1(cx10, cy10, ccx10, ccy10, m10, q10);
	q10++;
	moveUpHelper2(cx10, cy10, ccx10, ccy10, m10, q10);
}

//x[] x++ y[] y++ _x[] x-- _y[] y-- m?[] q?
void moveDownHelper(int* x, int* y, int* _x, int* _y, int* m, int q) {
	if (cell[x[q]][_y[q]] == 4)
		cell[x[q]][_y[q]] = m[4];
	if (cell[x[0]][_y[q]] == 4)
		cell[x[0]][_y[q]] = m[5];
	if (cell[_x[q]][_y[q]] == 4)
		cell[_x[q]][_y[q]] = m[6];


	q++;
	m[4] = cell[x[q]][_y[q]];
	cell[x[q]][_y[q]] = 4;
	m[5] = cell[x[0]][_y[q]];
	cell[x[0]][_y[q]] = 4;
	m[6] = cell[_x[q]][_y[q]];
	cell[_x[q]][_y[q]] = 4;
}
void moveDownHelper1(int* x, int* y, int* _x, int* _y, int* m, int q) {
	if (cell[x[q]][_y[q]] == 4)
		cell[x[q]][_y[q]] = m[4];
	if (cell[x[0]][_y[q]] == 4)
		cell[x[0]][_y[q]] = m[5];
	if (cell[_x[q]][_y[q]] == 4)
		cell[_x[q]][_y[q]] = m[6];
}
void moveDownHelper2(int* x, int* y, int* _x, int* _y, int* m, int q) {
	m[4] = cell[x[q]][_y[q]];
	cell[x[q]][_y[q]] = 4;
	m[5] = cell[x[0]][_y[q]];
	cell[x[0]][_y[q]] = 4;
	m[6] = cell[_x[q]][_y[q]];
	cell[_x[q]][_y[q]] = 4;
}
// cx2[] m2[], q2
void moveDown() {
	moveDownHelper1(cx2, cy2, ccx2, ccy2, m2, q2);
	q2++;
	moveDownHelper2(cx2, cy2, ccx2, ccy2, m2, q2);
}

void moveDown0() {
	moveDownHelper1(cx20, cy20, ccx20, ccy20, m20, q20);
	q20++;
	moveDownHelper2(cx20, cy20, ccx20, ccy20, m20, q20);
}

//x[] x++ y[] y++ _x[] x-- _y[] y-- m?[] q?
void moveRtHelper(int* x, int* y, int* _x, int* _y, int* m, int q) {
	if (cell[x[q]][y[q]] == 4)
		cell[x[q]][y[q]] = m[2];
	if (cell[x[q]][y[0]] == 4)
		cell[x[q]][y[0]] = m[3];
	if (cell[x[q]][_y[q]] == 4)
		cell[x[q]][_y[q]] = m[4];

	q++;
	m[2] = cell[x[q]][y[q]];
	cell[x[q]][y[q]] = 4;
	m[3] = cell[x[q]][y[0]];
	cell[x[q]][y[0]] = 4;
	m[4] = cell[x[q]][_y[q]];
	cell[x[q]][_y[q]] = 4;
}
void moveRtHelper1(int* x, int* y, int* _x, int* _y, int* m, int q) {
	if (cell[x[q]][y[q]] == 4)
		cell[x[q]][y[q]] = m[2];
	if (cell[x[q]][y[0]] == 4)
		cell[x[q]][y[0]] = m[3];
	if (cell[x[q]][_y[q]] == 4)
		cell[x[q]][_y[q]] = m[4];
}
void moveRtHelper2(int* x, int* y, int* _x, int* _y, int* m, int q) {
	m[2] = cell[x[q]][y[q]];
	cell[x[q]][y[q]] = 4;
	m[3] = cell[x[q]][y[0]];
	cell[x[q]][y[0]] = 4;
	m[4] = cell[x[q]][_y[q]];
	cell[x[q]][_y[q]] = 4;
}
// cx4[] m4[], q4 GPU num=3
void moveRt() {
	moveRtHelper1(cx4, cy4, ccx4, ccy4, m4, q4);
	q4++;
	moveRtHelper2(cx4, cy4, ccx4, ccy4, m4, q4);
}
// cx40[] m40[], q40 CPU num=3
void moveRt0() {
	moveRtHelper1(cx40, cy40, ccx40, ccy40, m40, q40);
	q40++;
	moveRtHelper2(cx40, cy40, ccx40, ccy40, m40, q40);
}
// cx401[] m401[], q401 uhdGPU num=3
void moveRt01() {
	moveRtHelper1(cx401, cy401, ccx401, ccy401, m401, q401);
	q401++;
	moveRtHelper2(cx401, cy401, ccx401, ccy401, m401, q401);
}

//x[] x++ y[] y++ _x[] x-- _y[] y-- m?[] q?
void moveLfHelper(int* x, int* y, int* _x, int* _y, int* m, int q) {
	if (cell[_x[q]][_y[q]] == 4)
		cell[_x[q]][_y[q]] = m[6];
	if (cell[_x[q]][y[0]] == 4)
		cell[_x[q]][y[0]] = m[7];
	if (cell[_x[q]][y[q]] == 4)
		cell[_x[q]][y[q]] = m[8];

	q++;
	m[6] = cell[_x[q]][_y[q]];
	cell[_x[q]][_y[q]] = 4;
	m[7] = cell[_x[q]][y[0]];
	cell[_x[q]][y[0]] = 4;
	m[8] = cell[_x[q]][y[q]];
	cell[_x[q]][y[q]] = 4;
}
void moveLfHelper1(int* x, int* y, int* _x, int* _y, int* m, int q) {
	if (cell[_x[q]][_y[q]] == 4)
		cell[_x[q]][_y[q]] = m[6];
	if (cell[_x[q]][y[0]] == 4)
		cell[_x[q]][y[0]] = m[7];
	if (cell[_x[q]][y[q]] == 4)
		cell[_x[q]][y[q]] = m[8];
}
void moveLfHelper2(int* x, int* y, int* _x, int* _y, int* m, int q) {
	m[6] = cell[_x[q]][_y[q]];
	cell[_x[q]][_y[q]] = 4;
	m[7] = cell[_x[q]][y[0]];
	cell[_x[q]][y[0]] = 4;
	m[8] = cell[_x[q]][y[q]];
	cell[_x[q]][y[q]] = 4;
}
// cx3[] m3[], q3 GPU num=3
void moveLf() {
	moveLfHelper1(cx3, cy3, ccx3, ccy3, m3, q3);
	q3++;
	moveLfHelper2(cx3, cy3, ccx3, ccy3, m3, q3);
}
// cx30[] m30[] q30 CPU num=3
void moveLf0() {
	moveLfHelper1(cx30, cy30, ccx30, ccy30, m30, q30);
	q30++;
	moveLfHelper2(cx30, cy30, ccx30, ccy30, m30, q30);
}
// cx301[] m301[] q301 num=3 uhdGPU
void moveLf01() {
	moveLfHelper1(cx301, cy301, ccx301, ccy301, m301, q301);
	q301++;
	moveLfHelper2(cx301, cy301, ccx301, ccy301, m301, q301);
}

void printCx() {
	for (int i = 0; i < 256; i++) {
		cout << "cx4: " << cx3[i] << endl;
		cout << "cy4: " << cy3[i] << endl;
		cout << "ccy4: " << ccy3[i] << endl;
		cout << "ccx4: " << ccx3[i] << endl;
	}
}

void setup(int x, int y, int m) {
	int w = (m * x) + 2;

	for (int i = (w - x); i < w; i++) {
		for (int j = 2; j < y + 2; j++) {

			cell[i][j] = (rand() % 2 + 2); // 2,3
		}
	}
}

void changeColor(GLfloat red, GLfloat green, GLfloat blue) {
	r = red;
	g = green;
	b = blue;
}

//Check status of individual cell and apply the rules: 3 is cancer, 2 is health cell, 4 is medicine
static int checkStatus(int status, int x, int y) {
	int cancerNeighbours = 0;
	int medicineNeighbours = 0;

	for (int i = (x - 1); i < (x + 2); i++) {
		if (cell[i][y - 1] == 3) {
			cancerNeighbours++;
		}
		if (cell[i][y + 1] == 3) {
			cancerNeighbours++;
		}
	}
	if (cell[x - 1][y] == 3) {
		cancerNeighbours++;
	}
	if (cell[x + 1][y] == 3) {
		cancerNeighbours++;
	}

	for (int i = (x - 1); i < (x + 2); i++) {
		if (cell[i][y - 1] == 4) {
			medicineNeighbours++;
		}
		if (cell[i][y + 1] == 4) {
			medicineNeighbours++;
		}
	}
	if (cell[x - 1][y] == 4) {
		medicineNeighbours++;
	}
	if (cell[x + 1][y] == 4) {
		medicineNeighbours++;
	}

	if (status == 3 && medicineNeighbours >= 6) {
		status = 2;
	}
	else if (status == 2 && cancerNeighbours >= 6) {
		status = 3;
	}
	return status;
}

//Display individual pixels.
static void display()
{
	glClear(GL_COLOR_BUFFER_BIT);

	GLfloat red, green, blue;

	for (int i = 5; i < (WIDTH - 5); i++) {
		for (int j = 5; j < (HEIGHT - 5); j++) {
			//Check the updated status of the current cell.
			int cellV = checkStatus(cell[i][j], i, j);
			if (cellV == 0) {
				red = r;
				green = 0.0f;
				blue = 1.0;
				cell[i][j] = 0;
			}
			else if (cellV == 2) {
				red = r;
				green = 0.4f;
				blue = b;
				cell[i][j] = 2;
			}
			else if (cellV == 3) {
				red = 0.4f;
				green = g;
				blue = b;
				cell[i][j] = 3;
			}
			else if (cellV == 4) {
				red = 1.0f;
				green = 1.0f;
				blue = 0.0f;
				cell[i][j] = 4;
			}

			glPointSize(1.0f);
			glColor3f(red, green, blue);
			glBegin(GL_POINTS);
			glVertex2i(i, j);
			glEnd();
		}
	}
	glutSwapBuffers();
}

void update(int value) {
	try {
		//=== test 2 ====
		/*moveInGPU();
		moveInGPU2();
		moveLeft();*/

		//== test 1 ====
		moveLf(); // GPU
		moveUp();
		moveDown();
		moveRt();
		moveUp5();

		moveLf0(); // CPU
		moveUp0();
		moveDown0();
		moveRt0();
		moveUp50();

		//== test 3 ===
		//moveInGPU(); // nvGPU
		//moveInGPU2();
		//moveLf01();   // uhdGPU
		//moveRt01();
		//moveDown0(); // CPU
		//moveUp50();
	}
	catch (...) {}
	glutPostRedisplay();
	glutTimerFunc(1000 / 30, update, 0);
}

int main(int argc, char** argv)
{
	int x = 1020;
	int y = 766;
	int m = 1;

	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH);
	glutInitWindowSize(WIDTH, HEIGHT);
	glutCreateWindow("Cell Growth Simulator");
	init();

	setup(x, y, m);

	//=== test 1 ========
	multipleInject_GPUCPU(800,500, 300,500, 500,500, 100,500, 400,500, 800, 300, 300, 300, 500, 300, 100, 300, 400, 300, 3, 5);

	//==== test 2 =======
	//inject(300, 400, 5); // GPU
	//inject2(500, 400, 5);
	//inject3(800, 400, 3);

	//=== test 3 ========
	//inject(300, 500, 5); // nvGPU
	//inject2(400, 500, 5);
	//inject_l(500, 400, 3, 3); // uhdGPU
	//inject_r(600, 400, 3, 3);
	//inject_d(700, 500, 3, 2); // CPU
	//inject_u(800, 500, 5, 2);

	glutDisplayFunc(display);
	glutTimerFunc(1000 / 30, update, 0);
	changeColor(r, g, b);
	glutMainLoop();
	
    return 0;
}