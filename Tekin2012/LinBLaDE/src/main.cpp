#include "iohandler.h"
#include "opts.h"
#include "ski/log.h"
Opts gOpts;

/**
 * Main function
 */
int main(int argc, char* argv[])
{
	/** Set Options and process command line arguments */
	try
	{
		if (!parse(argc, argv, gOpts))
			return EXIT_SUCCESS; //command requires exit without processing.
		/** Initialization */
		IOHandler Project;
		/** Processing */
		return Project.start();
	}
	catch (exception &aErr)
	{
		LOGD("Error: %s\n", aErr.what());
		return EXIT_FAILURE;
	}
	return EXIT_SUCCESS; //Should not come here
}
