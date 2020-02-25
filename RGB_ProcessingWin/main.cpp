#include <iostream>
#include <vector>
//Thread building blocks library
#include <tbb/task_scheduler_init.h>
//Free Image library
#include <FreeImagePlus.h>

//My includes
#include <functional>
#include <thread>
#include <chrono>
#include <cmath>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#include <tbb/blocked_range2d.h>
#include <tbb/parallel_reduce.h>

//Defines
#define M_PIf 3.14159265358979f  // reduces computation overhead when using pi.

using namespace std;
using namespace tbb;


struct pixel_rgb
{
	alignas(1) unsigned char r = 0;
	alignas(1) unsigned char g = 0;
	alignas(1) unsigned char b = 0;
};


// PART 1

// serial solution (using pixel_rgb type)
bool CombineImagesSerial(char* inImagePathA, char* inImagePathB, char* outImagePath, function<pixel_rgb(pixel_rgb, pixel_rgb)> fn)
{
	// Load input images from disk as FreeImagePlus images
	fipImage imgInA;
	fipImage imgInB;
	imgInA.load(inImagePathA);
	imgInB.load(inImagePathB);
	imgInA.convertTo24Bits();
	imgInB.convertTo24Bits();

	// Both input images must have the same dimensions
	if (imgInA.getWidth() != imgInB.getWidth() || imgInA.getHeight() != imgInB.getHeight()) return false;

	// Create an empty output image with the same dimensions as the inputs
	unsigned int width = imgInA.getWidth();
	unsigned int height = imgInA.getHeight();
	unsigned int numPixels = width * height;

	// Iterate over each pixel and apply the given function
	pixel_rgb* aPointer = (pixel_rgb*)imgInA.accessPixels(); // also acts as the output image
	pixel_rgb* bPointer = (pixel_rgb*)imgInB.accessPixels();
	for (uint64_t pixel = 0; pixel < numPixels; pixel++, aPointer++, bPointer++)
	{
		*aPointer = fn(*aPointer, *bPointer);
	}

	// Save output image to disk
	return imgInA.save(outImagePath);
}

// parallel solution (using float)
void CombineSubImage(float* aPointer, float* bPointer, float* outPointer, uint64_t numIterations, function<float(float, float)> fn)
{
	for (uint64_t pixel = 0; pixel < numIterations; pixel++, aPointer++, bPointer++, outPointer++)
	{
		*outPointer = fn(*aPointer, *bPointer);
	}
}

// parallel solution (using float)
bool CombineImagesParallel(char* inImagePathA, char* inImagePathB, char* outImagePath, uint64_t numThreads, function<float(float, float)> fn)
{
	// Load input images from disk as FreeImagePlus images
	fipImage imgInA;
	fipImage imgInB;
	imgInA.load(inImagePathA);
	imgInB.load(inImagePathB);
	imgInA.convertToFloat();
	imgInB.convertToFloat();

	// Both input images must have the same dimensions
	if (imgInA.getWidth() != imgInB.getWidth() || imgInA.getHeight() != imgInB.getHeight()) return false;

	// Create an empty output image with the same dimensions as the inputs
	uint64_t width = imgInA.getWidth();
	uint64_t height = imgInA.getHeight();
	uint64_t numPixels = width * height; // TODO: possibility of overflow
	fipImage imgOut(FIT_FLOAT, width, height, 32); // TODO: hardcoding bits per pixel, should consider dynamically checking size of floats

	// Create threads to process smaller sub-images
	vector<thread> threads;
	uint64_t stepsize = numPixels / numThreads;
	uint64_t remainder = numPixels - (stepsize * numThreads);
	float* aPointer = (float*)imgInA.accessPixels();
	float* bPointer = (float*)imgInB.accessPixels();
	float* outPointer = (float*)imgOut.accessPixels();
	for (int i = 0; i < numThreads; i++)
	{
		if (i == 0) 
		{
			threads.push_back(thread(CombineSubImage, aPointer, bPointer, outPointer, stepsize + remainder, fn));
			aPointer += stepsize + remainder;
			bPointer += stepsize + remainder;
			outPointer += stepsize + remainder;
		}
		else
		{
			threads.push_back(thread(CombineSubImage, aPointer, bPointer, outPointer, stepsize, fn));
			aPointer += stepsize;
			bPointer += stepsize;
			outPointer += stepsize;
		}
	}

	// Wait for the threads to finish executing
	for (auto& thread : threads)
	{
		thread.join();
	}

	// Save output image to disk
	imgOut.convertToType(FREE_IMAGE_TYPE::FIT_BITMAP);
	imgOut.convertTo24Bits();
	return imgOut.save(outImagePath);
}



// PART 2 (I have based some of the interface around the OpenCV module, but not the implementation)
// Link: https://docs.opencv.org/2.4/modules/imgproc/doc/filtering.html

// generic functor parent
class StencilTask
{
public:
	virtual float operator()(float x, float y) = 0;
};

// Gaussian kernal
//  use constructor to set Gaussian parameters 'sigma' and 'kernal pre-load radius' e.g. GaussianBlur MyGaussian = GaussianBlur(0.8f, 2)
//  once instantiated, object can be called as if it was a function e.g. MyGaussian(x_coord, y_coord)
class GaussianBlur : public StencilTask
{
private:
	float sigma;
	int kradius;
	vector<vector<float>> kernel;

	// Pre-load mask matrix to avoid computing the same value multiple times.
	// With kradius=0 no values will be pre-loaded and everything is computed dynamically.
	void loadKernel(int kradius)
	{
		// x and y start from -kradius and finish at kradius inclusively.
		for (int x = -(kradius); x <= kradius; x++)
		{
			vector<float> curRow;
			for (int y = -(kradius); y <= kradius; y++)
			{
				curRow.push_back(1.0f / (2.0f * M_PIf * pow(sigma, 2)) * exp(-((pow(x, 2) + pow(y, 2)) / (2.0f * pow(sigma, 2)))));
			}
			kernel.push_back(curRow);
		}
	}

public:
	// 'kradius' is the number of values to the left and right of the origin in the pre-loaded mask matrix.
	// It is not a hard limit on the number of computable values.
	GaussianBlur(float sigma, int kradius=0)
	{
		this->sigma = sigma;
		this->kradius = kradius;
		this->loadKernel(kradius);
	}

	float operator()(float x, float y)
	{
		if (abs(x) <= kradius && abs(y) <= kradius) return kernel[x + kradius][y + kradius]; //value HAS been pre-loaded
		else return 1.0f / (2.0f * M_PIf * pow(sigma, 2)) * exp(-((pow(x, 2) + pow(y, 2)) / (2.0f * pow(sigma, 2)))); //value has NOT been pre-loaded
	}
};

// NOTE: 'kradius' is the number of pixels away from the origin the mask is applied to, e.g. kradius=2 means a 5x5 matrix
// Both solutions use 'extend' edge handling

// serial stencil solution
bool ProcessImageStencilSerial(char* inImagePath, char* outImagePath, int kradius, StencilTask& STask)
{
	// Load input image from disk into memory
	fipImage imgIn;
	imgIn.load(inImagePath);
	imgIn.convertToFloat();

	// Load empty output image into memory
	unsigned int width = imgIn.getWidth();
	unsigned int height = imgIn.getHeight();
	unsigned int numPixels = width * height;
	fipImage imgOut(FIT_FLOAT, width, height, 32);

	// Setup variables for accessing memory
	float* inPointer = (float*)imgIn.accessPixels();
	float* outPointer = (float*)imgOut.accessPixels();
	float sum;
	unsigned int stencilIndex = 0;

	// Iterate over all pixels linearly
	for (unsigned int xOrigin = 0; xOrigin < width; xOrigin++)
	{
		for (unsigned int yOrigin = 0; yOrigin < height; yOrigin++, outPointer++)
		{
			sum = 0.0f;

			for (int yStencil = -(kradius); yStencil <= kradius; yStencil++)
			{
				// Make sure that the stencil's y-coord is within range. If not, snap to the nearest border pixel.
				int absoluteYStencil = (int)yOrigin + yStencil;
				if (absoluteYStencil < 0) absoluteYStencil = 0;
				else if (absoluteYStencil >= height) absoluteYStencil = height - 1;

				for (int xStencil = -(kradius); xStencil <= kradius; xStencil++)
				{
					// Make sure that the stencil's x-coord is within range. If not, snap to the nearest border pixel.
					int absoluteXStencil = xOrigin + xStencil;
					if (absoluteXStencil < 0) absoluteXStencil = 0;
					else if (absoluteXStencil >= width) absoluteXStencil = width - 1;

					stencilIndex = (absoluteYStencil * width) + absoluteXStencil;
					sum += inPointer[stencilIndex] * STask(xStencil, yStencil);
				}
			}
			*outPointer = sum;
		}
	}

	// Save output image to disk
	return imgOut.save(outImagePath);
}

// parallel stencil solution
bool ProcessImageStencilParallel(char* inImagePath, char* outImagePath, int kradius, StencilTask& STask)
{
	fipImage imgIn;
	imgIn.load(inImagePath);
	imgIn.convertToFloat();

	// Load empty output image into memory
	unsigned int width = imgIn.getWidth();
	unsigned int height = imgIn.getHeight();
	unsigned int numPixels = width * height;
	fipImage imgOut(FIT_FLOAT, width, height, 32);

	// Iterate over all pixels using TBB parallel_for
	float* inPointer = (float*)imgIn.accessPixels();
	float* outPointer = (float*)imgOut.accessPixels();

	// For each pixel in the input...
	parallel_for(blocked_range2d<int>(0, (int)height, 128, 0, (int)width, 128), [&](const blocked_range2d<int>& dim)
	{
			int xstart = dim.cols().begin();
			int ystart = dim.rows().begin();
			int xend = dim.cols().end();
			int yend = dim.rows().end();
			int xsum = 0;
			int ysum = 0;

			for (int x = xstart; x != xend; x++) 
			{
				for (int y = ystart; y != yend; y++)
				{
					for (int kx = -kradius; kx <= kradius; kx++)
					{
						xsum = x + kx;
						if (xsum < 0) xsum = 0;
						if (xsum >= width) xsum = width - 1;
						for (int ky = -kradius; ky <= kradius; ky++)
						{
							ysum = y + ky;
							if (ysum < 0) ysum = 0;
							if (ysum >= height) ysum = height - 1;

							outPointer[(y * width) + x] += inPointer[(ysum * width) + xsum] * STask(kx, ky);
						}
					}
				}
			}
	});

	// Save output image to disk
	imgOut.convertToType(FREE_IMAGE_TYPE::FIT_BITMAP);
	imgOut.convertTo24Bits();
	return imgOut.save(outImagePath);
}

// serial linear operation solution
bool ApplyToImageSerial(char* inImagePath, char* outImagePath, function<pixel_rgb(pixel_rgb)> fn)
{
	// Load input images from disk as FreeImagePlus images
	fipImage imgIn;
	imgIn.load(inImagePath);
	imgIn.convertTo24Bits();

	// Create an empty output image with the same dimensions as the inputs
	unsigned int width = imgIn.getWidth();
	unsigned int height = imgIn.getHeight();
	unsigned int numPixels = width * height;

	// Iterate over each pixel and apply the given function
	pixel_rgb* inPointer = (pixel_rgb*)imgIn.accessPixels();
	for (unsigned int pixel = 0; pixel < numPixels; pixel++, inPointer++)
	{
		*inPointer = fn(*inPointer);
	}

	// Save output image to disk
	return imgIn.save(outImagePath);
}

// parallel linear operation solution
bool ApplyToImageParallel(char* inImagePath, char* outImagePath, function<pixel_rgb(pixel_rgb)> fn)
{
	// Load input images from disk as FreeImagePlus images
	fipImage imgIn;
	imgIn.load(inImagePath);
	imgIn.convertToFloat();

	// Create an empty output image with the same dimensions as the inputs
	unsigned int width = imgIn.getWidth();
	unsigned int height = imgIn.getHeight();
	unsigned int numPixels = width * height; 
	fipImage imgOut(FIT_BITMAP, width, height, 24); 

	// Iterate over each pixel and apply the given function
	pixel_rgb* inPointer = (pixel_rgb*)imgIn.accessPixels();
	pixel_rgb* outPointer = (pixel_rgb*)imgOut.accessPixels();
	parallel_for(blocked_range<int>(0, (int)numPixels, 1024), [&](const blocked_range<int>& range) {
		for (int i = range.begin(); i < range.end(); i++)
		{
			outPointer[i] = fn(inPointer[i]);
		}
	});

	// Save output image to disk
	return imgOut.save(outImagePath);
}



// PART 3

int PixelsThatMeetCriteriaSerial(char* inImagePath, function<bool(unsigned char)> fn)
{
	// Load input images from disk as FreeImagePlus images
	fipImage imgIn;
	imgIn.load(inImagePath);
	imgIn.convertToGrayscale();
	unsigned int numPixels = imgIn.getWidth() * imgIn.getHeight();

	int sum = 0;
	// A grayscale image contains 8 bits per pixel, which is the same as a char (signed or unsigned).
	unsigned char* inPointer = (unsigned char*)imgIn.accessPixels();
	for (unsigned int pixel = 0; pixel < numPixels; pixel++, inPointer++)
	{
		if (fn(*inPointer)) sum++;
	}
	return sum;
}

int PixelsThatMeetCriteriaParallel(char* inImagePath, function<bool(float)> fn)
{
	// Load input images from disk as FreeImagePlus images
	fipImage imgIn;
	imgIn.load(inImagePath);
	imgIn.convertToFloat();
	int numPixels = imgIn.getWidth() * imgIn.getHeight();

	// Use parallel reduce to quickly sum values
	float* inPointer = (float*)imgIn.accessPixels();
	int sum = parallel_reduce(
		blocked_range<int>(0, numPixels, 1024),
		0.0f,

		[&](const blocked_range<int>& range, int initValue) {
			for (int i = range.begin(); i != range.end(); i++)
			{
				if (fn(inPointer[i])) initValue++;
			}
			return initValue;
		},

		[&](int a, int b) {
			return a + b;
		}
	);

	return sum;
}

bool MaskInvertSerial(char* inImagePath, char* maskImagePath, char* outImagePath)
{
	fipImage imgIn;
	fipImage imgMask;
	imgIn.load(inImagePath);
	imgMask.load(maskImagePath);
	imgIn.convertToGrayscale();
	imgMask.convertToGrayscale();

	unsigned int width = imgIn.getWidth();
	unsigned int height = imgMask.getHeight();
	unsigned int numPixels = width * height;

	unsigned char* inPointer = (unsigned char*)imgIn.accessPixels();
	unsigned char* maskPointer = (unsigned char*)imgMask.accessPixels();
	for (unsigned int pixel = 0; pixel < numPixels; pixel++, inPointer++, maskPointer++)
	{
		// Check if mask pixel is white. If it is, invert 
		if (*maskPointer == (unsigned char)0) *inPointer = ((unsigned char)255) - *inPointer;
	}
	
	return imgIn.save(outImagePath);
}



// MAIN FUNCTION

int main()
{
	int nt = task_scheduler_init::default_num_threads();
	task_scheduler_init T(nt);

	
	//Part 1 (Image Comparison): -----------------DO NOT REMOVE THIS COMMENT----------------------------//

	char TOP_1[] = "../Images/render_top_1.png";
	char TOP_2[] = "../Images/render_top_2.png";
	char BOTTOM_1[] = "../Images/render_bottom_1.png";
	char BOTTOM_2[] = "../Images/render_bottom_2.png";

	char OUT_STAGE_1_A[] = "../Images/stage1_top.png";
	char OUT_STAGE_1_B[] = "../Images/stage1_bottom.png";
	char OUT_STAGE_1_COMBINED[] = "../Images/stage1_combined.png";
	
	// Lambda functions for part 1
	auto and = [](float a, float b)->float { if (a == b) return 1.0f; else return 0.0f; };
	auto sum = [](float a, float b)->float { return (a/2.0f) + (b/2.0f); };

	auto and_rgb = [](pixel_rgb a, pixel_rgb b)->pixel_rgb { 
		pixel_rgb output;
		if (a.r == b.r && a.g == b.g && a.b == b.b) {
			output.r = (unsigned char)255;
			output.g = (unsigned char)255;
			output.b = (unsigned char)255;
		}
		else {
			output.r = (unsigned char)0;
			output.g = (unsigned char)0;
			output.b = (unsigned char)0;
		}
		return output;
	};

	auto sum_rgb = [](pixel_rgb a, pixel_rgb b)->pixel_rgb {
		pixel_rgb output;
		output.r = (a.r / (unsigned char)2) + (b.r / (unsigned char)2);
		output.g = (a.g / (unsigned char)2) + (b.g / (unsigned char)2);
		output.b = (a.b / (unsigned char)2) + (b.b / (unsigned char)2);
		return output;
	};

	char answer = 'N';

	cout << "Run part 1 (Y/N)? ";
	cin >> answer;
	if (answer == 'Y' || answer == 'y')
	{
		cout << "Run sequential solution? (Y/N)? ";
		cin >> answer;
		if (answer == 'Y' || answer == 'y')
		{
			// Sequential solution:
			std::chrono::steady_clock::time_point begins = std::chrono::steady_clock::now();
			CombineImagesSerial(TOP_1, TOP_2, OUT_STAGE_1_A, and_rgb);
			CombineImagesSerial(BOTTOM_1, BOTTOM_2, OUT_STAGE_1_B, and_rgb);
			CombineImagesSerial(OUT_STAGE_1_A, OUT_STAGE_1_B, OUT_STAGE_1_COMBINED, sum_rgb);
			std::chrono::steady_clock::time_point ends = std::chrono::steady_clock::now();
			
			auto durations = chrono::duration_cast<std::chrono::microseconds>(ends - begins).count();
			cout << "Duration for serial: " << durations << " microseconds" << endl;

			// TIME TESTS...
			//  Run 1: 61849068 microseconds (61.8 seconds)
			//  Run 2: 62057598 microseconds (62.1 seconds)
			//  Run 3: 62317908 microseconds (62.3 seconds)
			//  Run 4: 62419654 microseconds (62.4 seconds)
			//  Run 5: 62217542 microseconds (62.2 seconds)
			//  Average: 62172354 microseconds (62.2 seconds)
		}

		cout << "Run parallel solution? (Y/N)? ";
		cin >> answer;
		if (answer == 'Y' || answer == 'y')
		{
			// Parallel solution:
			std::chrono::steady_clock::time_point beginp = std::chrono::steady_clock::now();
			CombineImagesParallel(TOP_1, TOP_2, OUT_STAGE_1_A, nt, and);
			CombineImagesParallel(BOTTOM_1, BOTTOM_2, OUT_STAGE_1_B, nt, and);
			CombineImagesParallel(OUT_STAGE_1_A, OUT_STAGE_1_B, OUT_STAGE_1_COMBINED, nt, sum);
			std::chrono::steady_clock::time_point endp = std::chrono::steady_clock::now();
			
			auto durationp = chrono::duration_cast<std::chrono::microseconds>(endp - beginp).count();
			cout << "Duration for parallel: " << durationp << " microseconds" << endl;

			// TIME TESTS...
			//  Run 1: 18476175 (18.5 seconds)
			//  Run 2: 18615964 (18.6 seconds)
			//  Run 3: 18592525 (18.6 seconds)
			//  Run 4: 18568345 (18.6 seconds)
			//  Run 5: 18602575 (18.6 seconds)
			//  Average: 18571117 (18.6 seconds) (3.35x faster than sequential on average)
		}
	}


	//Part 2 (Blur & post-processing): -----------DO NOT REMOVE THIS COMMENT----------------------------//

	char OUT_STAGE_2_BLURRED[] = "../Images/stage2_blurred.png";
	char OUT_STAGE_2_THRESHOLD[] = "../Images/stage2_threshold.png";
	
	// Lambda functions for part 2
	auto binaryThreshold2 = [](pixel_rgb x)->pixel_rgb { 
		pixel_rgb black;
		pixel_rgb white;
		white.r, white.g, white.b = 255, 255, 255;
		if (x.r != 0 || x.g != 0 || x.b != 0) return white;
		else return black;
	};

	cout << "Run part 2 (Y/N)? ";
	cin >> answer;
	if (answer == 'Y' || answer == 'y')
	{
		cout << "Run serial solution (Y/N)? ";
		cin >> answer;
		if (answer == 'Y' || answer == 'y')
		{
			// Serial solutions
			std::chrono::steady_clock::time_point begins = std::chrono::steady_clock::now();
			ProcessImageStencilSerial(OUT_STAGE_1_COMBINED, OUT_STAGE_2_BLURRED, 2, GaussianBlur(0.8f, 3));
			std::chrono::steady_clock::time_point ends = std::chrono::steady_clock::now();

			auto durations = chrono::duration_cast<std::chrono::milliseconds>(ends - begins).count();
			cout << "Duration for serial stencil" << ": " << durations << " milliseconds" << endl;

			std::chrono::steady_clock::time_point begint = std::chrono::steady_clock::now();
			ApplyToImageSerial(OUT_STAGE_2_BLURRED, OUT_STAGE_2_THRESHOLD, binaryThreshold2);
			std::chrono::steady_clock::time_point endt = std::chrono::steady_clock::now();

			auto durationt = chrono::duration_cast<std::chrono::milliseconds>(endt - begint).count();
			cout << "Duration for parallel threshold" << ": " << durationt << " milliseconds" << endl;
		}

		cout << "Run parallel solution (Y/N)? ";
		cin >> answer;
		if (answer == 'Y' || answer == 'y')
		{
			// Parallel solutions
			std::chrono::steady_clock::time_point begins = std::chrono::steady_clock::now();
			//ProcessImageStencilParallel(OUT_STAGE_1_COMBINED, OUT_STAGE_2_BLURRED, 2, GaussianBlur(0.8f, 5));
			std::chrono::steady_clock::time_point ends = std::chrono::steady_clock::now();

			auto durations = chrono::duration_cast<std::chrono::milliseconds>(ends - begins).count();
			cout << "Duration for parallel stencil" << ": " << durations << " milliseconds" << endl;

			// TIME TESTS... (kernal radius of 2, meaning 5x5 kernal)
			//  Duration for parallel run 1 : 28426 milliseconds
			//  Duration for parallel run 2 : 28520 milliseconds
			//  Duration for parallel run 3 : 28607 milliseconds
			//  Duration for parallel run 4 : 28636 milliseconds
			//  Duration for parallel run 5 : 28610 milliseconds
			//  Average : 28560 (28.6 seconds)

			std::chrono::steady_clock::time_point begint = std::chrono::steady_clock::now();
			ApplyToImageParallel(OUT_STAGE_2_BLURRED, OUT_STAGE_2_THRESHOLD, binaryThreshold2);
			std::chrono::steady_clock::time_point endt = std::chrono::steady_clock::now();

			auto durationt = chrono::duration_cast<std::chrono::milliseconds>(endt - begint).count();
			cout << "Duration for parallel threshold" << ": " << durationt << " milliseconds" << endl;
		}
	}

	//Part 3 (Image Mask): -----------------------DO NOT REMOVE THIS COMMENT----------------------------//

	char OUT_STAGE_3[] = "../Images/stage3.png";

	// Lambdas for part 3
	auto checkPixelIsWhite = [](float x)->bool { if (x <= 0.0f) return true; else return false; };
	auto checkPixelIsWhiteGrayscale = [](unsigned char x)->bool { if (x == (unsigned char)0) return true; else return false; };

	cout << "Run part 3 (Y/N)? ";
	cin >> answer;
	if (answer == 'Y' || answer == 'y')
	{
		cout << "Run serial solution (Y/N)? ";
		cin >> answer;
		if (answer == 'Y' || answer == 'y')
		{
			int vals = PixelsThatMeetCriteriaSerial(OUT_STAGE_2_THRESHOLD, checkPixelIsWhiteGrayscale);
			cout << vals << " white pixels" << endl;
			cout << ((float)vals / (5000.0f * 7000.0f)) * 100.0f << "%";

			MaskInvertSerial(TOP_1, OUT_STAGE_2_THRESHOLD, OUT_STAGE_3);
		}

		cout << "Run parallel solution (Y/N)? ";
		cin >> answer;
		if (answer == 'Y' || answer == 'y')
		{
			int vals = PixelsThatMeetCriteriaParallel(OUT_STAGE_2_THRESHOLD, checkPixelIsWhite);
			cout << vals << " white pixels" << endl;
			cout << ((float)vals / (5000.0f * 7000.0f)) * 100.0f << "%";
		}
	}

	return 0;
}