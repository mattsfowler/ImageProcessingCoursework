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

// -------------------- TYPES --------------------

struct pixel_rgb
{
	alignas(1) unsigned char r = 0;
	alignas(1) unsigned char g = 0;
	alignas(1) unsigned char b = 0;
};

// generic functor parent
class StencilTask
{
public:
	virtual float operator()(float x, float y) = 0;
};

// Gaussian functor
//  use constructor to set Gaussian parameters 'sigma' and 'kernal pre-load radius' e.g. GaussianBlur MyGaussian = GaussianBlur(0.8f, 2)
//  once instantiated, object can be called as if it was a function e.g. MyGaussian(x_coord, y_coord)
// NOTE: 'kradius' is the number of pixels away from the origin that the mask is applied to, e.g. kradius=2 means a 5x5 matrix
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
				curRow.push_back(1.0f / (2.0f * M_PIf * pow(sigma, 2)) * exp(-((pow(x, 2) + pow(y, 2)) / (2.0f * pow(sigma, 2.0f)))));
			}
			kernel.push_back(curRow);
		}
	}

public:
	// 'kradius' is the number of values to the left and right of the origin in the pre-loaded mask matrix.
	// It is not a hard limit on the number of computable values.
	GaussianBlur(float sigma, int kradius = 0)
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

// A component of ComineImagesParallel. Combines only a sub-section of the given images.
void CombineSubImageAnd(pixel_rgb* aPointer, pixel_rgb* bPointer, pixel_rgb* outPointer, unsigned int numIterations)
{
	// White and black pixel definitions
	pixel_rgb WHITE_PIXEL;
	pixel_rgb BLACK_PIXEL;
	WHITE_PIXEL.r = 255; WHITE_PIXEL.g = 255; WHITE_PIXEL.b = 255;
	BLACK_PIXEL.r = 0; BLACK_PIXEL.g = 0; BLACK_PIXEL.b = 0;
	for (unsigned int pixel = 0; pixel < numIterations; pixel++, aPointer++, bPointer++, outPointer++)
	{
		if (aPointer->r == bPointer->r
			&& aPointer->g == bPointer->g
			&& aPointer->b == bPointer->b)
		{
			*aPointer = BLACK_PIXEL;
		}
		else
		{
			*aPointer = WHITE_PIXEL;
		}
	}
}
// A component of ComineImagesParallel. Combines only a sub-section of the given images.
void CombineSubImageSum(pixel_rgb* aPointer, pixel_rgb* bPointer, pixel_rgb* outPointer, unsigned int numIterations)
{
	for (unsigned int pixel = 0; pixel < numIterations; pixel++, aPointer++, bPointer++, outPointer++)
	{
		aPointer->r = (aPointer->r / (unsigned char)2) + (bPointer->r / (unsigned char)2);
		aPointer->g = (aPointer->g / (unsigned char)2) + (bPointer->g / (unsigned char)2);
		aPointer->b = (aPointer->b / (unsigned char)2) + (bPointer->b / (unsigned char)2);
	}
}


int main()
{
	int nt = task_scheduler_init::default_num_threads();
	task_scheduler_init T(nt);

	// Image file paths
	char IN_TOP_1[] = "../Images/render_top_1.png";
	char IN_TOP_2[] = "../Images/render_top_2.png";
	char IN_BOTTOM_1[] = "../Images/render_bottom_1.png";
	char IN_BOTTOM_2[] = "../Images/render_bottom_2.png";
	char OUT_STAGE_1_TOP[] = "../Images/stage1_top.png";
	char OUT_STAGE_1_BOTTOM[] = "../Images/stage1_bottom.png";
	char OUT_STAGE_1_COMBINED[] = "../Images/stage1_combined.png";
	char OUT_STAGE_2_BLURRED[] = "../Images/stage2_blurred.png";
	char OUT_STAGE_2_THRESHOLD[] = "../Images/stage2_threshold.png";
	char OUT_STAGE_3[] = "../Images/stage3_final.png";

	// Number of times each stage will be executed
	const int STAGE_1_SEQ_ITERATIONS = 3;
	const int STAGE_1_PAR_ITERATIONS = 3;
	const int STAGE_2_SEQ_ITERATIONS = 1;
	const int STAGE_2_PAR_ITERATIONS = 1;
	const int STAGE_3_SEQ_ITERATIONS = 3;
	const int STAGE_3_PAR_ITERATIONS = 3;
	std::chrono::steady_clock::time_point start;
	std::chrono::steady_clock::time_point end;
	float average = 0.0f;

	// FIP image objects:
	fipImage imgInputA;
	fipImage imgInputB;
	fipImage imgOutput;
	// White and black pixel definitions
	pixel_rgb WHITE_PIXEL;
	pixel_rgb BLACK_PIXEL;
	WHITE_PIXEL.r = 255; WHITE_PIXEL.g = 255; WHITE_PIXEL.b = 255;
	BLACK_PIXEL.r = 0; BLACK_PIXEL.g = 0; BLACK_PIXEL.b = 0;
	
	//Part 1 (Image Comparison): -----------------DO NOT REMOVE THIS COMMENT----------------------------//
	

	// Part 1 sequential solution:
	cout << "Part one (serial) (" << STAGE_1_SEQ_ITERATIONS << " runs):" << endl;
	for (int i = 0; i < STAGE_1_SEQ_ITERATIONS; i++)
	{
		// COMBINE TOP
		// Load images using Free Image Plus library
		imgInputA.load(IN_TOP_1);
		imgInputB.load(IN_TOP_2);
		imgInputA.convertTo24Bits();
		imgInputB.convertTo24Bits();
		// Image dimensions
		unsigned int numPixels = imgInputA.getWidth() * imgInputA.getHeight();
		// Iterate over each pixel
		pixel_rgb* aPointer = (pixel_rgb*)imgInputA.accessPixels(); // also acts as the output image
		pixel_rgb* bPointer = (pixel_rgb*)imgInputB.accessPixels();
		start = std::chrono::steady_clock::now();
		for (unsigned int pixel = 0; pixel < numPixels; pixel++, aPointer++, bPointer++)
		{
			if (aPointer->r == bPointer->r
				&& aPointer->g == bPointer->g
				&& aPointer->b == bPointer->b)
			{
				*aPointer = BLACK_PIXEL;
			}
			else
			{
				*aPointer = WHITE_PIXEL;
			}
		}
		end = std::chrono::steady_clock::now();
		// Save output image to disk
		imgInputA.save(OUT_STAGE_1_TOP);

		auto duration_p1_s1 = chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
		cout << "  Run " << i + 1 << ": (top) " << duration_p1_s1 << "ms, ";
		average += (float)duration_p1_s1;

		// COMBINE BOTTOM
		// Load images using Free Image Plus library
		imgInputA.load(IN_BOTTOM_1);
		imgInputB.load(IN_BOTTOM_1);
		imgInputA.convertTo24Bits();
		imgInputB.convertTo24Bits();
		// Iterate over each pixel
		aPointer = (pixel_rgb*)imgInputA.accessPixels(); // also acts as the output image
		bPointer = (pixel_rgb*)imgInputB.accessPixels();
		start = std::chrono::steady_clock::now();
		for (unsigned int pixel = 0; pixel < numPixels; pixel++, aPointer++, bPointer++)
		{
			if (aPointer->r == bPointer->r
				&& aPointer->g == bPointer->g
				&& aPointer->b == bPointer->b)
			{
				*aPointer = BLACK_PIXEL;
			}
			else
			{
				*aPointer = WHITE_PIXEL;
			}
		}
		end = std::chrono::steady_clock::now();
		// Save output image to disk
		imgInputA.save(OUT_STAGE_1_BOTTOM);

		auto duration_p1_s2 = chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
		cout << "(bottom) " << duration_p1_s2 << "ms, ";
		average += (float)duration_p1_s2;

		// COMBINE TOP AND BOTTOM
		// Load images using Free Image Plus library
		imgInputA.load(OUT_STAGE_1_TOP);
		imgInputB.load(OUT_STAGE_1_BOTTOM);
		imgInputA.convertTo24Bits();
		imgInputB.convertTo24Bits();
		// Iterate over each pixel
		aPointer = (pixel_rgb*)imgInputA.accessPixels(); // also acts as the output image
		bPointer = (pixel_rgb*)imgInputB.accessPixels();
		start = std::chrono::steady_clock::now();
		for (unsigned int pixel = 0; pixel < numPixels; pixel++, aPointer++, bPointer++)
		{
			aPointer->r = (aPointer->r / (unsigned char)2) + (bPointer->r / (unsigned char)2);
			aPointer->g = (aPointer->g / (unsigned char)2) + (bPointer->g / (unsigned char)2);
			aPointer->b = (aPointer->b / (unsigned char)2) + (bPointer->b / (unsigned char)2);
		}
		end = std::chrono::steady_clock::now();
		// Save output image to disk
		imgInputA.save(OUT_STAGE_1_COMBINED);

		auto duration_p1_s3 = chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
		cout << "(combine) " << duration_p1_s3 << "ms" << endl;
		average += (float)duration_p1_s3;
	}

	if (STAGE_1_SEQ_ITERATIONS > 0) average = average / (float)STAGE_1_SEQ_ITERATIONS;
	cout << "Average: " << average << endl << endl;
	average = 0.0f;


	// Part 1 parallel solution:
	cout << "Part one (parallel) (" << STAGE_1_PAR_ITERATIONS << " runs):" << endl;
	for (int i = 0; i < STAGE_1_PAR_ITERATIONS; i++)
	{	
		// COMBINE TOP
		// Load input images from disk as FreeImagePlus images
		imgInputA.load(IN_TOP_1);
		imgInputB.load(IN_TOP_2);
		imgInputA.convertTo24Bits();
		imgInputB.convertTo24Bits();
		// Create an empty output image with the same dimensions as the inputs
		unsigned int numPixels = imgInputA.getWidth() * imgInputA.getHeight();
		// Create threads to process smaller sub-images
		vector<thread> threads;
		unsigned int stepsize = numPixels / nt;
		unsigned int remainder = numPixels % nt;
		// Split image into segments that can be processed by different threads
		pixel_rgb* aPointer = (pixel_rgb*)imgInputA.accessPixels();
		pixel_rgb* bPointer = (pixel_rgb*)imgInputB.accessPixels();
		start = std::chrono::steady_clock::now();
		for (int i = 0; i < nt; i++)
		{
			if (i == 0)
			{
				threads.push_back(thread(CombineSubImageAnd, aPointer, bPointer, aPointer, stepsize + remainder));
				aPointer += stepsize + remainder;
				bPointer += stepsize + remainder;
			}
			else
			{
				threads.push_back(thread(CombineSubImageAnd, aPointer, bPointer, aPointer, stepsize));
				aPointer += stepsize;
				bPointer += stepsize;
			}
		}
		// Wait for the threads to finish executing
		for (auto& thread : threads)
		{
			thread.join();
		}
		end = std::chrono::steady_clock::now();
		// Save output image to disk
		imgInputA.save(OUT_STAGE_1_TOP);

		auto duration_p1_p1 = chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
		cout << "  Run " << i + 1 << ": (top)" << duration_p1_p1 << "ms, ";
		average += (float)duration_p1_p1;

		// COMBINE BOTTOM
		// Load input images from disk as FreeImagePlus images
		imgInputA.load(IN_BOTTOM_1);
		imgInputB.load(IN_BOTTOM_2);
		imgInputA.convertTo24Bits();
		imgInputB.convertTo24Bits();
		// Reset thread vector
		threads.clear();
		// Split image into segments that can be processed by different threads
		aPointer = (pixel_rgb*)imgInputA.accessPixels();
		bPointer = (pixel_rgb*)imgInputB.accessPixels();
		start = std::chrono::steady_clock::now();
		for (int i = 0; i < nt; i++)
		{
			if (i == 0)
			{
				threads.push_back(thread(CombineSubImageAnd, aPointer, bPointer, aPointer, stepsize + remainder));
				aPointer += stepsize + remainder;
				bPointer += stepsize + remainder;
			}
			else
			{
				threads.push_back(thread(CombineSubImageAnd, aPointer, bPointer, aPointer, stepsize));
				aPointer += stepsize;
				bPointer += stepsize;
			}
		}
		// Wait for the threads to finish executing
		for (auto& thread : threads)
		{
			thread.join();
		}
		end = std::chrono::steady_clock::now();
		// Save output image to disk
		imgInputA.save(OUT_STAGE_1_BOTTOM);

		auto duration_p1_p2 = chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
		cout << "(bottom) " << duration_p1_p2 << "ms, ";
		average += (float)duration_p1_p2;

		// COMBINE TOP AND BOTTOM
		// Load input images from disk as FreeImagePlus images
		imgInputA.load(OUT_STAGE_1_TOP);
		imgInputB.load(OUT_STAGE_1_BOTTOM);
		imgInputA.convertTo24Bits();
		imgInputB.convertTo24Bits();
		// Reset thread vector
		threads.clear();
		// Split image into segments that can be processed by different threads
		aPointer = (pixel_rgb*)imgInputA.accessPixels();
		bPointer = (pixel_rgb*)imgInputB.accessPixels();
		start = std::chrono::steady_clock::now();
		for (int i = 0; i < nt; i++)
		{
			if (i == 0)
			{
				threads.push_back(thread(CombineSubImageSum, aPointer, bPointer, aPointer, stepsize + remainder));
				aPointer += stepsize + remainder;
				bPointer += stepsize + remainder;
			}
			else
			{
				threads.push_back(thread(CombineSubImageSum, aPointer, bPointer, aPointer, stepsize));
				aPointer += stepsize;
				bPointer += stepsize;
			}
		}
		// Wait for the threads to finish executing
		for (auto& thread : threads)
		{
			thread.join();
		}
		end = std::chrono::steady_clock::now();
		// Save output image to disk
		imgInputA.save(OUT_STAGE_1_COMBINED);

		auto duration_p1_p3 = chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
		cout << "(combine) " << duration_p1_p3 << "ms" << endl;
		average += (float)duration_p1_p3;
	}

	if (STAGE_1_PAR_ITERATIONS > 0) average = average / (float)STAGE_1_PAR_ITERATIONS;
	cout << "Average: " << average << endl << endl;
	average = 0.0f;


	//Part 2 (Blur & post-processing): -----------DO NOT REMOVE THIS COMMENT----------------------------//
	

	// blur parameters
	float sigma = 0.8f;
	int kernal_radius = 1;
	// Setup blur functor
	GaussianBlur BlurFunc = GaussianBlur(sigma, kernal_radius);

	// Part 2 sequential solution:
	cout << "Part two (serial) (" << STAGE_2_SEQ_ITERATIONS << " runs): " << endl;
	for (int i = 0; i < STAGE_2_SEQ_ITERATIONS; i++)
	{
		// GAUSSIAN BLUR
		// Load input image from disk into memory
		imgInputA.load(OUT_STAGE_1_COMBINED);
		imgInputA.convertToFloat();
		// Load empty output image into memory
		unsigned int width = imgInputA.getWidth();
		unsigned int height = imgInputA.getHeight();
		unsigned int numPixels = width * height;
		imgOutput = fipImage(FIT_FLOAT, width, height, 32);
		// Setup variables for accessing memory
		float* fInPointer = (float*)imgInputA.accessPixels();
		float* fOutPointer = (float*)imgOutput.accessPixels();
		unsigned int stencilIndex = 0;
		float sum = 0.0f;
		// Iterate over all pixels linearly
		start = std::chrono::steady_clock::now();
		for (int yOrigin = 0; yOrigin < height; yOrigin++)
		{
			for (int xOrigin = 0; xOrigin < width; xOrigin++, fOutPointer++)
			{
				sum = 0.0f;

				for (int yStencil = -(kernal_radius); yStencil <= kernal_radius; yStencil++)
				{
					// Make sure that the stencil's y-coord is within range. If not, snap to the nearest border pixel.
					int absoluteYStencil = yOrigin + yStencil;
					if (absoluteYStencil < 0) absoluteYStencil = 0;
					if (absoluteYStencil >= height) absoluteYStencil = height - 1;

					for (int xStencil = -(kernal_radius); xStencil <= kernal_radius; xStencil++)
					{
						// Make sure that the stencil's x-coord is within range. If not, snap to the nearest border pixel.
						int absoluteXStencil = xOrigin + xStencil;
						if (absoluteXStencil < 0) absoluteXStencil = 0;
						if (absoluteXStencil >= width) absoluteXStencil = width - 1;

						stencilIndex = (absoluteYStencil * width) + absoluteXStencil;
						sum += fInPointer[stencilIndex] * BlurFunc(xStencil, yStencil);
					}
				}
				*fOutPointer = sum;
			}
		}
		end = std::chrono::steady_clock::now();
		// Save output image to disk
		imgOutput.convertToType(FREE_IMAGE_TYPE::FIT_BITMAP);
		imgOutput.convertTo24Bits();
		imgOutput.save(OUT_STAGE_2_BLURRED);

		auto duration_p2_s1 = chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
		cout << "  Run " << i+1 << ": (gaussian) " << duration_p2_s1 << "ms, ";
		
		// THRESHOLD (any non-black pixels become white)
		// Load input images from disk as FreeImagePlus images
		imgInputA.load(OUT_STAGE_2_BLURRED);
		imgInputA.convertTo24Bits();
		// Iterate over each pixel and apply the given function
		pixel_rgb* inPointer = (pixel_rgb*)imgInputA.accessPixels();
		start = std::chrono::steady_clock::now();
		for (unsigned int pixel = 0; pixel < numPixels; pixel++, inPointer++)
		{
			if (inPointer->r != 0 || inPointer->g != 0 || inPointer->b != 0)
			{
				*inPointer = WHITE_PIXEL;
			}
		}
		end = std::chrono::steady_clock::now();
		// Save output image to disk
		imgInputA.save(OUT_STAGE_2_THRESHOLD);

		auto duration_p2_s2 = chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
		cout << "(threshold) " << ": " << duration_p2_s2 << "ms" << endl;
		average += (float)duration_p2_s1 + (float)duration_p2_s2;
	}

	if (STAGE_2_SEQ_ITERATIONS > 0) average = average / (float)STAGE_2_SEQ_ITERATIONS;
	cout << "Average: " << average << endl << endl;
	average = 0.0f;


	// Part 2 parallel solution:
	cout << "Part two (parallel) (" << STAGE_2_PAR_ITERATIONS << " runs): " << endl;
	for (int i = 0; i < STAGE_2_PAR_ITERATIONS; i++)
	{
		// GAUSSIAN BLUR
		// Load input image
		imgInputA.load(OUT_STAGE_2_BLURRED);
		imgInputA.convertToFloat();
		// Load empty output image into memory
		unsigned int width = imgInputA.getWidth();
		unsigned int height = imgInputA.getHeight();
		unsigned int numPixels = width * height;
		imgOutput = fipImage(FIT_FLOAT, width, height, 32);
		// Iterate over all pixels using TBB parallel_for
		float* fInPointer = (float*)imgInputA.accessPixels();
		float* fOutPointer = (float*)imgOutput.accessPixels();
		// For each pixel in the input...
		start = std::chrono::steady_clock::now();
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
						for (int kx = -kernal_radius; kx <= kernal_radius; kx++)
						{
							xsum = x + kx;
							if (xsum < 0) xsum = 0;
							if (xsum >= width) xsum = width - 1;
							for (int ky = -kernal_radius; ky <= kernal_radius; ky++)
							{
								ysum = y + ky;
								if (ysum < 0) ysum = 0;
								if (ysum >= height) ysum = height - 1;

								fOutPointer[(y * width) + x] += fInPointer[(ysum * width) + xsum] * BlurFunc(kx, ky);
							}
						}
					}
				}
			}
		);
		end = std::chrono::steady_clock::now();
		// Save output image to disk
		imgOutput.convertToType(FREE_IMAGE_TYPE::FIT_BITMAP);
		imgOutput.convertTo24Bits();
		imgOutput.save(OUT_STAGE_2_BLURRED);

		auto duration_p2_p1 = chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
		cout << "  Run " << i + 1 << ": (gaussian) " << duration_p2_p1 << "ms, ";
		
		// THRESHOLD
		// Load input images from disk as FreeImagePlus images
		imgInputA.load(OUT_STAGE_2_BLURRED);
		imgInputA.convertTo24Bits();
		// Iterate over each pixel and apply the given function
		pixel_rgb* inPointer = (pixel_rgb*)imgInputA.accessPixels();
		start = std::chrono::steady_clock::now();
		parallel_for(blocked_range<int>(0, (int)numPixels, 1024), [&](const blocked_range<int>& range) {
			for (int i = range.begin(); i < range.end(); i++)
			{
				if (inPointer[i].r != 0 || inPointer[i].g != 0 || inPointer[i].b != 0)
				{
					inPointer[i] = WHITE_PIXEL;
				}
			}
		});
		end = std::chrono::steady_clock::now();
		// Save output image to disk
		imgInputA.save(OUT_STAGE_2_THRESHOLD);

		auto duration_p2_p2 = chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
		cout << "(threshold) " << ": " << duration_p2_p2 << "ms" << endl;
		average += (float)duration_p2_p1 + (float)duration_p2_p2;
	}

	if (STAGE_2_PAR_ITERATIONS > 0) average = average / (float)STAGE_2_PAR_ITERATIONS;
	cout << "Average: " << average << endl << endl;
	average = 0.0f;
	

	//Part 3 (Image Mask): -----------------------DO NOT REMOVE THIS COMMENT----------------------------//


	// Part 3 sequential solution:
	cout << "Part three (serial) (" << STAGE_3_SEQ_ITERATIONS << " runs): " << endl;
	for (int i = 0; i < STAGE_3_SEQ_ITERATIONS; i++)
	{
		cout << "  Run " << i+1 << ": (count pixels) ";
		
		// NUMBER OF WHITE PIXELS
		// Load input images from disk as FreeImagePlus images
		imgInputB.load(OUT_STAGE_2_THRESHOLD);
		imgInputB.convertTo24Bits();
		// Image dimensions
		unsigned int numPixels = imgInputB.getWidth() * imgInputB.getHeight();
		// Iterate over each pixel and count how many pixels meet the given criteria
		int sum = 0;
		pixel_rgb* inPointer = (pixel_rgb*)imgInputB.accessPixels();
		start = std::chrono::steady_clock::now();
		for (unsigned int pixel = 0; pixel < numPixels; pixel++, inPointer++)
		{
			if (inPointer->r == 255
				&& inPointer->g == 255
				&& inPointer->b == 255)
			{
				sum++;
			}
		}
		end = std::chrono::steady_clock::now();

		cout << sum << " pixels (" << ((float)sum / (5000.0f * 7000.0f)) * 100.0f << "%) - ";
		auto duration_p3_s1 = chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
		cout << duration_p3_s1 << "ms, ";

		// Load the input image and the mask from the disk
		imgInputA.load(IN_TOP_1);
		imgInputA.convertTo24Bits();
		// For each pixel in the input, check that the corresponding pixel in the mask meets the condition
		// If it does invert the pixel in the input image
		inPointer = (pixel_rgb*)imgInputA.accessPixels();
		pixel_rgb* maskPointer = (pixel_rgb*)imgInputB.accessPixels();
		start = std::chrono::steady_clock::now();
		for (unsigned int pixel = 0; pixel < numPixels; pixel++, inPointer++, maskPointer++)
		{
			if (maskPointer->r == 255
				&& maskPointer->g == 255
				&& maskPointer->b == 255)
			{
				inPointer->r = 255 - inPointer->r;
				inPointer->g = 255 - inPointer->g;
				inPointer->b = 255 - inPointer->b;
			}
		}
		end = std::chrono::steady_clock::now();
		// Save result to the disk
		imgInputA.save(OUT_STAGE_3);

		auto duration_p3_s2 = chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
		cout << "(mask filter) " << duration_p3_s2 << "ms" << endl;
		average += (float)duration_p3_s1 + (float)duration_p3_s2;
	}

	if (STAGE_3_SEQ_ITERATIONS > 0) average = average / (float)STAGE_3_SEQ_ITERATIONS;
	cout << "Average: " << average << endl << endl;
	average = 0.0f;

	
	// Part 3 parallel solution:
	cout << "Part three (parallel) (" << STAGE_3_PAR_ITERATIONS << " runs): " << endl;
	for (int i = 0; i < STAGE_3_PAR_ITERATIONS; i++)
	{
		cout << "  Run " << i + 1 << ": (count pixels) ";
		
		// Load input images from disk as FreeImagePlus images
		imgInputB.load(OUT_STAGE_2_THRESHOLD);
		imgInputB.convertTo24Bits();
		// Image dimensions
		int numPixels = imgInputB.getWidth() * imgInputB.getHeight();
		pixel_rgb* inPointer = (pixel_rgb*)imgInputB.accessPixels();
		// Sum together values using parallel reduce
		start = std::chrono::steady_clock::now();
		int sum = parallel_reduce(
			blocked_range<int>(0, numPixels, 1024),
			0,

			[&](const blocked_range<int>& range, int initValue) {
				for (int i = range.begin(); i != range.end(); i++)
				{
					if (inPointer[i].r == 255
						&& inPointer[i].g == 255
						&& inPointer[i].b == 255)
					{
						initValue++;
					}
				}
				return initValue;
			},

			[&](int a, int b) {
				return a + b;
			}
		);
		end = std::chrono::steady_clock::now();

		cout << sum << " pixels (" << ((float)sum / (5000.0f * 7000.0f)) * 100.0f << "%) - ";
		auto duration_p3_p1 = chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
		cout << duration_p3_p1 << "ms, ";
		
		// Load the input image and the mask from the disk
		imgInputA.load(IN_TOP_1);
		imgInputA.convertTo24Bits();
		// For each pixel in the input, check that the corresponding pixel in the mask meets the condition
		// If it does invert the pixel in the input image
		inPointer = (pixel_rgb*)imgInputA.accessPixels();
		pixel_rgb* maskPointer = (pixel_rgb*)imgInputB.accessPixels();
		start = std::chrono::steady_clock::now();
		parallel_for(blocked_range<int>(0, numPixels, 1024), [&](blocked_range<int>& range) {
			for (int p = range.begin(); p < range.end(); p++)
			{
				if (maskPointer[p].r == 255
					&& maskPointer[p].g == 255
					&& maskPointer[p].b == 255)
				{
					inPointer[p].r = 255 - inPointer[p].r;
					inPointer[p].g = 255 - inPointer[p].g;
					inPointer[p].b = 255 - inPointer[p].b;
				}
			}
		});
		end = std::chrono::steady_clock::now();
		// Save result to the disk
		imgInputA.save(OUT_STAGE_3);

		auto duration_p3_p2 = chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
		cout << "(mask filter) " << duration_p3_p2 << "ms" << endl;
		average += (float)duration_p3_p1 + (float)duration_p3_p2;
	}

	if (STAGE_3_PAR_ITERATIONS > 0) average = average / (float)STAGE_3_PAR_ITERATIONS;
	cout << "Average: " << average << "ms" << endl << endl;
	average = 0.0f;
	
	cin;
	return 0;
}