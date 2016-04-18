#include <algorithm>
#include <string>
#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>

#include "../common/image.hpp"
#include "../stitcher/stitchingResult.hpp"

#include "inputImage.hpp"

#ifndef OPTONAUT_IO_HEADER
#define OPTONAUT_IO_HEADER

namespace optonaut {

    /*
     * Reads the ID from a given filename. 
     */
    int IdFromFileName(const std::string &in);

    /*
     * Creates all non-existing directories in the given path. 
     */
    void CreateDirectories(const std::string &path);
    
    /*
     * Deletes the given directory. 
     */
    void DeleteDirectories(const std::string &path);

    /*
     * Checks wether the given path is a directory. 
     */
    bool IsDirectory(const std::string &path);
 	
    /*
     * Returns true if a file exists. 
     */ 
    bool FileExists(const std::string &fileName);

    /*
     * Reads an input iamge from file. Tries to find a image data file
     * by changing the extension to json. 
     *
     * @param path The path of the image file. 
     * @param shallow True, if the image should be loaded in a shallow way. 
     * That means that the image data is not actually loaded, but instead
     * the image source is saved for later loading. 
     */
    InputImageP InputImageFromFile(const std::string &path, bool shallow = true);

    /*
     * Writes an input image to a file. Also creates a data file at the same location.
     *
     * @param image The image to save.
     * @param path The destination path. 
     */
    void InputImageToFile(const InputImageP image, const std::string &path);

    /*
     * Reads a stitching result from a file.
     */
    StitchingResultP StitchingResultFromFile(const std::string &path, const std::string &extension);

    /*
     * Writes a stitching result to a file. 
     */
    void StitchingResultToFile(StitchingResultP image, const std::string &path, const std::string &extension, bool maskOnly = false);

    /*
     * Reads a file and parses file contents to T. 
     *
     * @tparam T The type to parse file contents to.
     * @param buf The buffer to store the results to.
     * @param len The count of elements to read.
     * @param file The path of the file to read.   
     */
	template <typename T>
	void BufferFromStringFile(T buf[], int len, std::string file) {
	    std::ifstream input(file);

	    for(int i = 0; i < len; i++) {
	        input >> buf[i];
	    }
	}

    /*
     * Reads binary data from a file. 
     */
	void BufferFromBinFile(unsigned char buf[], size_t len, std::string file);
    /*
     * Writes binary data to a file. 
     */
	void BufferToBinFile(unsigned char buf[], size_t len, std::string file);
    
    /*
     * Saves an exposure map to the given path. 
     */
    void SaveExposureMap(const std::map<size_t, double> &exposure, const std::string &path);
    
    /*
     * Loads an exposure map from the given path. 
     */
    std::map<size_t, double> LoadExposureMap(const std::string &path);
    
    /*
     * Saves a vector to a file. 
     */
    template <typename T>
    void SaveListGeneric(const std::vector<T> input, const std::string &path);

    /*
     * Loads a vector from a file. 
     */
    template <typename T>
    std::vector<T> LoadListGeneric(const std::string &path);
   
    /*
     * Saves aring map to a given file.
     */ 
    void SaveRingMap(const std::vector<std::vector<InputImageP>> &rings, const std::string &path);
    
    /*
     * Loads a ring map from a file. 
     */
    std::vector<std::vector<size_t>> LoadRingMap(const std::string &path);

    /*
     * Loads all images from a directory. 
     */
    std::vector<InputImageP> LoadAllImagesFromDirectory(const std::string &path, const std::string &extension);

    /*
     * Saves an image to a file. 
     */
    void SaveImage(Image &image, const std::string &path);

    /*
     * Saves a list of ints to a file. 
     */
    void SaveIntList(const std::vector<int> &vals, const std::string &path);

    /*
     * Loads a list of ints to a file. 
     */
    std::vector<int> LoadIntList(const std::string &path);
}

#endif
