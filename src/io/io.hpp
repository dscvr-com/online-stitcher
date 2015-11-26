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

    int IdFromFileName(const std::string &in);
    void CreateDirectories(const std::string &path);
    void DeleteDirectories(const std::string &path);
    bool IsDirectory(const std::string &path);

    InputImageP InputImageFromFile(const std::string &path, bool shallow = true);
    void InputImageToFile(const InputImageP image, const std::string &path);
    StitchingResultP StitchingResultFromFile(const std::string &path, const std::string &extension);
    void StitchingResultToFile(StitchingResultP image, const std::string &path, const std::string &extension, bool maskOnly = false);
 	bool FileExists(const std::string &fileName);

	template <typename T>
	void BufferFromStringFile(T buf[], int len, std::string file) {
	    std::ifstream input(file);

	    for(int i = 0; i < len; i++) {
	        input >> buf[i];
	    }
	}

	void BufferFromBinFile(unsigned char buf[], size_t len, std::string file);
	void BufferToBinFile(unsigned char buf[], size_t len, std::string file);
    
    void SaveExposureMap(const std::map<size_t, double> &exposure, const std::string &path);
    std::map<size_t, double> LoadExposureMap(const std::string &path);
    
    template <typename T>
    void SaveListGeneric(const std::vector<T> input, const std::string &path);
    template <typename T>
    std::vector<T> LoadListGeneric(const std::string &path);
    
    void SaveRingMap(const std::vector<std::vector<InputImageP>> &rings, const std::string &path);
    std::vector<std::vector<size_t>> LoadRingMap(const std::string &path);
    std::vector<InputImageP> LoadAllImagesFromDirectory(const std::string &path, const std::string &extension);
    void SaveImage(Image &image, const std::string &path);

    void SaveIntList(const std::vector<int> &vals, const std::string &path);
    std::vector<int> LoadIntList(const std::string &path);
}

#endif
