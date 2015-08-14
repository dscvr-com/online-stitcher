#include <algorithm>
#include <string>
#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>

#include "lib/tinyxml2/tinyxml2.h"
#include "image.hpp"

#ifndef OPTONAUT_IO_HEADER
#define OPTONAUT_IO_HEADER

namespace optonaut {

    int IdFromFileName(const string &in);
	bool StringEndsWith(const std::string& a, const std::string& b);

	ImageP ImageFromFile(std::string path);

	template <typename T>
	void BufferFromStringFile(T buf[], int len, std::string file) {
	    std::ifstream input(file);

	    for(int i = 0; i < len; i++) {
	        input >> buf[i];
	    }
	}

	void BufferFromBinFile(unsigned char buf[], int len, std::string file);
}

#endif
