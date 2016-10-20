#include <string>
#include <sstream>
#include <iostream>
#include <opencv2/core.hpp>

#ifndef OPTONAUT_SUPPORT_HEADER
#define OPTONAUT_SUPPORT_HEADER

namespace optonaut {
    /*
     * @returns True, if string a ends with string b. False otherwise. 
     */
	inline bool StringEndsWith(const std::string& a, const std::string& b) {
	    if (b.size() > a.size()) return false;
	    return std::equal(a.begin() + a.size() - b.size(), a.end(), b.begin());
	}

    /*
     * Gets the name of the directory from a path. 
     *
     * @param dir The path to get the directory name for. 
     *
     * @returns the path without the file name. 
     */
    inline std::string GetDirectoryName(const std::string &dir) {
        size_t last = dir.find_last_of("/"); 

        if(last == std::string::npos)
            return "";
        else
            return dir.substr(0, last);
    }
   
    /*
     * Parses an object or struct with the given type from the given char array using
     * iostream.
     *
     * @tparam T The type to parse for.
     * @param data The data to parse. 
     *
     * @returns The parsed data.
     */ 
    template <typename T>
	inline T ParseFromCharArray(const char* data) {
		T val;
        std::istringstream text(data);
		text >> val;
		return val;
	}
   
    /*
     * Parses an int from a given string.
     *
     * @param data The string to parse.
     *
     * @returns The parsed integer. 
     */ 
    inline int ParseInt(const std::string &data) {
        return ParseFromCharArray<int>(data.c_str());
    }

   
    /*
     * Converts a struct or object to string using the iostream library.
     *
     * @tparam The type of the object to convert to string.
     * @param i The object to convert to string. 
     *
     * @returns The string representation of the given object. 
     */ 
    template <typename T>
    inline std::string ToString(T i) {
        std::ostringstream text;
        text << i;
        return text.str();
    }

    /*
     * Samples a subpixel from an image. 
     * No mat checking for speed. 
     */
    template <typename T>
    inline T SampleLinear(const cv::Mat& img, const float inx, const float iny)
    {
        int x = (int)inx;
        int y = (int)iny;

        if((float)x == inx && (float)y == iny) {
           // Special case - we have EXACTLY a single pixel. 
           return img.at<T>(y, x); 
        }

        int x0 = x;
        int x1 = x + 1;
        int y0 = y; 
        int y1 = y + 1; 

        if(y1 >= img.rows) {
            y1 = img.rows - 1;
        }
        if(x1 >= img.cols) {
            x1 = img.cols - 1;
        }

        float a = inx - (float)x;
        float c = iny - (float)y;

        return 
            (img.at<T>(y0, x0) * (1.f - a) + img.at<T>(y0, x1) * a) * (1.f - c) +
            (img.at<T>(y1, x0) * (1.f - a) + img.at<T>(y1, x1) * a) * c;
    }
    
    /*
     * Samples a subpixel from an image. 
     * No mat checking for speed. 
     */
    template <typename T>
    inline T SampleNearest(const cv::Mat& img, const float inx, const float iny)
    {
        return img.at<T>((int)(iny + 0.5f), (int)(inx + 0.5f));
    }
}

#endif
