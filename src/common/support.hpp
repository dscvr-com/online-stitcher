#include <string>
#include <sstream>

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
}

#endif
