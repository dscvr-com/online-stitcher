#include <string>
#include <sstream>

#ifndef OPTONAUT_SUPPORT_HEADER
#define OPTONAUT_SUPPORT_HEADER

namespace optonaut {
	inline bool StringEndsWith(const std::string& a, const std::string& b) {
	    if (b.size() > a.size()) return false;
	    return std::equal(a.begin() + a.size() - b.size(), a.end(), b.begin());
	}

    inline std::string GetDirectoryName(const std::string &dir) {
        size_t last = dir.find_last_of("/"); 

        if(last == std::string::npos)
            return "";
        else
            return dir.substr(0, last);
    }
    
    template <typename T>
	inline T ParseFromCharArray(const char* data) {
		T val;
        std::istringstream text(data);
		text >> val;
		return val;
	}
    
    inline int ParseInt(const std::string &data) {
        return ParseFromCharArray<int>(data.c_str());
    }

    
    template <typename T>
    inline std::string ToString(T i) {
        std::ostringstream text;
        text << i;
        return text.str();
    }
}

#endif
