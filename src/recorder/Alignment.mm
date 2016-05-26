#include <opencv2/opencv.hpp>
#import <GLKit/GLKit.h>
#import <Foundation/foundation.h>
#include <vector>
#include <string>
#define OPTONAUT_TARGET_PHONE

#include "stitcher.hpp"
#include "Alignment.h"
#include "globalAlignment.hpp"
#include "Stores.h"
#include "CommonInternal.h"
#include "progressCallback.hpp"
#include "projection.hpp"
#include "panoramaBlur.hpp"

@implementation Alignment

-(id)init {
    self = [super init];
    return self;
};

struct AlignmentCancellation {
};

-(void)align {
    optonaut::GlobalAlignment globalaligner(Stores::post, Stores::left, Stores::right);
    
    try {
        globalaligner.Finish();
        
    } catch (AlignmentCancellation c) { }
};
@end