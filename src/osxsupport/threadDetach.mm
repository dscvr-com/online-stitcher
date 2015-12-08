#import <Cocoa/Cocoa.h>
#import "threadDetach.h"

@interface ThreadDetacher:NSObject
{
}

-(id)init;
-(void)Run;
@end

@implementation ThreadDetacher
-(id)init
{
  return [super init];
}

-(void)Run
{
 //Do nothing
}
@end

void MarkAsMultithreaded(){
    ThreadDetacher* _thread = [[ThreadDetacher alloc] init];
    [NSThread detachNewThreadSelector:@selector(Run)
                       toTarget:_thread
                     withObject:nil];
}
