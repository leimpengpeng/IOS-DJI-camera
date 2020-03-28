
//  CameraFPVViewController.m

//  DJISdkDemo

//

//  Copyright © 2015 DJI. All rights reserved.

//

/**
 
 *  This file demonstrates how to receive the video data from DJICamera and display the video using DJIVideoPreviewer.
 
 */

#import "CameraFPVViewController.h"
#import "DemoUtility.h"
#import "VideoPreviewerSDKAdapter.h"
#import <DJIWidget/DJIVideoPreviewer.h>
#import <DJISDK/DJISDK.h>
#import <CoreML/CoreML.h>
#import <Vision/Vision.h>
#import "MobileNet_050_160.h"
@interface CameraFPVViewController () <DJICameraDelegate, VideoFrameProcessor>{
    NSInteger occurrences;
    MLModel *model;
    VNCoreMLModel *m;
    VNCoreMLRequest *rq;
    NSMutableArray *startTimes;
    
}



@property (weak, nonatomic) IBOutlet UILabel *messageLabel;
@property (weak, nonatomic) IBOutlet UILabel *obj2;
@property (weak, nonatomic) IBOutlet UILabel *obj3;

@property (nonatomic) unsigned long numberOfResults;
@property (retain, nonatomic) NSArray *results;
@property(nonatomic, weak) IBOutlet UIView* fpvView;
@property(nonatomic, assign) BOOL needToSetMode;
@property(nonatomic) VideoPreviewerSDKAdapter *previewerAdapter;
@property(atomic) CVPixelBufferRef currentPixelBuffer;
@property (weak, nonatomic) IBOutlet UISegmentedControl *DecodeModeSegment;
@end
@implementation CameraFPVViewController
- (void)viewDidLoad {
    [super viewDidLoad];
    [self.DecodeModeSegment setHidden:YES]  ;
    [DJIVideoPreviewer instance].enableHardwareDecode = 1;
    
    //Loading mobilenet model
    model = [[[MobileNet_050_160 alloc] init] model];
    occurrences = 0;
    m = [VNCoreMLModel modelForMLModel: model error:nil];
    rq = [[VNCoreMLRequest alloc] initWithModel: m completionHandler: (VNRequestCompletionHandler) ^(VNRequest *request, NSError *error){
        
        dispatch_async(dispatch_get_main_queue(), ^{
            NSTimeInterval start, stop;
            stop = [[NSDate date] timeIntervalSince1970];
            
            start = [[startTimes objectAtIndex: 0] doubleValue];
            
            [startTimes removeObjectAtIndex: 0];
            self.messageLabel.text = @"done";
            self.numberOfResults = request.results.count;
            self.results = [request.results copy];
            VNClassificationObservation *topResult = ((VNClassificationObservation *)(self.results[0]));
            VNClassificationObservation *result2 = ((VNClassificationObservation *)(self.results[1]));
            VNClassificationObservation *result3 = ((VNClassificationObservation *)(self.results[2]));
            self.messageLabel.text = [NSString stringWithFormat: @"%f: %@", topResult.confidence, topResult.identifier];
           self.obj2.text = [NSString stringWithFormat: @"%f: %@", result2.confidence, result2.identifier];
            self.obj3.text =  [NSString stringWithFormat: @"%f: %@", result3.confidence, result3.identifier];
          
        
            //self.
        });
        
    }];

    
    DJICamera* camera = [DemoComponentHelper fetchCamera];
    
    if (camera) {
        camera.delegate = self;
    }
    self.needToSetMode = YES;
    [[DJIVideoPreviewer instance] start];
    self.previewerAdapter = [VideoPreviewerSDKAdapter adapterWithDefaultSettings];
    [self.previewerAdapter start];
    if (([camera.displayName isEqualToString:DJICameraDisplayNameMavic2ZoomCamera] ||
         [camera.displayName isEqualToString:DJICameraDisplayNameMavic2ProCamera])) {
        [self.previewerAdapter setupFrameControlHandler];
    }
    [[DJIVideoPreviewer instance] registFrameProcessor:self];
    [DJIVideoPreviewer instance].enableHardwareDecode = 1 ;

}

-(void) viewWillAppear:(BOOL)animated
{
    [super viewWillAppear:animated];
    // self.fpvView.layer.sublayers = nil;
     [[DJIVideoPreviewer instance] setView:self.fpvView];
    
}

-(void) viewWillDisappear:(BOOL)animated
{
    [super viewWillDisappear:animated];
    // Call unSetView during exiting to release the memory
    [[DJIVideoPreviewer instance] unSetView];
    if (self.previewerAdapter) {
        [self.previewerAdapter stop];
        self.previewerAdapter = nil;
    }
}


-(void) onImageViewTap:(UIGestureRecognizer*)recognized{
    UIView* view = recognized.view;
    [view removeFromSuperview];
}
#pragma mark - DJICameraDelegate
-(void)camera:(DJICamera *)camera didUpdateSystemState:(DJICameraSystemState *)systemState
{
    if (systemState.mode == DJICameraModePlayback ||
        systemState.mode == DJICameraModeMediaDownload) {
        if (self.needToSetMode) {
            self.needToSetMode = NO;
            WeakRef(obj);
            [camera setMode:DJICameraModeShootPhoto withCompletion:^(NSError * _Nullable error) {
                if (error) {
                    WeakReturn(obj);
                    obj.needToSetMode = YES;
                }
            }];
        }
    }
}

-(void)camera:(DJICamera *)camera didUpdateTemperatureData:(float)temperature {
   // self.fpvTemperatureData.text = [NSString stringWithFormat:@"%f", temperature];
}

#pragma mark - VideoFrameProcessor
- (BOOL)videoProcessorEnabled {
    return YES;
}
# pragma mark Face Recognition
- (void)drawFaceRect:(CIImage*)image{
    
    VNDetectFaceLandmarksRequest *faceLandmarks = [VNDetectFaceLandmarksRequest new];
    VNSequenceRequestHandler *faceLandmarksDetectionRequest = [VNSequenceRequestHandler new];
    [faceLandmarksDetectionRequest performRequests:@[faceLandmarks] onCIImage:image error:nil];
    
    for(VNFaceObservation *observation in faceLandmarks.results){
        //draw rect on face
        CGRect boundingBox = observation.boundingBox;
        CGSize size = CGSizeMake(boundingBox.size.width * self.fpvView.bounds.size.width, boundingBox.size.height * self.fpvView.bounds.size.height);
        CGPoint origin = CGPointMake(boundingBox.origin.x * self.fpvView.bounds.size.width, (1-boundingBox.origin.y)*self.fpvView.bounds.size.height - size.height);
        
        CAShapeLayer *layer = [CAShapeLayer layer];
        
        layer.frame = CGRectMake(origin.x, origin.y, size.width, size.height);
        layer.borderColor = [UIColor redColor].CGColor;
        layer.borderWidth = 2;
        
        [self.fpvView.layer addSublayer:layer];
    }
}
- (void)detectFace:(CIImage*)image{
    //create req
    VNDetectFaceRectanglesRequest *faceDetectionReq = [VNDetectFaceRectanglesRequest new];
    NSDictionary *d = [[NSDictionary alloc] init];
    //req handler
    VNImageRequestHandler *handler = [[VNImageRequestHandler alloc] initWithCIImage:image options:d];
    //send req to handler
    [handler performRequests:@[faceDetectionReq] error:nil];
    
    //is there a face?
    for(VNFaceObservation *observation in faceDetectionReq.results){
        if(observation){
            CGRect boundingBox = observation.boundingBox;
            CGSize size = CGSizeMake(boundingBox.size.width * self.fpvView.bounds.size.width, boundingBox.size.height * self.fpvView.bounds.size.height);
            CGPoint origin = CGPointMake(boundingBox.origin.x * self.fpvView.bounds.size.width, (1-boundingBox.origin.y)*self.fpvView.bounds.size.height - size.height);
            
            CAShapeLayer *layer = [CAShapeLayer layer];
            
            layer.frame = CGRectMake(origin.x, origin.y, size.width, size.height);
            layer.borderColor = [UIColor redColor].CGColor;
            layer.borderWidth = 2;
         
            [self.fpvView.layer addSublayer:layer];
          
        }
    }
}
-(void) videoProcessFrame:(VideoFrameYUV*)frame {
    if ([DJIVideoPreviewer instance].enableHardwareDecode &&
        (frame->cv_pixelbuffer_fastupload != NULL)) {
        CVPixelBufferRef pixelBuffer = frame->cv_pixelbuffer_fastupload;
        if (self.currentPixelBuffer) {
            CVPixelBufferRelease(self.currentPixelBuffer);
        }
        self.currentPixelBuffer = pixelBuffer;
        CVPixelBufferRetain(pixelBuffer);
        CVImageBufferRef imageBuffer =  pixelBuffer;
        occurrences++;
        if (occurrences == 15){
            CIImage* ori_image = [[CIImage alloc] initWithCVPixelBuffer:imageBuffer options:nil];
            CIImage *image = [ori_image imageByApplyingTransform:(CGAffineTransformMakeScale(0.1, 0.1))];
            NSTimeInterval start = [[NSDate date] timeIntervalSince1970];
            [startTimes addObject: [NSNumber numberWithDouble: start]];
            NSArray *a = @[rq];
            NSDictionary *d = [[NSDictionary alloc] init];
            VNImageRequestHandler *handler = [[VNImageRequestHandler alloc] initWithCIImage:image options:d];
            dispatch_sync(dispatch_get_main_queue(), ^{
                [handler performRequests:a error:nil];
              //  [self detectFace:image];
                });
            occurrences = 0;
          
        }
    } else {
        
        self.currentPixelBuffer = nil;
        
    }
    
}

@end

