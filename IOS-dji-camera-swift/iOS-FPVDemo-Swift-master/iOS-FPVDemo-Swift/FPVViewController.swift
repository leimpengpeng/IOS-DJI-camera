//
//  FPVViewController.swift
//  iOS-FPVDemo-Swift
//

import UIKit
import DJISDK
import VideoPreviewer
import CoreML
import Vision
import CoreMedia

@available(iOS 11.0, *)
class FPVViewController: UIViewController,  DJIVideoFeedListener, DJISDKManagerDelegate, DJIBaseProductDelegate, DJICameraDelegate,VideoFrameProcessor {
    
    
    
    var isRecording : Bool!
    var camera : DJICamera!
    var occurance = 0
    let model = Inceptionv3()
    var requests = [VNCoreMLRequest]()
    var startTimes: [CFTimeInterval] = []
    let inputWidth = 299
    let inputHeight = 299
    var framesDone = 0
    var frameCapturingStartTime = CACurrentMediaTime()
    
    var inflightBuffer = 0
    let semaphore = DispatchSemaphore(value: FPVViewController.maxInflightBuffers)
    static let maxInflightBuffers = 3
    @IBOutlet var recordTimeLabel: UILabel!
    
    @IBOutlet var captureButton: UIButton!
    
    @IBOutlet var recordButton: UIButton!
    
    @IBOutlet var recordModeSegmentControl: UISegmentedControl!
    
    @IBOutlet var fpvView: UIView!
    @IBOutlet weak var debug: UILabel!
    @IBOutlet weak var predictionLabel: UILabel!
    
    @IBOutlet weak var timeLabel: UILabel!
    @IBOutlet weak var ImageView: UIImageView!
    override func viewDidAppear(_ animated: Bool) {
        super.viewDidAppear(animated)
        VideoPreviewer.instance()?.start()
        VideoPreviewer.instance().setView(self.fpvView)
        DJISDKManager.registerApp(with: self)
        debug.text = "true"
    }
    func setUpVision() {
        guard let visionModel = try? VNCoreMLModel(for: model.model) else {
            print("Error: could not create Vision model")
            return
        }
        
        for _ in 0..<FPVViewController.maxInflightBuffers {
            let request = VNCoreMLRequest(model: visionModel, completionHandler: visionRequestDidComplete)
            request.imageCropAndScaleOption = .centerCrop
            requests.append(request)
        }
    }
    // MARK: - Doing inference
      typealias Prediction = (String, Double)
    func predict(pixelBuffer: CVPixelBuffer) {
        debug.text = "true5"
        // Measure how long it takes to predict a single video frame.
        let startTime = CACurrentMediaTime()
        
        // Resize the input using vImage.
        if let resizedPixelBuffer = resizePixelBuffer(pixelBuffer,
                                                      width: inputWidth,
                                                      height: inputHeight) {
            debug.text = "true6"
            // Give the resized input to our model.
            if let prediction = try? model.prediction(image: resizedPixelBuffer) {
                let top5 = top(5, prediction.classLabelProbs)
                let elapsed = CACurrentMediaTime() - startTime
                
                DispatchQueue.main.async {
                    self.show(results: top5, elapsed: elapsed)
                }
            } else {
                print("BOGUS")
            }
        }
        self.semaphore.signal()
    }
    
    func predictUsingVision(pixelBuffer: CVPixelBuffer) {
        // Measure how long it takes to predict a single video frame. Note that
        // predict() can be called on the next frame while the previous one is
        // still being processed. Hence the need to queue up the start times.
        startTimes.append(CACurrentMediaTime())
        debug.text = "vtrue6"
        // Vision will automatically resize the input image.
        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer)
        let request = requests[inflightBuffer]
        
        // For better throughput, we want to schedule multiple Vision requests
        // in parllel. These need to be separate instances, and inflightBuffer
        // is the index of the current request object to use.
        inflightBuffer += 1
        if inflightBuffer >= FPVViewController.maxInflightBuffers {
            inflightBuffer = 0
        }
        
        // Because perform() will block until after the request completes, we
        // run it on a concurrent background queue, so that the next frame can
        // be scheduled in parallel with this one.
        DispatchQueue.global().async {
            try? handler.perform([request])
        }
    }
    public func top(_ k: Int, _ prob: [String: Double]) -> [(String, Double)] {
        return Array(prob.map { x in (x.key, x.value) }
            .sorted(by: { a, b -> Bool in a.1 > b.1 })
            .prefix(min(k, prob.count)))
    }
  
    func visionRequestDidComplete(request: VNRequest, error: Error?) {
        if let observations = request.results as? [VNClassificationObservation] {
            
            // The observations appear to be sorted by confidence already, so we
            // take the top 5 and map them to an array of (String, Double) tuples.
            let top5 = observations.prefix(through: 4)
                .map { ($0.identifier, Double($0.confidence)) }
            
            let elapsed = CACurrentMediaTime() - startTimes.remove(at: 0)
            
            DispatchQueue.main.async {
                self.show(results: top5, elapsed: elapsed)
            }
        }
        
      
    }
    
    func show(results: [Prediction], elapsed: CFTimeInterval) {
        debug.text = "true7"
        var s: [String] = []
        for (i, pred) in results.enumerated() {
            s.append(String(format: "%d: %@ (%3.2f%%)", i + 1, pred.0, pred.1 * 100))
        }
        predictionLabel.text = s.joined(separator: "\n\n")
        
        let fps = self.measureFPS()
        timeLabel.text = String(format: "Elapsed %.5f seconds - %.2f FPS", elapsed, fps)
    }
    
    func measureFPS() -> Double {
        // Measure how many frames were actually delivered per second.
        framesDone += 1
        let frameCapturingElapsed = CACurrentMediaTime() - frameCapturingStartTime
        let currentFPSDelivered = Double(framesDone) / frameCapturingElapsed
        if frameCapturingElapsed > 1 {
            framesDone = 0
            frameCapturingStartTime = CACurrentMediaTime()
        }
        return currentFPSDelivered
    }
    override func viewWillDisappear(_ animated: Bool) {
        super.viewWillDisappear(animated)
        
        VideoPreviewer.instance().setView(nil)
        DJISDKManager.videoFeeder()?.primaryVideoFeed.remove(self)

    }
    
    override func viewDidLoad() {
        super.viewDidLoad()
       
       // VideoPreviewer.instance().enableHardwareDecode = true

      
        VideoPreviewer.instance()?.registFrameProcessor(self)
      
        predictionLabel.text = "123"
        timeLabel.text = ""
        
        setUpVision()
        frameCapturingStartTime = CACurrentMediaTime()
        
        recordTimeLabel.isHidden = true
    }

    override func didReceiveMemoryWarning() {
        super.didReceiveMemoryWarning()
    }
    
    //
    //  Helpers
    //
    func createPixelBuffer(fromFrame frame: VideoFrameYUV) -> CVPixelBuffer? {
        var initialPixelBuffer: CVPixelBuffer?
        let _: CVReturn = CVPixelBufferCreate(kCFAllocatorDefault, Int(frame.width), Int(frame.height), kCVPixelFormatType_420YpCbCr8Planar, nil, &initialPixelBuffer)
        
        guard let pixelBuffer = initialPixelBuffer,
            CVPixelBufferLockBaseAddress(pixelBuffer, []) == kCVReturnSuccess
            else {
                return nil
        }
        
        let yPlaneWidth = CVPixelBufferGetWidthOfPlane(pixelBuffer, 0)
        let yPlaneHeight = CVPixelBufferGetHeightOfPlane(pixelBuffer, 0)
        
        let uPlaneWidth = CVPixelBufferGetWidthOfPlane(pixelBuffer, 1)
        let uPlaneHeight = CVPixelBufferGetHeightOfPlane(pixelBuffer, 1)
        
        let vPlaneWidth = CVPixelBufferGetWidthOfPlane(pixelBuffer, 2)
        let vPlaneHeight = CVPixelBufferGetHeightOfPlane(pixelBuffer, 2)
        
        let yDestination = CVPixelBufferGetBaseAddressOfPlane(pixelBuffer, 0)
        memcpy(yDestination, frame.luma, yPlaneWidth * yPlaneHeight)
        
        let uDestination = CVPixelBufferGetBaseAddressOfPlane(pixelBuffer, 1)
        memcpy(uDestination, frame.chromaB, uPlaneWidth * uPlaneHeight)
        
        let vDestination = CVPixelBufferGetBaseAddressOfPlane(pixelBuffer, 2)
        memcpy(vDestination, frame.chromaR, vPlaneWidth * vPlaneHeight)
        
        
        
        
        CVPixelBufferUnlockBaseAddress(pixelBuffer, [])
        
        return pixelBuffer
    }
    
    func createImageUI(fromFrame frame: VideoFrameYUV) -> UIImage? {
        var initialPixelBuffer: CVPixelBuffer?
        let _: CVReturn = CVPixelBufferCreate(kCFAllocatorDefault, Int(frame.width), Int(frame.height), kCVPixelFormatType_420YpCbCr8Planar, nil, &initialPixelBuffer)
        
        guard let pixelBuffer = initialPixelBuffer,
            CVPixelBufferLockBaseAddress(pixelBuffer, []) == kCVReturnSuccess
            else {
                return nil
        }
        
        let yPlaneWidth = CVPixelBufferGetWidthOfPlane(pixelBuffer, 0)
        let yPlaneHeight = CVPixelBufferGetHeightOfPlane(pixelBuffer, 0)
        
        let uPlaneWidth = CVPixelBufferGetWidthOfPlane(pixelBuffer, 1)
        let uPlaneHeight = CVPixelBufferGetHeightOfPlane(pixelBuffer, 1)
        
        let vPlaneWidth = CVPixelBufferGetWidthOfPlane(pixelBuffer, 2)
        let vPlaneHeight = CVPixelBufferGetHeightOfPlane(pixelBuffer, 2)
        
        let yDestination = CVPixelBufferGetBaseAddressOfPlane(pixelBuffer, 0)
        memcpy(yDestination, frame.luma, yPlaneWidth * yPlaneHeight)
        
        let uDestination = CVPixelBufferGetBaseAddressOfPlane(pixelBuffer, 1)
        memcpy(uDestination, frame.chromaB, uPlaneWidth * uPlaneHeight)
        
        let vDestination = CVPixelBufferGetBaseAddressOfPlane(pixelBuffer, 2)
        memcpy(vDestination, frame.chromaR, vPlaneWidth * vPlaneHeight)
        
        
        let width: Int = CVPixelBufferGetWidth(pixelBuffer)
        let height: Int = CVPixelBufferGetHeight(pixelBuffer)
        let bytesPerRow: Int = CVPixelBufferGetBytesPerRow(pixelBuffer)
        let lumaBuffer = CVPixelBufferGetBaseAddress(pixelBuffer)
        let grayColorSpace: CGColorSpace = CGColorSpaceCreateDeviceGray()
        let context: CGContext = CGContext(data: lumaBuffer, width: width, height: height, bitsPerComponent: 8, bytesPerRow: bytesPerRow , space: grayColorSpace, bitmapInfo: CGImageAlphaInfo.none.rawValue)!
        let dstImageFilter: CGImage = context.makeImage()!
        let imageRect : CGRect = CGRect(x: 0, y: 0, width: width, height: height)
        context.draw(dstImageFilter, in: imageRect)
        let image = UIImage(cgImage: dstImageFilter)
        
        
        CVPixelBufferUnlockBaseAddress(pixelBuffer, [])
        
        return image
    }
    func videoProcessorEnabled() -> Bool {
        return true
    }
    
    func videoProcessFrame(_ frame: UnsafeMutablePointer<VideoFrameYUV>!) {
        let pixelBuffer = createPixelBuffer (fromFrame:frame.pointee)
       // let image = createImageUI (fromFrame:frame.pointee)
        if (pixelBuffer != nil){
            debug.text = "true3"
            if (occurance == 30){
           debug.text = "true4"
           // let imageView = UIImageView(image: image)
           // imageView.frame = CGRect(x: 0, y: 0, width: 100, height: 200)
            //
            DispatchQueue.main.sync(execute: {() -> Void in
             // self.ImageView.addSubview(imageView)
                self.predictUsingVision(pixelBuffer: pixelBuffer!)
                
            })
                occurance = 0
            }
            occurance = occurance + 1


        }
        else {
            debug.text = "false3"
        }
        
    }
    
    func videoProcessFailedFrame() {
        
        debug.text = "false"
    }
    func fetchCamera() -> DJICamera? {
        let product = DJISDKManager.product()
        
        if (product == nil) {
            return nil
        }
        
        if (product!.isKind(of: DJIAircraft.self)) {
            return (product as! DJIAircraft).camera
        } else if (product!.isKind(of: DJIHandheld.self)) {
            return (product as! DJIHandheld).camera
        }
        
        return nil
    }
    
    func formatSeconds(seconds: UInt) -> String {
        let date = Date(timeIntervalSince1970: TimeInterval(seconds))
        
        let dateFormatter = DateFormatter()
        dateFormatter.dateFormat = "mm:ss"
        
        return(dateFormatter.string(from: date))
    }
    
    //
    //  DJIBaseProductDelegate
    //
    
    func productConnected(_ product: DJIBaseProduct?) {
        
        NSLog("Product Connected")
        
        
        if (product != nil) {
            product!.delegate = self
            
            camera = self.fetchCamera()
            
            if (camera != nil) {
                camera!.delegate = self
                
                VideoPreviewer.instance().start()

            }
        }
    }
    
    func productDisconnected() {
        
        NSLog("Product Disconnected")

        camera = nil
        
        VideoPreviewer.instance().clearVideoData()
        VideoPreviewer.instance().close()
        
    }
    
    //
    //  DJISDKManagerDelegate
    //
    
    func appRegisteredWithError(_ error: Error?) {
        
        if (error != nil) {
            NSLog("Register app failed! Please enter your app key and check the network.")
        } else {
            NSLog("Register app succeeded!")
        }
        
        DJISDKManager.startConnectionToProduct()
        DJISDKManager.videoFeeder()?.primaryVideoFeed.add(self, with: nil)
        
    }
    
    //
    //  DJICameraDelegate
    //
    
    func camera(_ camera: DJICamera, didUpdate cameraState: DJICameraSystemState) {
        self.isRecording = cameraState.isRecording
        self.recordTimeLabel.isHidden = !self.isRecording
        
        self.recordTimeLabel.text = formatSeconds(seconds: cameraState.currentVideoRecordingTimeInSeconds)
        
        if (self.isRecording == true) {
            self.recordButton.setTitle("Stop Record", for: UIControlState.normal)
        } else {
            self.recordButton.setTitle("Start Record", for: UIControlState.normal)
        }
        
        if (cameraState.mode == DJICameraMode.shootPhoto) {
            self.recordModeSegmentControl.selectedSegmentIndex = 0
        } else {
            self.recordModeSegmentControl.selectedSegmentIndex = 1
        }
        
    }
    
    //
    //  DJIVideoFeedListener
    //
    
    func videoFeed(_ videoFeed: DJIVideoFeed, didUpdateVideoData rawData: Data) {
        
        let videoData = rawData as NSData
        let videoBuffer = UnsafeMutablePointer<UInt8>.allocate(capacity: videoData.length)
        
        videoData.getBytes(videoBuffer, length: videoData.length)
        
        
        VideoPreviewer.instance().push(videoBuffer, length: Int32(videoData.length))
    }
    
    //
    //  IBAction Methods
    //    
    
    @IBAction func captureAction(_ sender: UIButton) {
       
        if (camera != nil) {
            camera.setMode(DJICameraMode.shootPhoto, withCompletion: { (error) in
                
                if (error != nil) {
                    NSLog("Set Photo Mode Error: " + String(describing: error))
                }
            
                self.camera.startShootPhoto(completion: { (error) in
                    if (error != nil) {
                        NSLog("Shoot Photo Mode Error: " + String(describing: error))
                    }
                })
            })
        }
    }
    
    @IBAction func recordAction(_ sender: UIButton) {
        
        if (camera != nil) {
            if (self.isRecording) {
                camera.stopRecordVideo(completion: { (error) in
                    if (error != nil) {
                        NSLog("Stop Record Video Error: " + String(describing: error))
                    }
                })
            } else {
                camera.setMode(DJICameraMode.recordVideo,  withCompletion: { (error) in
                    
                    self.camera.startRecordVideo(completion: { (error) in
                        if (error != nil) {
                            NSLog("Stop Record Video Error: " + String(describing: error))
                        }
                    })
                })
            }
        }
    }
    
    
    @IBAction func recordModeSegmentChange(_ sender: UISegmentedControl) {
        
        if (camera != nil) {
            if (sender.selectedSegmentIndex == 0) {
                camera.setMode(DJICameraMode.shootPhoto,  withCompletion: { (error) in
                    
                })
                
            } else if (sender.selectedSegmentIndex == 1) {
                camera.setMode(DJICameraMode.recordVideo,  withCompletion: { (error) in
                    
                })
                
                
            }
        }
    }
    

}
